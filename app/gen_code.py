# app/gen_code.py
from __future__ import annotations
"""
Codegen track:
- Collect candidate CSV paths (uploaded + deterministic/duckdb outputs)
- Ask LLM to write safe Python that prints EXACTLY one JSON array
- Run code in a sandboxed exec (whitelisted libs)
- Persist code, stdout, and answers.json
- Return {"array":[...]} or {} on failure
"""

import os
import io
import re
import json
import glob
import time
import base64
import builtins
from typing import Any, Dict, List, Optional

# Try optional providers
def _load_openai():
    try:
        import openai  # pip install openai>=1.0
        return openai
    except Exception:
        return None

def _load_gemini():
    try:
        import google.generativeai as genai  # pip install google-generativeai
        return genai
    except Exception:
        return None

# -------------------- helpers --------------------

def _outdir(workdir: str) -> str:
    d = os.path.join(workdir, f"codegen_run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(d, exist_ok=True)
    return d

def _read_prompt(path: str, fallback: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return fallback

def _write(path: str, data: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)

def _save_bytes(path: str, b: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b)

def _list_candidate_csvs(workdir: str) -> List[str]:
    # deterministic / duckdb outputs
    cand = []
    # deterministic scraped_data tables
    cand += glob.glob(os.path.join(workdir, "deterministic_run_*", "scraped_data", "*.csv"))
    # duckdb materialized tables (if you add this track)
    cand += glob.glob(os.path.join(workdir, "duckdb_run_*", "table_*.csv"))
    # keep only first few to stay small
    # choose most recent deterministic first
    cand = sorted(cand, key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[:4]

def _write_uploaded_to_csvs(sources: Dict[str, Any], out_dir: str) -> List[str]:
    import pandas as pd
    csvs: List[str] = []
    inc_dir = os.path.join(out_dir, "incoming")
    os.makedirs(inc_dir, exist_ok=True)
    for name, meta in (sources.get("files") or {}).items():
        b = meta.get("bytes", b"")
        if not b:
            continue
        low = (name or "").lower()
        if low.endswith(".csv"):
            p = os.path.join(inc_dir, os.path.basename(name))
            _save_bytes(p, b)
            csvs.append(p)
        elif low.endswith(".json"):
            # Try array of objects → DataFrame
            try:
                obj = json.loads(b.decode("utf-8", errors="replace"))
                import pandas as pd
                if isinstance(obj, list):
                    df = pd.DataFrame(obj)
                elif isinstance(obj, dict):
                    df = pd.DataFrame(obj)
                else:
                    continue
                p = os.path.join(inc_dir, os.path.basename(name) + ".csv")
                df.to_csv(p, index=False)
                csvs.append(p)
            except Exception:
                pass
        elif low.endswith(".xlsx") or low.endswith(".xls"):
            try:
                df = pd.read_excel(io.BytesIO(b))
                p = os.path.join(inc_dir, os.path.splitext(os.path.basename(name))[0] + ".csv")
                df.to_csv(p, index=False)
                csvs.append(p)
            except Exception:
                pass
        # ignore other types for codegen
    return csvs

def _extract_code_block(txt: str) -> Optional[str]:
    # Prefer fenced ```python blocks
    m = re.search(r"```python\s+([\s\S]*?)```", txt, re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*([\s\S]*?)```", txt)
    if m:
        return m.group(1).strip()
    # Fall back to raw text
    return txt.strip() if txt.strip() else None

def _capture_last_json_array(stdout_text: str) -> Optional[List[Any]]:
    # Find the last bracketed array; robust to extra prints
    cand = None
    for m in re.finditer(r"\[[\s\S]*\]", stdout_text):
        cand = m.group(0)
    if not cand:
        return None
    try:
        arr = json.loads(cand)
        return arr if isinstance(arr, list) else None
    except Exception:
        return None

# -------------------- sandbox runner --------------------

def _run_code_in_sandbox(code: str, globals_in: Dict[str, Any]) -> str:
    """
    Execute user code with a restricted global namespace.
    We allow common libs used in the prompt; forbid dangerous builtins.
    """
    import sys
    import types
    import numpy as np
    import pandas as pd
    import duckdb
    import networkx as nx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import statistics
    import collections
    import math
    import re as _re
    import json as _json
    import base64 as _b64
    import io as _io

    # Safe builtins
    SAFE_BUILTINS = {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "range": range, "enumerate": enumerate, "zip": zip, "sorted": sorted,
        "map": map, "filter": filter, "all": all, "any": any, "next": next,
        "isinstance": isinstance, "print": print, "float": float, "int": int, "str": str,
        "list": list, "dict": dict, "set": set, "tuple": tuple
    }

    # Module namespace (pre-imported)
    sandbox_globals = {
        "__builtins__": SAFE_BUILTINS,
        "np": np, "numpy": np,
        "pd": pd, "pandas": pd,
        "duckdb": duckdb,
        "nx": nx, "networkx": nx,
        "matplotlib": matplotlib, "plt": plt,
        "statistics": statistics,
        "collections": collections,
        "math": math,
        "re": _re,
        "json": _json,
        "base64": _b64,
        "io": _io,
    }
    # Inject user variables
    sandbox_globals.update(globals_in)

    # Capture stdout
    buf = io.StringIO()
    old_stdout = os.sys.stdout
    try:
        os.sys.stdout = buf
        exec(compile(code, "<codegen>", "exec"), sandbox_globals, None)
    finally:
        os.sys.stdout = old_stdout
    return buf.getvalue()

# -------------------- providers --------------------

def _ask_openai_code(prompt: str, csv_paths: List[str], questions: List[str], qtypes: List[str]) -> Optional[str]:
    openai = _load_openai()
    if openai is None:
        return None
    try:
        client = openai.OpenAI()
        msgs = [
            {"role": "system", "content": "You are a meticulous data analyst who ONLY outputs Python code when asked."},
            {"role": "user", "content": f"{prompt}\n\nCSV_PATHS:\n{json.dumps(csv_paths, indent=2)}\n\nQUESTIONS:\n{json.dumps(questions, ensure_ascii=False)}\nQTYPES:\n{json.dumps(qtypes)}\nReturn ONLY Python code."},
        ]
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0,
        )
        txt = resp.choices[0].message.content or ""
        return _extract_code_block(txt)
    except Exception:
        return None

def _ask_gemini_code(prompt: str, csv_paths: List[str], questions: List[str], qtypes: List[str]) -> Optional[str]:
    genai = _load_gemini()
    if genai is None:
        return None
    try:
        genai.configure()  # uses GOOGLE_API_KEY
        model = genai.GenerativeModel("gemini-1.5-flash")
        parts = [
            "Return ONLY Python code (no prose).",
            prompt,
            f"CSV_PATHS:\n{json.dumps(csv_paths, indent=2)}",
            f"QUESTIONS:\n{json.dumps(questions, ensure_ascii=False)}",
            f"QTYPES:\n{json.dumps(qtypes)}",
        ]
        out = model.generate_content(parts)
        txt = (out.text or "").strip()
        return _extract_code_block(txt)
    except Exception:
        return None

# -------------------- main entry --------------------

async def generate_and_run(
    questions: List[str],
    qtypes: List[str],
    sources: Dict[str, Any],
    workdir: str,
    budget_ms: int,
) -> Dict[str, Any]:
    out_dir = _outdir(workdir)

    # 1) Gather candidate CSVs
    csv_paths = _write_uploaded_to_csvs(sources, out_dir)
    csv_paths += _list_candidate_csvs(workdir)
    # dedupe (preserve order)
    seen = set(); csv_paths = [p for p in csv_paths if not (p in seen or seen.add(p))]

    # If nothing to work with, bail early
    if not csv_paths:
        _write(os.path.join(out_dir, "answers.json"), json.dumps({}, indent=2))
        return {}

    # 2) Load prompt
    default_prompt = """You write SAFE Python to answer questions from provided CSV(s). You do NOT browse or delete files.

CONTEXT:
- You may receive one or more CSV file paths (from uploaded data, scraped tables, or materialization).
- Choose the most relevant CSV(s) based on column names/content.
- Always produce a SINGLE JSON array with one element per question, IN ORDER.
- If a question asks for a plot, return a base64 PNG data URI under 100,000 bytes: "data:image/png;base64,...".
- If insufficient information exists, return null for that position.

LIBS ALLOWED:
- pandas, numpy, duckdb, networkx, matplotlib (Agg backend), io, base64, json, re, math, statistics, collections
- DISALLOWED: subprocess, file deletion, network access.

REQUIRED BEHAVIOR:
1) Parse currency-like strings robustly. Support $/commas and units: billion/bn, million/m, thousand/k.
2) For correlations: ensure numeric columns, drop NaNs, return float in [-1,1] rounded to 6 decimals.
3) For scatter+regression plot: dotted red regression line; label axes; compress PNG under 100 kB.
4) For "earliest" type with a threshold, pick the earliest year/date that meets the numeric threshold.
5) For "count" with threshold and optional date filters (before/after YYYY), honor filters.
6) If any question appears graph-like (edge count, highest/average degree, density, shortest path X-Y, graph plot, degree histogram):
   - Load edge list from a CSV that contains either columns ['source','target'] or use the first two non-identical columns as edges.
   - Use networkx (Graph, undirected) to compute:
     - edge_count: number of edges
     - highest_degree_node: node with max degree (break ties lexicographically)
     - average_degree: mean degree (round(6))
     - density: nx.density(G) (round(6))
     - shortest_path_A_B: list of node IDs between nodes A and B if reachable, else null
     - graph_plot: draw spring_layout, small nodes/labels, save as PNG <100k
     - degree_histogram: histogram of degrees; PNG <100k
7) PRINT exactly one JSON array string to stdout. No explanations, keys, or extra prints.
"""
    prompt = _read_prompt(os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "codegen_prompt.txt"), default_prompt)

    # 3) Ask providers (OpenAI → Gemini fallback)
    code = _ask_openai_code(prompt, csv_paths, questions, qtypes)
    if code is None:
        code = _ask_gemini_code(prompt, csv_paths, questions, qtypes)

    code_path = os.path.join(out_dir, "code.py")
    _write(code_path, code or "# failed to generate code\nprint([])")

    if code is None:
        # No model available or failed → bail gracefully
        _write(os.path.join(out_dir, "answers.json"), json.dumps({}, indent=2))
        return {}

    # 4) Run code in sandbox
    try:
        stdout_text = _run_code_in_sandbox(code, {
            "csv_paths": csv_paths,
            "questions": questions,
            "qtypes": qtypes,
        })
    except Exception as e:
        _write(os.path.join(out_dir, "stdout.txt"), f"[error] {e}")
        _write(os.path.join(out_dir, "answers.json"), json.dumps({}, indent=2))
        return {}

    _write(os.path.join(out_dir, "stdout.txt"), stdout_text)

    # 5) Capture last JSON array
    arr = _capture_last_json_array(stdout_text) or []
    # Persist
    _write(os.path.join(out_dir, "answers.json"), json.dumps({"array": arr}, indent=2))
    return {"array": arr}
