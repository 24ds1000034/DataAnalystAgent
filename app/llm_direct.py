# app/llm_direct.py
from __future__ import annotations
"""
LLM-direct track:
- Sample a small CSV snippet (uploaded or deterministic output)
- Ask LLM to answer STRICTLY from this snippet
- Return {"array":[...]} or {} when insufficient
"""

import os
import io
import csv
import glob
import json
import time
from typing import Any, Dict, List, Optional

def _outdir(workdir: str) -> str:
    d = os.path.join(workdir, f"direct_llm_run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(d, exist_ok=True)
    return d

def _load_openai():
    try:
        import openai
        return openai
    except Exception:
        return None

def _load_gemini():
    try:
        import google.generativeai as genai
        return genai
    except Exception:
        return None

def _pick_csv(workdir: str, sources: Dict[str, Any]) -> Optional[str]:
    inc = []
    for name, meta in (sources.get("files") or {}).items():
        if (name or "").lower().endswith(".csv") and meta.get("bytes"):
            p = os.path.join(workdir, "direct_snippets", os.path.basename(name))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(meta["bytes"])
            inc.append(p)
    if inc:
        return inc[0]
    # Fallback to deterministic scraped tables
    found = glob.glob(os.path.join(workdir, "deterministic_run_*", "scraped_data", "*.csv"))
    return found[0] if found else None

def _read_head(path: str, max_rows: int = 60, max_chars: int = 25000) -> str:
    out_lines = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                out_lines.append(line.rstrip("\n"))
                if i >= max_rows:
                    break
    except Exception:
        return ""
    txt = "\n".join(out_lines)
    return txt[:max_chars]

def _read_prompt(path: str, fallback: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return fallback

async def try_answer_on_snippet(
    questions: List[str],
    sources: Dict[str, Any],
    workdir: str,
    budget_ms: int
) -> Dict[str, Any]:
    out_dir = _outdir(workdir)
    csv_path = _pick_csv(workdir, sources)
    if not csv_path:
        with open(os.path.join(out_dir, "answers.json"), "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)
        return {}

    sample = _read_head(csv_path)
    with open(os.path.join(out_dir, "sample.csv"), "w", encoding="utf-8") as f:
        f.write(sample)

    default_prompt = """You are given a small CSV snippet and a list of questions.
Answer STRICTLY using only the provided snippet. If the snippet lacks the info, respond null at that position.
Return ONLY a JSON array with one element per question, in order. No explanations or extra text.

Rules:
- Do not guess beyond the snippet.
- If a numeric value needs units normalization, explain nothing—just compute and return the number.
- If a question asks for a plot, respond null (this track does not produce images).
"""
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "direct_on_snippet_prompt.txt")
    prompt = _read_prompt(prompt_path, default_prompt)

    msgs_openai = [
        {"role": "system", "content": "Strict snippet-only QA. Output JSON array only."},
        {"role": "user", "content": f"{prompt}\n\nCSV SNIPPET (first lines):\n```csv\n{sample}\n```\n\nQUESTIONS:\n{json.dumps(questions, ensure_ascii=False)}"},
    ]

    # Provider 1: OpenAI
    arr = None
    openai = _load_openai()
    if openai is not None:
        try:
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=msgs_openai,
                temperature=0,
            )
            txt = (resp.choices[0].message.content or "").strip()
            # Grab last JSON array
            import re, json as _json
            cand = None
            for m in re.finditer(r"\[[\s\S]*\]", txt):
                cand = m.group(0)
            if cand:
                val = _json.loads(cand)
                if isinstance(val, list):
                    arr = val
        except Exception:
            arr = None

    # Provider 2: Gemini fallback
    if arr is None:
        genai = _load_gemini()
        if genai is not None:
            try:
                genai.configure()
                model = genai.GenerativeModel("gemini-1.5-flash")
                out = model.generate_content([
                    "Return ONLY a JSON array. No prose.",
                    f"{prompt}\n\nCSV SNIPPET:\n```csv\n{sample}\n```",
                    f"QUESTIONS:\n{json.dumps(questions, ensure_ascii=False)}",
                ])
                txt = (out.text or "").strip()
                import re, json as _json
                cand = None
                for m in re.finditer(r"\[[\s\S]*\]", txt):
                    cand = m.group(0)
                if cand:
                    val = _json.loads(cand)
                    if isinstance(val, list):
                        arr = val
            except Exception:
                arr = None

    if arr is None:
        with open(os.path.join(out_dir, "answers.json"), "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)
        return {}

    with open(os.path.join(out_dir, "answers.json"), "w", encoding="utf-8") as f:
        json.dump({"array": arr}, f, indent=2)
    return {"array": arr}
