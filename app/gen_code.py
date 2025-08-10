from __future__ import annotations
import os, json, time, subprocess, sys, glob
from typing import Any, Dict, List, Optional
import pandas as pd

PROMPT = """You are to write executable Python code to answer the given questions using a single CSV.
Rules:
- Use pandas and numpy; matplotlib if a scatter plot is requested (dotted red regression line).
- Load the CSV at path CSV_PATH provided below.
- Compute answers in order; print a JSON array to stdout (and nothing else).
- For any plot, return a base64 PNG data URI under 100,000 bytes.
- No network access; no file deletion; do not read other files.

Inputs:
CSV_PATH: {csv_path}
Columns: {columns}
Questions:
{questions}
"""

def _outdir(base: str) -> str:
    ts=time.strftime("%Y%m%d_%H%M%S"); d=os.path.join(base, f"codegen_run_{ts}")
    os.makedirs(d, exist_ok=True); return d

def _latest_csv(workdir: str) -> Optional[str]:
    cand=sorted(glob.glob(os.path.join(workdir,"deterministic_run_*","scraped_data","table_*.csv")), reverse=True)
    return cand[0] if cand else None

def _load_keys():
    try:
        from dotenv import load_dotenv; load_dotenv()
    except: pass
    return os.getenv("OPENAI_API_KEY"), os.getenv("GOOGLE_API_KEY")

def _ask_openai_for_code(api_key: str, csv_path: str, columns: List[str], questions: List[str]) -> Optional[str]:
    try:
        import openai
        client=openai.OpenAI(api_key=api_key)
        resp=client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"Write ONLY Python code."},
                      {"role":"user","content":PROMPT.format(csv_path=csv_path, columns=columns, questions="\n".join(f"{i+1}. {q}" for i,q in enumerate(questions)))}],
            temperature=0
        )
        return resp.choices[0].message.content
    except Exception:
        return None

def _ask_gemini_for_code(api_key: str, csv_path: str, columns: List[str], questions: List[str]) -> Optional[str]:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model=genai.GenerativeModel("gemini-1.5-flash")
        out=model.generate_content(
            [{"role":"user","parts":[PROMPT.format(csv_path=csv_path, columns=columns, questions="\n".join(f"{i+1}. {q}" for i,q in enumerate(questions)))]}],
        )
        return out.text
    except Exception:
        return None

def _extract_code(text: str) -> Optional[str]:
    if not text: return None
    import re
    m=re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    return (m.group(1) if m else text).strip()

def _run_code(py_path: str, timeout=25) -> Optional[List[Any]]:
    try:
        proc=subprocess.run([sys.executable, py_path], capture_output=True, timeout=timeout, check=False, text=True)
        out=proc.stdout.strip()
        import json as _j
        return _j.loads(out) if out.startswith("[") else None
    except Exception:
        return None

async def run_codegen(questions: List[str], workdir: str, budget_ms: int) -> Dict[str, Any]:
    out_dir=_outdir(workdir)
    csv=_latest_csv(workdir)
    if not csv or not os.path.exists(csv):
        with open(os.path.join(out_dir,"notes.json"),"w") as f: json.dump({"status":"no-csv"}, f, indent=2)
        return {}
    try:
        df=pd.read_csv(csv)
    except Exception as e:
        with open(os.path.join(out_dir,"notes.json"),"w") as f: json.dump({"status":"csv-error","error":str(e)}, f, indent=2)
        return {}
    columns=list(df.columns)
    oai, gem = _load_keys()
    code=None; prov="none"
    if oai:
        prov="openai"; code=_extract_code(_ask_openai_for_code(oai, csv, columns, questions) or "")
    if not code and gem:
        prov="gemini"; code=_extract_code(_ask_gemini_for_code(gem, csv, columns, questions) or "")
    if not code:
        with open(os.path.join(out_dir,"codegen.json"),"w") as f: json.dump({"provider":prov,"status":"no-code"}, f, indent=2)
        return {}

    py_path=os.path.join(out_dir, "code_llm.py")
    with open(py_path,"w",encoding="utf-8") as f: f.write(code)

    answers=_run_code(py_path, timeout=min(25, max(10, budget_ms//1000)))
    if answers is None:
        with open(os.path.join(out_dir,"codegen.json"),"w") as f: json.dump({"provider":prov,"status":"run-failed"}, f, indent=2)
        return {}
    with open(os.path.join(out_dir,"code_output.json"),"w",encoding="utf-8") as f: json.dump(answers, f, indent=2)
    if len(answers)!=len(questions):
        answers=(answers+[None]*len(questions))[:len(questions)]
    return {"array": answers}
