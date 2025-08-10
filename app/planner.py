from __future__ import annotations
import os, re, json, time
from typing import Any, Dict, List, Optional

_URL_RE = re.compile(r"https?://[^\s)]+", re.I)
_S3_RE  = re.compile(r"s3://[^\s)]+", re.I)
_SQL_BLOCK = re.compile(r"```sql\s+([\s\S]*?)```", re.I)

def _outdir(base: str) -> str:
    d = os.path.join(base, f"planner_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(d, exist_ok=True); return d

def _load_keys():
    try:
        from dotenv import load_dotenv; load_dotenv()
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY"), os.getenv("GOOGLE_API_KEY")

def _extract_urls(text: str) -> List[str]:
    urls = _URL_RE.findall(text or "")
    urls += _S3_RE.findall(text or "")
    seen=set(); return [u for u in urls if not (u in seen or seen.add(u))]

def _find_sql_block(txt: str) -> Optional[str]:
    m = _SQL_BLOCK.search(txt or "")
    if m:
        sql = m.group(1).strip()
        return sql if sql else None
    return None

def _ask_openai(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        import openai, json as _j
        client = openai.OpenAI()
        sys = open("prompts/planner_prompt.txt","r",encoding="utf-8").read()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":_j.dumps(payload, ensure_ascii=False)}],
            temperature=0, response_format={"type":"json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return None

def _ask_gemini(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        import google.generativeai as genai, re as _re, json as _j
        genai.configure()
        sys = open("prompts/planner_prompt.txt","r",encoding="utf-8").read()
        model = genai.GenerativeModel("gemini-1.5-flash")
        out = model.generate_content([sys, _j.dumps(payload, ensure_ascii=False)])
        txt = (out.text or "").strip()
        m = _re.search(r"\{[\s\S]*\}\s*$", txt)
        return _j.loads(m.group(0)) if m else None
    except Exception:
        return None

def _fallback(payload: Dict[str, Any]) -> Dict[str, Any]:
    uploaded = payload.get("uploaded_files") or []
    urls = payload.get("detected_urls") or []
    s3_urls = [u for u in urls if u.startswith("s3://")]
    http_urls=[u for u in urls if u.startswith("http")]
    targets=[]
    if uploaded:
        targets.append({"why":"uploaded file present","strategy":"uploaded_file","url":None,"max_tables":2,"selectors":[],"sql":None})
    if s3_urls:
        sql = payload.get("sql_block") or f"SELECT * FROM read_parquet('{s3_urls[0]}') LIMIT 5000"
        targets.append({"why":"s3 parquet referenced","strategy":"duckdb_parquet","url":s3_urls[0],"max_tables":1,"selectors":[],"sql":sql})
    if http_urls:
        targets.append({"why":"html has tables","strategy":"html_tables","url":http_urls[0],"max_tables":2,"selectors":[],"sql":None})
    if not targets:
        targets.append({"why":"no obvious source","strategy":"html_tables","url":None,"max_tables":1,"selectors":[],"sql":None})
    return {"targets":targets,"stop_when":"first_table_covers_all_questions","time_budget_ms":30000,"decider":"fallback"}

async def build_plan(question_text: str, uploaded_files: Dict[str, bytes], workdir: str) -> Dict[str, Any]:
    out_dir = _outdir(workdir)
    urls = _extract_urls(question_text)
    sql_block = _find_sql_block(question_text)
    payload = {"question_text":question_text, "uploaded_files":list(uploaded_files.keys()), "detected_urls":urls, "has_sql_block":bool(sql_block)}
    oai, gem = _load_keys()
    plan = _ask_openai(payload) if oai else None
    if plan is None and gem:
        plan = _ask_gemini(payload)
    if plan is None:
        payload["sql_block"] = sql_block
        plan = _fallback(payload)
    with open(os.path.join(out_dir,"plan.json"),"w",encoding="utf-8") as f:
        json.dump(plan,f,indent=2,ensure_ascii=False)
    return plan
