from __future__ import annotations
import os, io, re, json, time, glob
from typing import Any, Dict, List, Optional
import pandas as pd
import requests
from bs4 import BeautifulSoup

def _outdir(base: str) -> str:
    ts=time.strftime("%Y%m%d_%H%M%S"); d=os.path.join(base, f"direct_llm_run_{ts}")
    os.makedirs(d, exist_ok=True); return d

def _latest_csv(workdir: str) -> Optional[str]:
    pats = [
        os.path.join(workdir,"deterministic_run_*","scraped_data","table_*.csv"),
        os.path.join(workdir,"duckdb_run_*","table_*.csv"),
    ]
    files=[]
    for p in pats: files.extend(glob.glob(p, recursive=True))
    if not files: return None
    return sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)[0]

def _load_keys():
    try:
        from dotenv import load_dotenv; load_dotenv()
    except: pass
    return os.getenv("OPENAI_API_KEY"), os.getenv("GOOGLE_API_KEY")

def _mk_data_snippet(df: pd.DataFrame, max_rows=200, max_cols=8) -> str:
    cols=list(df.columns)[:max_cols]
    tiny=df[cols].head(max_rows)
    buf=io.StringIO(); tiny.to_csv(buf, index=False); return buf.getvalue()

def _ask_openai(api_key: str, questions: List[str], snippet: str) -> Optional[List[Any]]:
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        sys = ("Data analyst mode. Use ONLY the CSV snippet; if insufficient → null. "
               "Plots → null. Return ONLY a JSON array with one item per question; "
               "counts as integers; correlations floats [-1,1] (6 decimals); strings non-empty.")
        user = json.dumps({"csv_snippet": snippet, "questions": questions}, ensure_ascii=False)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0, response_format={"type":"json_object"}
        )
        obj=json.loads(resp.choices[0].message.content)
        for k in ("answers","array","data","result"):
            if isinstance(obj.get(k), list):
                arr=obj[k]; break
        else:
            if isinstance(obj, list): arr=obj
            else: return None
        if len(arr)!=len(questions): arr=(arr+[None]*len(questions))[:len(questions)]
        return arr
    except Exception:
        return None

def _ask_gemini(api_key: str, questions: List[str], snippet: str) -> Optional[List[Any]]:
    try:
        import google.generativeai as genai, re, json as _j
        genai.configure(api_key=api_key); model=genai.GenerativeModel("gemini-1.5-flash")
        prompt = ("Return ONLY a JSON array of answers derived strictly from the CSV snippet. "
                  "Do not invent values. Plot requests: null. If unsure: null.\n\n"
                  "CSV snippet:\n" + snippet + "\n\nQuestions:\n" + "\n".join(f"{i+1}. {q}" for i,q in enumerate(questions)))
        out=model.generate_content(prompt); txt=(out.text or "").strip()
        m=re.search(r"\[[\s\S]*\]", txt); 
        if not m: return None
        arr=_j.loads(m.group(0))
        if not isinstance(arr, list): return None
        if len(arr)!=len(questions): arr=(arr+[None]*len(questions))[:len(questions)]
        return arr
    except Exception:
        return None

_URL_RE = re.compile(r"https?://[^\s)]+", re.I)
def _extract_urls(text: str) -> List[str]: return _URL_RE.findall(text or "")

def _read_tables_lxml(html: str) -> List[pd.DataFrame]:
    try: return [df for df in pd.read_html(html, flavor="lxml") if df is not None and not df.empty]
    except Exception: return []

def _read_tables_bs(html: str) -> List[pd.DataFrame]:
    soup = BeautifulSoup(html, "html.parser"); dfs=[]
    for tbl in soup.find_all("table"):
        rows=[]
        for tr in tbl.find_all("tr"):
            cells=tr.find_all(["td","th"])
            if not cells: continue
            rows.append([c.get_text(strip=True) for c in cells])
        if rows:
            w=max(len(r) for r in rows); rows=[r+[""]*(w-len(r)) for r in rows]
            df=pd.DataFrame(rows[1:], columns=[str(c) for c in rows[0]]); dfs.append(df)
    return dfs

def _best_df(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if not dfs: return None
    def score(df: pd.DataFrame) -> float:
        nums=sum(pd.to_numeric(df[c], errors="coerce").notna().mean() for c in df.columns)
        return nums + 0.2*df.shape[1]
    return sorted(dfs, key=score, reverse=True)[0]

async def answer_with_llm_direct(questions: List[str], workdir: str, budget_ms: int, question_text: Optional[str]=None) -> Dict[str, Any]:
    out_dir=_outdir(workdir)
    csv=_latest_csv(workdir)
    df: Optional[pd.DataFrame] = None

    if csv and os.path.exists(csv):
        try: df=pd.read_csv(csv)
        except Exception: df=None

    if df is None and question_text:
        urls=_extract_urls(question_text)
        for u in urls:
            try:
                r=requests.get(u, timeout=12, headers={"User-Agent":"DataAnalystAgent/0.4"})
                r.raise_for_status()
                dfs=_read_tables_lxml(r.text) or _read_tables_bs(r.text)
                cand=_best_df(dfs)
                if cand is not None and not cand.empty:
                    df=cand; break
            except Exception:
                continue

    if df is None or df.empty:
        with open(os.path.join(out_dir,"notes.json"),"w") as f: json.dump({"status":"no-csv"}, f, indent=2)
        return {}

    snippet=_mk_data_snippet(df)
    oai, gem = _load_keys()
    ans=None
    if oai: ans=_ask_openai(oai, questions, snippet)
    if ans is None and gem: ans=_ask_gemini(gem, questions, snippet)
    if ans is None:
        with open(os.path.join(out_dir,"direct_answers.json"),"w") as f: json.dump({"provider":"none","status":"no-answer"}, f, indent=2)
        return {}
    with open(os.path.join(out_dir,"direct_answers.json"),"w", encoding="utf-8") as f: json.dump(ans, f, indent=2, ensure_ascii=False)
    return {"array": ans}
