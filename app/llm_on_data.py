from __future__ import annotations
import os, re, io, json, time, glob, base64
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def _outdir(base: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(base, f"on_data_run_{ts}")
    os.makedirs(d, exist_ok=True)
    return d

def _latest_csv_from_runs(workdir: str) -> Optional[str]:
    cand=sorted(glob.glob(os.path.join(workdir,"deterministic_run_*","scraped_data","table_*.csv")), reverse=True)
    return cand[0] if cand else None

def _load_keys():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY"), os.getenv("GOOGLE_API_KEY")

# --- helpers (parsing/normalizing) ---
_CURRENCY_RX = re.compile(r"[^\d\.\-]+")
def to_numeric_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc": return pd.to_numeric(s, errors="coerce")
    vals = s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    vals = vals.str.replace(_CURRENCY_RX, "", regex=True)
    return pd.to_numeric(vals, errors="coerce")

def find_best_col(cols: List[str], target: str) -> Optional[str]:
    target = (target or "").strip().lower()
    if not cols or not target: return None
    for c in cols:
        if str(c).strip().lower()==target: return c
    def norm(s: str) -> str: return re.sub(r"[^a-z0-9]+","_", s.strip().lower())
    def grams(s: str) -> set: s2=norm(s); return set(s2[i:i+3] for i in range(max(1,len(s2)-2)))
    best, score = None, -1.0
    T = grams(target)
    for c in cols:
        C = grams(str(c))
        inter = len(T & C); uni = len(T | C)
        sc = inter/uni if uni else 0.0
        if sc > score:
            best, score = c, sc
    return best if score >= 0.35 else None

def parse_column_pair_request(q: str) -> Optional[Tuple[str,str]]:
    m = re.search(r"(?:between|of)\s+([A-Za-z0-9 _\-./]+?)\s+(?:and|&)\s+([A-Za-z0-9 _\-./]+)", q, re.I)
    if not m: return None
    return m.group(1).strip(), m.group(2).strip()

def wants_correlation(q: str) -> bool: return bool(re.search(r"\bcorrelat", q, re.I))
def wants_scatter(q: str) -> bool:     return bool(re.search(r"\bscatter\s*plot|\bscatterplot", q, re.I))
def wants_count(q: str) -> bool:       return bool(re.search(r"\bcount|how many\b", q, re.I))
def wants_earliest(q: str) -> bool:    return bool(re.search(r"\bearliest|first\b", q, re.I))
def parse_threshold(q: str) -> Optional[float]:
    m = re.search(r"\$?\s*([0-9][0-9,\.]*)\s*(b|bn|m|million|k|thousand)?", q, re.I)
    if not m: return None
    num = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit in ("b","bn"): num *= 1_000_000_000
    elif unit in ("m","million"): num *= 1_000_000
    elif unit in ("k","thousand"): num *= 1_000
    return num
def parse_year(q: str, which: str="before") -> Optional[int]:
    if which=="before": m = re.search(r"(?:before|prior to)\s*(\d{4})", q, re.I)
    else:               m = re.search(r"(?:after|from)\s*(\d{4})", q, re.I)
    return int(m.group(1)) if m else None

def _png_opt(img_bytes: bytes, max_bytes: int = 100_000) -> bytes:
    im = Image.open(io.BytesIO(img_bytes)).convert("P", palette=Image.ADAPTIVE, colors=128)
    out = io.BytesIO(); im.save(out, format="PNG", optimize=True)
    b = out.getvalue()
    while len(b) > max_bytes and im.size[0] > 320:
        im = im.resize((int(im.size[0]*0.9), int(im.size[1]*0.9)))
        out = io.BytesIO(); im.save(out, format="PNG", optimize=True)
        b = out.getvalue()
    return b

def _scatter_b64(x, y, xlabel: str, ylabel: str, max_bytes: int=100_000) -> Optional[str]:
    x = np.asarray(x); y = np.asarray(y)
    m = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[m], y[m]
    if x.size < 2 or y.size < 2: return None
    if x.size > 5000:
        idx = np.linspace(0, x.size-1, 5000).astype(int)
        x, y = x[idx], y[idx]
    slope, intercept = np.polyfit(x, y, 1)
    fig = plt.figure(figsize=(6.4,4.8), dpi=115)
    ax = plt.gca()
    ax.scatter(x, y, s=8, alpha=0.7)
    ax.plot([x.min(), x.max()], [slope*x.min()+intercept, slope*x.max()+intercept], linestyle="--", color="red")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(_png_opt(buf.getvalue(), max_bytes)).decode()

# --- LLM column mapping ---
def _ask_openai_for_mapping(api_key: str, colnames: List[str], questions: List[str]) -> Optional[Dict[str, str]]:
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        sys = (
            "You are a schema-mapping assistant for analytics. "
            "Given a list of table columns and the user questions, map semantic fields "
            "['rank','peak','year','title','metric'] to the best actual column names (or null). "
            "Return ONLY JSON: {\"column_map\": { ... }} with those keys."
        )
        user = json.dumps({"columns": colnames, "questions": questions}, ensure_ascii=False)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0, response_format={"type":"json_object"},
        )
        obj = json.loads(resp.choices[0].message.content)
        cm = obj.get("column_map")
        if isinstance(cm, dict):
            return {k: (v if v else None) for k, v in cm.items()}
    except Exception:
        return None
    return None

def _ask_gemini_for_mapping(api_key: str, colnames: List[str], questions: List[str]) -> Optional[Dict[str,str]]:
    try:
        import google.generativeai as genai, re
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Map columns to semantic fields ['rank','peak','year','title','metric'].\n"
            "Columns:\n" + json.dumps(colnames, ensure_ascii=False) + "\n\n"
            "Questions:\n" + json.dumps(questions, ensure_ascii=False) + "\n\n"
            "Return ONLY JSON: {\"column_map\": {...}}."
        )
        out = model.generate_content(prompt)
        txt = (out.text or "").strip()
        m = re.search(r"\{[\s\S]*\}", txt)
        if not m: return None
        obj = json.loads(m.group(0))
        cm = obj.get("column_map")
        if isinstance(cm, dict):
            return {k: (v if v else None) for k, v in cm.items()}
    except Exception:
        return None
    return None

def _llm_column_map(colnames: List[str], questions: List[str]) -> Dict[str, Optional[str]]:
    oai, gem = _load_keys()
    mapping: Optional[Dict[str,str]] = None
    if oai:
        mapping = _ask_openai_for_mapping(oai, colnames, questions)
    if not mapping and gem:
        mapping = _ask_gemini_for_mapping(gem, colnames, questions)
    if not mapping:
        mapping = {}
        mapping["rank"] = find_best_col(colnames, "rank")
        mapping["peak"] = find_best_col(colnames, "peak")
        mapping["year"] = find_best_col(colnames, "year") or find_best_col(colnames, "date")
        mapping["title"] = find_best_col(colnames, "title") or find_best_col(colnames, "name") or find_best_col(colnames, "film") or find_best_col(colnames, "movie")
        for cand in ["worldwide gross","gross","revenue","value","amount","total","price","score","peak"]:
            c = find_best_col(colnames, cand)
            if c:
                mapping["metric"] = c; break
        mapping.setdefault("metric", None)
    return {k: (mapping.get(k) or None) for k in ["rank","peak","year","title","metric"]}

async def small_sample_insights(
    questions: List[str],
    sources: Dict[str, Any],
    workdir: str,
    budget_ms: int,
) -> Dict[str, Any]:
    out_dir = _outdir(workdir)
    csv_path = _latest_csv_from_runs(workdir)

    # Fallback to uploaded files if deterministic hasn't saved CSV yet
    if not csv_path or not os.path.exists(csv_path):
        # write a temp CSV from the largest uploaded DF, if any
        from .loaders import try_load_known
        file_dfs = try_load_known(sources.get("files", {}))
        if file_dfs:
            df0 = max(file_dfs.values(), key=lambda d: d.shape[0] * d.shape[1])
            csv_path = os.path.join(out_dir, "fallback.csv")
            df0.to_csv(csv_path, index=False)

    if not csv_path or not os.path.exists(csv_path):
        with open(os.path.join(out_dir, "notes.json"), "w", encoding="utf-8") as f:
            json.dump({"status": "no-csv"}, f, indent=2)
        return {}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        with open(os.path.join(out_dir, "notes.json"), "w", encoding="utf-8") as f:
            json.dump({"status": "csv-load-error", "error": str(e)}, f, indent=2)
        return {}

    df.columns = [str(c) for c in df.columns]
    colnames = list(df.columns)
    column_map = _llm_column_map(colnames, questions)

    with open(os.path.join(out_dir, "mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"columns": colnames, "column_map": column_map, "questions": questions}, f, indent=2)

    answers: List[Any] = []
    for q in questions:
        q = str(q or "").strip()
        ans = None

        if wants_correlation(q):
            pair = parse_column_pair_request(q)
            a = column_map.get("rank") if pair is None else find_best_col(colnames, pair[0])
            b = column_map.get("peak") if pair is None else find_best_col(colnames, pair[1])
            if a and b and a in df.columns and b in df.columns:
                x = to_numeric_series(df[a]); y = to_numeric_series(df[b])
                if x.notna().sum() >= 2 and y.notna().sum() >= 2:
                    ans = round(float(x.corr(y)), 6)
            answers.append(ans); continue

        if wants_scatter(q):
            pair = parse_column_pair_request(q)
            a = column_map.get("rank") if pair is None else find_best_col(colnames, pair[0])
            b = column_map.get("peak") if pair is None else find_best_col(colnames, pair[1])
            if a and b and a in df.columns and b in df.columns:
                x = to_numeric_series(df[a]); y = to_numeric_series(df[b])
                mask = x.notna() & y.notna()
                if mask.sum() >= 2:
                    b64 = _scatter_b64(x[mask].values, y[mask].values, a, b, max_bytes=100_000)
                    ans = b64
                    if b64:
                        with open(os.path.join(out_dir, "plot.png"), "wb") as pf:
                            import base64
                            png_bytes = base64.b64decode(b64.split(",",1)[1])
                            pf.write(png_bytes)
            answers.append(ans); continue

        if wants_count(q):
            thr = parse_threshold(q)
            metric_c = column_map.get("metric")
            year_c = column_map.get("year")
            if thr is not None and metric_c and metric_c in df.columns:
                s = to_numeric_series(df[metric_c])
                mask = s >= thr
                if year_c and year_c in df.columns:
                    y = to_numeric_series(df[year_c])
                    before = parse_year(q, "before")
                    after  = parse_year(q, "after")
                    if before is not None: mask = mask & (y < before)
                    if after  is not None: mask = mask & (y >= after)
                ans = int(mask.sum())
            answers.append(ans); continue

        if wants_earliest(q):
            thr = parse_threshold(q)
            metric_c = column_map.get("metric")
            title_c  = column_map.get("title")
            year_c   = column_map.get("year")
            if thr is not None and metric_c and title_c and metric_c in df.columns and title_c in df.columns:
                s = to_numeric_series(df[metric_c])
                mask = s >= thr
                if year_c and year_c in df.columns:
                    y = to_numeric_series(df[year_c])
                    tmp = df[mask].copy(); tmp["_y"] = y[mask]; tmp = tmp.dropna(subset=["_y"])
                    if not tmp.empty:
                        row = tmp.sort_values("_y").iloc[0]
                        ans = str(row[title_c])
                else:
                    tmp = df[mask]
                    if not tmp.empty:
                        ans = str(tmp.iloc[0][title_c])
            answers.append(ans); continue

        answers.append(None)

    with open(os.path.join(out_dir, "answers.json"), "w", encoding="utf-8") as f:
        json.dump({"array": answers}, f, indent=2)

    return {"array": answers}
