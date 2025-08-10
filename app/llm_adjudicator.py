from __future__ import annotations
import os, json, time, base64, re
from typing import Any, Dict, List, Optional

_rx_count = re.compile(r"\b(count|how many)\b", re.I)
_rx_corr  = re.compile(r"\bcorrelat", re.I)
_rx_scatter = re.compile(r"\bscatter\s*plot|\bscatterplot", re.I)
_rx_earliest = re.compile(r"\bearliest|first\b", re.I)

def kind_of(q: str) -> str:
    q = q or ""
    if _rx_scatter.search(q):  return "scatter"
    if _rx_corr.search(q):     return "corr"
    if _rx_count.search(q):    return "count"
    if _rx_earliest.search(q): return "earliest"
    return "generic"

def _is_intlike(x: Any) -> bool:
    if isinstance(x, bool): return False
    if isinstance(x, int): return True
    if isinstance(x, float): return x.is_integer() and x >= 0
    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() in ("nan","none","null"): return False
        return s.isdigit()
    return False

def _to_int(x: Any) -> Optional[int]:
    try:
        if isinstance(x, int): return x if x >= 0 else None
        if isinstance(x, float): return int(x) if x >= 0 else None
        if isinstance(x, str): return int(float(x))
    except: pass
    return None

def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except: return None

def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip()) and x.strip().lower() not in ("nan","none","null","na","n/a")

def _is_data_uri_png_b64(x: Any, max_bytes: int = 100_000) -> bool:
    if not isinstance(x, str): return False
    if not x.startswith("data:image/png;base64,"): return False
    try:
        raw = base64.b64decode(x.split(",", 1)[1], validate=True)
        return len(raw) <= max_bytes
    except Exception:
        return False

def validate_value(kind: str, v: Any) -> bool:
    if v is None: return False
    if kind == "count":   return _to_int(v) is not None
    if kind == "corr":
        vf = _to_float(v); return vf is not None and -1.000001 <= vf <= 1.000001
    if kind == "scatter": return _is_data_uri_png_b64(v)
    if kind == "earliest":return _is_nonempty_str(v)
    if isinstance(v, str) and v.strip().lower() in ("nan","none","null","na"): return False
    return True

def normalize_value(kind: str, v: Any) -> Any:
    if v is None: return None
    if kind == "count":   return _to_int(v)
    if kind == "corr":
        vf = _to_float(v); 
        if vf is None: return None
        return round(max(-1.0, min(1.0, vf)), 6)
    if kind == "earliest":return v.strip() if isinstance(v, str) else v
    return v

def choose_programmatic(kind: str, candidates: List[Dict[str, Any]], priority: List[str]) -> Any:
    valid = []
    for c in candidates:
        if validate_value(kind, c["value"]):
            valid.append({"source": c["source"], "value": normalize_value(kind, c["value"])})
    if not valid: return None
    from collections import Counter
    counter = Counter([json.dumps(v["value"], sort_keys=True) for v in valid])
    if counter:
        top_json, top_count = counter.most_common(1)[0]
        if top_count >= 2:
            try: return json.loads(top_json)
            except: pass
    for src in priority:
        for v in valid:
            if v["source"] == src:
                return v["value"]
    if kind in ("count","corr"):
        nums = [v["value"] for v in valid if isinstance(v["value"], (int, float))]
        if nums:
            nums = sorted(nums)
            return nums[len(nums)//2]
    return valid[0]["value"]

def _load_keys():
    try:
        from dotenv import load_dotenv; load_dotenv()
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY"), os.getenv("GOOGLE_API_KEY")  # type: ignore[name-defined]

def _ask_openai(api_key: str, questions: List[str], per_q_options: List[List[Dict[str, Any]]],
                programmatic: List[Any]) -> Optional[List[Any]]:
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        sys = (
            "You are a strict adjudicator. You will receive for each question a small set of "
            "ALREADY-COMPUTED candidate answers from multiple tools. "
            "You MUST pick ONLY from the provided candidates, or null if all are invalid. "
            "Enforce types:\n"
            "- count: integer >= 0\n- corr: float in [-1,1] rounded to 6 decimals\n"
            "- earliest: non-empty string (not 'nan')\n- scatter: data:image/png;base64 under 100kB (already validated)\n"
            "Return ONLY a JSON array with one element per question."
        )
        user = json.dumps({
            "questions": questions,
            "candidates": per_q_options,
            "programmatic_recommendation": programmatic
        }, ensure_ascii=False)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0, response_format={"type":"json_object"},
        )
        obj = json.loads(resp.choices[0].message.content)
        for k in ("answers","array","data","result"):
            if isinstance(obj.get(k), list):
                arr = obj[k]
                if len(arr) != len(questions):
                    arr = (arr + [None]*len(questions))[:len(questions)]
                return arr
        if isinstance(obj, list):
            arr = obj
            if len(arr) != len(questions):
                arr = (arr + [None]*len(questions))[:len(questions)]
            return arr
    except Exception:
        return None
    return None

def _ask_gemini(api_key: str, questions: List[str], per_q_options: List[List[Dict[str, Any]]],
                programmatic: List[Any]) -> Optional[List[Any]]:
    try:
        import google.generativeai as genai, re, json as _j
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Adjudicate among provided candidates. Pick ONLY from candidates or null. Types:\n"
            "- count: integer >= 0\n- corr: float [-1,1], 6 decimals\n- earliest: non-empty string (not 'nan')\n"
            "- scatter: valid data:image/png;base64 under 100kB (already validated)\n"
            "Return ONLY a JSON array with one element per question.\n\n"
            + json.dumps({"questions": questions, "candidates": per_q_options, "programmatic_recommendation": programmatic}, ensure_ascii=False)
        )
        out = model.generate_content(prompt)
        txt = (out.text or "").strip()
        m = re.search(r"\[[\s\S]*\]", txt)
        if not m: return None
        arr = _j.loads(m.group(0))
        if isinstance(arr, list):
            if len(arr) != len(questions):
                arr = (arr + [None]*len(questions))[:len(questions)]
            return arr
    except Exception:
        return None
    return None

def _outdir(base: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(base, f"adjudicator_run_{ts}")
    os.makedirs(d, exist_ok=True)
    return d

async def final_llm_adjudicate(
    questions: List[str],
    track_arrays: Dict[str, List[Any]],
    workdir: str,
    budget_ms: int,
) -> List[Any]:
    out_dir = _outdir(workdir)
    # include image fallback in priority (after codegen)
    sources_in_priority = ["direct", "on_data", "deterministic", "codegen", "image"]
    n = len(questions)
    per_q_options: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
    for src in sources_in_priority:
        arr = track_arrays.get(src) or []
        if len(arr) < n:
            arr = (arr + [None]*n)[:n]
        for i in range(n):
            per_q_options[i].append({"source": src, "value": arr[i]})

    programmatic: List[Any] = []
    for i, q in enumerate(questions):
        k = kind_of(q)
        programmatic.append(choose_programmatic(k, per_q_options[i], sources_in_priority))

    with open(os.path.join(out_dir, "candidates.json"), "w", encoding="utf-8") as f:
        json.dump({"questions": questions, "candidates": per_q_options, "programmatic": programmatic}, f, indent=2, ensure_ascii=False)

    oai_key, gem_key = _load_keys()
    final: Optional[List[Any]] = None
    if oai_key:
        final = _ask_openai(oai_key, questions, per_q_options, programmatic)
    if final is None and gem_key:
        final = _ask_gemini(gem_key, questions, per_q_options, programmatic)

    if final is None:
        final = programmatic

    normalized = []
    for i, v in enumerate(final):
        k = kind_of(questions[i])
        if validate_value(k, v):
            normalized.append(normalize_value(k, v))
        else:
            normalized.append(None)

    with open(os.path.join(out_dir, "final.json"), "w", encoding="utf-8") as f:
        json.dump({"final": normalized}, f, indent=2, ensure_ascii=False)

    return normalized
