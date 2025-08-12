from __future__ import annotations
import os, re, json, time
from typing import Any, Dict, List, Optional

_NUM = re.compile(r"^\s*(\d+)[\.\)]\s+(.*\S)\s*$")
_BUL = re.compile(r"^\s*[-*]\s+(.*\S)\s*$", re.M)
_CODE_FENCE = re.compile(r"```(json|JSON)?\s*([\s\S]*?)```", re.M)

# Graph lexicon → graph_* qtypes
_RX_GRAPH_EDGE     = re.compile(r"\bedge(s)?\b", re.I)
_RX_GRAPH_DEGREE   = re.compile(r"\bdegree\b", re.I)
_RX_GRAPH_DENSITY  = re.compile(r"\bdensity\b", re.I)
_RX_GRAPH_SHORTEST = re.compile(r"\bshortest\s*path\b", re.I)
_RX_GRAPH_PLOT     = re.compile(r"\b(graph\s*plot|network\s*plot|degree\s*histogram)\b", re.I)
_RX_CORR           = re.compile(r"\bcorrelat", re.I)
_RX_SCATTER        = re.compile(r"\bscatter\s*plot|\bscatterplot", re.I)
_RX_COUNT          = re.compile(r"\bcount\b|\bhow\s+many\b", re.I)
_RX_EARLIEST       = re.compile(r"\bearliest\b|\bfirst\b", re.I)

def _outdir(base: str) -> str:
    d = os.path.join(base, f"contract_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(d, exist_ok=True)
    return d

def _load_keys():
    try:
        from dotenv import load_dotenv; load_dotenv()
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY"), os.getenv("GOOGLE_API_KEY")

def _strip(s: str) -> str:
    return (s or "").replace("\r", "")

def _json_fences(txt: str) -> List[Dict[str, Any]]:
    objs=[]
    for m in _CODE_FENCE.finditer(txt or ""):
        body=m.group(2).strip()
        if not body: continue
        try:
            obj=json.loads(body)
            if isinstance(obj, dict):
                objs.append(obj)
        except Exception:
            continue
    return objs

def _questionish_score(keys: List[str]) -> float:
    # Heuristic: question-like keys contain '?' or >= 5 words, or start with interrogatives
    qwords=("what","which","when","where","why","how")
    score=0; n=max(1,len(keys))
    for k in keys:
        s=str(k).strip()
        if "?" in s: score+=1.0; continue
        if len(s.split())>=5: score+=0.8; continue
        if s.lower().startswith(qwords): score+=0.7; continue
    return score/n

def _rowish_score(keys: List[str]) -> float:
    # Heuristic for sample rows / schemas (short, snake/camel keys)
    score=0; n=max(1,len(keys))
    for k in keys:
        s=str(k)
        if len(s.split())<=2 and (("_" in s) or (s.islower() and s.isidentifier())):
            score+=1.0
    return score/n

def _find_best_object_schema(txt: str) -> Optional[Dict[str, Any]]:
    cands=_json_fences(txt)
    if not cands: return None
    # prefer the dict whose keys look like user questions, not sample row columns
    best=None; best_sc=-1
    for d in cands:
        keys=list(d.keys())
        qsc=_questionish_score(keys)
        rsc=_rowish_score(keys)
        sc=qsc - 0.3*rsc
        if sc>best_sc:
            best_sc=sc; best=d
    # keep only if sufficiently questionish
    if best and _questionish_score(list(best.keys()))>=0.5:
        return best
    return None

def _extract_questions_locally(txt: str) -> List[str]:
    lines = _strip(txt).split("\n")
    out: List[str] = []
    # Prefer “Questions/Tasks” section if present
    start = 0
    for i, ln in enumerate(lines):
        if re.search(r"^\s*(questions|tasks)\s*[:\-]?\s*$", ln, re.I):
            start = i + 1
            break
    lines = lines[start:]
    # numbered
    for ln in lines:
        m = _NUM.match(ln)
        if m: out.append(m.group(2).strip())
    if out: return out
    # bullets
    bul = _BUL.findall("\n".join(lines))
    if bul: return [b.strip() for b in bul if b.strip()]
    # fallback single
    t = txt.strip()
    return [t] if t else []

def _detect_format_and_schema(txt: str) -> Dict[str, Any]:
    txt = _strip(txt)
    if re.search(r"respond\s+with\s+a\s+json\s+object", txt, re.I):
        schema = _find_best_object_schema(txt)
        return {
            "type": "object",
            "schema": schema,
            "object_keys": list(schema.keys()) if isinstance(schema, dict) else None
        }
    return {"type": "array", "schema": None, "object_keys": None}

def _infer_qtype(q: str) -> str:
    if _RX_SCATTER.search(q):   return "scatter"
    if _RX_CORR.search(q):      return "corr"
    if _RX_EARLIEST.search(q):  return "earliest"
    if _RX_COUNT.search(q):     return "count"
    # graph family
    if _RX_GRAPH_SHORTEST.search(q): return "graph_shortest_path"
    if _RX_GRAPH_DENSITY.search(q):  return "graph_density"
    if _RX_GRAPH_DEGREE.search(q):
        if re.search(r"highest", q, re.I): return "graph_highest_degree"
        if re.search(r"average|avg", q, re.I): return "graph_average_degree"
        return "graph_degree"
    if _RX_GRAPH_EDGE.search(q): return "graph_edge_count"
    if _RX_GRAPH_PLOT.search(q):
        if re.search(r"degree\s*histogram", q, re.I): return "graph_degree_histogram"
        return "graph_plot"
    return "generic"

def _ask_openai(question_text: str) -> Optional[Dict[str, Any]]:
    try:
        import openai
        client=openai.OpenAI()
        sys=("Contract decider. Choose output {'type':'array'|'object','schema':object|null} "
             "ONLY pick 'object' if prompt explicitly asks. Extract ordered questions. "
             "Infer qtypes per question. Return ONLY JSON: {output:{},questions:[],qtypes:[],object_keys:null|[]}."
        )
        resp=client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":question_text}],
            temperature=0, response_format={"type":"json_object"},
        )
        obj=json.loads(resp.choices[0].message.content)
        out=obj.get("output") or {}
        typ=out.get("type") or "array"
        sch=out.get("schema") if isinstance(out.get("schema"), dict) else None
        q =obj.get("questions") or []
        qt=obj.get("qtypes") or ["generic"]*len(q)
        ok=obj.get("object_keys")
        return {"output":{"type":typ,"schema":sch},"questions":q,"qtypes":qt,"object_keys":ok}
    except Exception:
        return None

def _ask_gemini(question_text: str) -> Optional[Dict[str, Any]]:
    try:
        import google.generativeai as genai, re as _re, json as _j
        genai.configure()
        model=genai.GenerativeModel("gemini-1.5-flash")
        prompt=("Return ONLY JSON: {output:{type:'array'|'object',schema:null|object},questions:[],qtypes:[],object_keys:null|[]}. "
                "Choose 'object' only if explicitly requested; else 'array'.")
        out=model.generate_content([prompt, question_text])
        txt=(out.text or "").strip()
        m=_re.search(r"\{[\s\S]*\}\s*$", txt)
        if not m: return None
        obj=_j.loads(m.group(0))
        outp=obj.get("output") or {}
        typ=outp.get("type") or "array"
        sch=outp.get("schema") if isinstance(outp.get("schema"), dict) else None
        q  =obj.get("questions") or []
        qt =obj.get("qtypes") or ["generic"]*len(q)
        ok =obj.get("object_keys")
        return {"output":{"type":typ,"schema":sch},"questions":q,"qtypes":qt,"object_keys":ok}
    except Exception:
        return None

async def build_contract(question_text: str, workdir: str) -> Dict[str, Any]:
    out_dir = _outdir(workdir)

    fmt = _detect_format_and_schema(question_text)
    questions = _extract_questions_locally(question_text)
    qtypes = [_infer_qtype(q) for q in questions]

    # Try LLM refinement (optional)
    oai_key, gem_key = _load_keys()
    decided = _ask_openai(question_text) if oai_key else None
    if decided is None and gem_key:
        decided = _ask_gemini(question_text)

    if decided and (decided.get("questions") or []):
        if len(decided["questions"]) >= len(questions):
            questions = decided["questions"]
        if decided.get("output"):
            # Keep 'object' only if explicitly requested
            if decided["output"]["type"] == "object":
                if re.search(r"respond\s+with\s+a\s+json\s+object", question_text, re.I):
                    fmt["type"]="object"; fmt["schema"]=decided["output"].get("schema")
                    if not fmt.get("object_keys") and fmt["schema"]:
                        fmt["object_keys"] = list(fmt["schema"].keys())
                else:
                    fmt["type"]="array"; fmt["schema"]=None
            else:
                fmt["type"]=decided["output"]["type"]; fmt["schema"]=decided["output"].get("schema")
        qtypes = decided.get("qtypes") or qtypes
        if decided.get("object_keys"):
            fmt["object_keys"] = decided.get("object_keys")

    contract = {"output":{"type":fmt["type"],"schema":fmt["schema"]},
                "questions":questions, "qtypes":qtypes,
                "object_keys":fmt.get("object_keys")}
    with open(os.path.join(out_dir, "contract.json"), "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, ensure_ascii=False)
    return contract
