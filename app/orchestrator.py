# app/orchestrator.py
from __future__ import annotations
import os
import json
import time
import asyncio
from typing import Any, Dict, List, Optional

from .contract import build_contract
from .deterministic import try_answer_deterministic
from .duckdb_track import materialize_duckdb_samples

# Optional tracks — load lazily
try:
    from .gen_code import generate_and_run as codegen_generate_and_run  # type: ignore
except Exception:
    codegen_generate_and_run = None

try:
    from .llm_direct import try_answer_on_snippet  # type: ignore
except Exception:
    try_answer_on_snippet = None

GRAPH_QTYPES = {
    "graph_edge_count",
    "graph_highest_degree",
    "graph_average_degree",
    "graph_density",
    "graph_shortest_path",
    "graph_plot",
    "graph_degree_histogram",
}
PLOT_QTYPES = {"scatter", "graph_plot", "graph_degree_histogram"}

def _mk_sources(uploaded_files: Dict[str, bytes]) -> Dict[str, Any]:
    files: Dict[str, Dict[str, Any]] = {}
    for name, b in (uploaded_files or {}).items():
        files[name] = {
            "filename": name,
            "mimetype": _guess_mime(name),
            "bytes": b,
            "size": len(b or b""),
        }
    return {"files": files, "notes": []}

def _guess_mime(name: str) -> str:
    n = (name or "").lower()
    if n.endswith(".csv"): return "text/csv"
    if n.endswith(".json"): return "application/json"
    if n.endswith(".xlsx"): return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if n.endswith(".xls"): return "application/vnd.ms-excel"
    if n.endswith(".zip"): return "application/zip"
    if n.endswith(".png"): return "image/png"
    if n.endswith(".jpg") or n.endswith(".jpeg"): return "image/jpeg"
    if n.endswith(".parquet"): return "application/octet-stream"
    return "application/octet-stream"

def _is_data_uri_png(v: Any) -> bool:
    return isinstance(v, str) and v.startswith("data:image/png;base64,")

def _valid_by_qtype(qtype: str, val: Any) -> bool:
    if val is None: return True
    if qtype == "count": return isinstance(val, int) and val >= 0
    if qtype == "corr":  return isinstance(val, (int, float)) and -1.0 <= float(val) <= 1.0
    if qtype == "earliest": return isinstance(val, str) and bool(val.strip())
    if qtype == "scatter":  return _is_data_uri_png(val)
    if qtype in GRAPH_QTYPES:
        if qtype == "graph_edge_count": return isinstance(val, int) and val >= 0
        if qtype in {"graph_average_degree","graph_density"}: return isinstance(val, (int,float))
        if qtype == "graph_highest_degree": return isinstance(val, (str,int,float))
        if qtype == "graph_shortest_path": return isinstance(val, list) or val is None
        if qtype in {"graph_plot","graph_degree_histogram"}: return _is_data_uri_png(val)
    return isinstance(val, (str, int, float, bool))

def _pick_consensus(candidates: List[Any], qtype: str) -> Any:
    for v in candidates:  # deterministic > codegen > llm_direct (ordering done below)
        if v is not None and _valid_by_qtype(qtype, v):
            return v
    return None

def _finalize_shape(out_type: str, questions: List[str], object_keys: Optional[List[str]], merged: List[Any]) -> Any:
    n = len(questions)
    arr = list(merged[:n]) + [None] * max(0, n - len(merged))
    if out_type == "object" and object_keys:
        if len(object_keys) > len(arr):
            arr += [None] * (len(object_keys) - len(arr))
        return {k: arr[i] for i, k in enumerate(object_keys)}
    return arr

async def run_request(
    question_text: str,
    uploaded_files: Dict[str, bytes],
    json_attachments: List[Dict[str, Any]],
    schema: Dict[str, Any],
    request_id: str,
    workdir: str,
) -> Any:
    """
    Returns ONLY the final raw payload (array by default, or object if explicitly requested).
    """
    # 1) Contract
    contract = await build_contract(question_text, workdir)
    out_type = (contract.get("output") or {}).get("type", "array")
    questions: List[str] = contract.get("questions") or [question_text.strip()]
    qtypes: List[str] = contract.get("qtypes") or ["generic"] * len(questions)
    object_keys: Optional[List[str]] = contract.get("object_keys")
    if out_type not in ("array","object"):
        out_type = "array"; object_keys = None

    # 2) Sources
    sources = _mk_sources(uploaded_files)

    # 3) Decide tracks
    want_duckdb = ("s3://" in question_text) or ("read_parquet" in question_text.lower())
    run_deterministic = True
    need_codegen = any(qt in GRAPH_QTYPES or qt in {"corr", "scatter"} for qt in qtypes)
    need_llm_direct = True

    # 4) Launch in parallel
    tasks: Dict[str, asyncio.Task] = {}

    if want_duckdb:
        tasks["duckdb"] = asyncio.create_task(
            materialize_duckdb_samples(question_text, workdir, 40_000)
        )

    if run_deterministic:
        tasks["deterministic"] = asyncio.create_task(
            try_answer_deterministic(questions, sources, workdir, 55_000, question_text)
        )

    if need_llm_direct and try_answer_on_snippet is not None:
        try:
            tasks["llm_direct"] = asyncio.create_task(
                try_answer_on_snippet(questions, sources, workdir, 25_000)
            )
        except Exception:
            pass

    if need_codegen and codegen_generate_and_run is not None:
        try:
            tasks["codegen"] = asyncio.create_task(
                codegen_generate_and_run(questions, qtypes, sources, workdir, 45_000)
            )
        except Exception:
            pass

    done, pending = await asyncio.wait(tasks.values(), timeout=170, return_when=asyncio.ALL_COMPLETED)
    for p in pending:
        p.cancel()

    # 5) Collect outputs
    track_outputs: Dict[str, Dict[str, Any]] = {}
    for name, t in tasks.items():
        try:
            res = t.result()
            if isinstance(res, dict):
                track_outputs[name] = res
        except Exception as e:
            track_outputs[name] = {"error": str(e)}

    # 6) Merge candidates (priority: deterministic > codegen > llm_direct)
    num_q = len(questions)
    merged: List[Any] = []
    for i in range(num_q):
        ordered = []
        for trk in ("deterministic", "codegen", "llm_direct"):
            arr = (track_outputs.get(trk) or {}).get("array")
            if isinstance(arr, list) and i < len(arr):
                ordered.append(arr[i])
        qtype = qtypes[i] if i < len(qtypes) else "generic"
        merged.append(_pick_consensus(ordered, qtype))

    final_payload = _finalize_shape(out_type, questions, object_keys, merged)

    # Persist adjudication snapshot (for debugging)
    adj_dir = os.path.join(workdir, f"adjudicator_run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(adj_dir, exist_ok=True)
    with open(os.path.join(adj_dir, "final.json"), "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    with open(os.path.join(adj_dir, "candidates.json"), "w", encoding="utf-8") as f:
        json.dump({"questions": questions, "qtypes": qtypes, "tracks": list(track_outputs.keys()), "raw": track_outputs}, f, indent=2)

    return final_payload
