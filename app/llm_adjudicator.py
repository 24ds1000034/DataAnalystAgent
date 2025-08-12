# app/llm_adjudicator.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

GRAPH_QTYPES = {
    "graph_edge_count",
    "graph_highest_degree",
    "graph_average_degree",
    "graph_density",
    "graph_shortest_path",
    "graph_plot",
    "graph_degree_histogram",
}

def is_data_uri_png(v: Any) -> bool:
    return isinstance(v, str) and v.startswith("data:image/png;base64,")

def valid_by_qtype(qtype: str, val: Any) -> bool:
    if val is None:
        return True
    if qtype == "count":
        return isinstance(val, int) and val >= 0
    if qtype == "corr":
        if isinstance(val, (int, float)):
            return -1.0 <= float(val) <= 1.0
        return False
    if qtype == "earliest":
        return isinstance(val, str) and len(val.strip()) > 0
    if qtype == "scatter":
        return is_data_uri_png(val)
    if qtype in GRAPH_QTYPES:
        if qtype == "graph_edge_count":
            return isinstance(val, int) and val >= 0
        if qtype in {"graph_average_degree", "graph_density"}:
            return isinstance(val, (int, float))
        if qtype == "graph_highest_degree":
            return isinstance(val, (str, int, float))
        if qtype == "graph_shortest_path":
            return isinstance(val, list) or val is None
        if qtype in {"graph_plot", "graph_degree_histogram"}:
            return is_data_uri_png(val)
    return isinstance(val, (str, int, float, bool))

def pick_consensus(
    per_track: Dict[str, Any],
    qtype: str,
    priority=("deterministic", "codegen", "llm_direct"),
) -> Any:
    for trk in priority:
        if trk in per_track:
            v = per_track[trk]
            if valid_by_qtype(qtype, v) and v is not None:
                return v
    return None

def finalize_shape(
    out_type: str,
    questions: List[str],
    qtypes: List[str],
    object_keys: Optional[List[str]],
    merged: List[Any],
) -> Any:
    n = len(questions)
    arr = list(merged[:n]) + [None] * max(0, n - len(merged))
    if out_type == "object" and object_keys:
        if len(object_keys) > len(arr):
            arr += [None] * (len(object_keys) - len(arr))
        return {k: arr[i] for i, k in enumerate(object_keys)}
    return arr
