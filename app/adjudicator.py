from __future__ import annotations
from typing import Any, Dict, List, Union

def _coerce_array(n: int, track_res: Dict[str, Any]) -> List[Any]:
    if not isinstance(track_res, dict):
        return [None] * n
    arr = None
    for k in ("array", "data", "result"):
        if k in track_res and isinstance(track_res[k], list):
            arr = track_res[k]
            break
    if arr is None:
        return [None] * n
    if len(arr) != n:
        arr = (arr + [None] * n)[:n]
    return arr

def adjudicate(questions: List[str], tracks: List[Dict[str, Any]]) -> Union[List[Any], Dict[str, Any]]:
    n = len(questions)
    out = [None] * n
    for t in tracks:
        arr = _coerce_array(n, t)
        for i in range(n):
            if out[i] is None and arr[i] is not None:
                out[i] = arr[i]
    return out
