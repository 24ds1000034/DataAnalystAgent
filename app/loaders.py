from __future__ import annotations
from typing import Dict, Any, Mapping
import io
import pandas as pd

def _as_bytes(val: Any) -> bytes | None:
    if isinstance(val, (bytes, bytearray)):
        return bytes(val)
    if isinstance(val, Mapping) and "bytes" in val and isinstance(val["bytes"], (bytes, bytearray)):
        return bytes(val["bytes"])
    return None

def try_load_known(files: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Load CSV/TSV/JSON/Excel from in-memory file map.
    Returns {filename: DataFrame} (best-effort).
    """
    out: Dict[str, pd.DataFrame] = {}
    for name, meta in (files or {}).items():
        b = _as_bytes(meta)
        if not b:
            continue
        low = (name or "").lower()
        try:
            if low.endswith(".csv"):
                out[name] = pd.read_csv(io.BytesIO(b))
            elif low.endswith(".tsv"):
                out[name] = pd.read_csv(io.BytesIO(b), sep="\t")
            elif low.endswith(".json"):
                out[name] = pd.read_json(io.BytesIO(b), lines=False)
            elif low.endswith(".xlsx") or low.endswith(".xls"):
                out[name] = pd.read_excel(io.BytesIO(b))
        except Exception:
            continue
    return out
