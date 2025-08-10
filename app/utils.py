from __future__ import annotations
import os, json, time
from typing import Any, Dict, List, Optional

def now_ms() -> int:
    return int(time.time() * 1000)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def ensure_request_dir(rid: str) -> str:
    base = os.path.join("output", rid)
    ensure_dir(base)
    return base

def validate_and_finalize(
    answers: Any,
    schema: Optional[Dict[str, Any]],
    elapsed_ms: int,
    envelope: bool = False,
    questions_len: Optional[int] = None,
    object_keys: Optional[List[str]] = None,
):
    """
    Ensure the returned structure matches the decided contract exactly.
    """
    out = None
    typ = (schema or {}).get("type", "array").lower() if isinstance(schema, dict) else "array"

    # Normalize array answers to correct length
    if typ == "array":
        arr = answers if isinstance(answers, list) else []
        if questions_len is not None:
            if len(arr) < questions_len:
                arr = arr + [None] * (questions_len - len(arr))
            elif len(arr) > questions_len:
                arr = arr[:questions_len]
        out = arr

    elif typ == "object":
        # If we have object_keys, map in order
        if object_keys and isinstance(answers, list):
            arr = answers
            if questions_len is not None and len(arr) != questions_len:
                if len(arr) < questions_len:
                    arr = arr + [None] * (questions_len - len(arr))
                else:
                    arr = arr[:questions_len]
            out = {k: (arr[i] if i < len(arr) else None) for i, k in enumerate(object_keys)}
        elif isinstance(answers, dict):
            out = answers
        else:
            # As a last resort, only return {"value":[...]} if the prompt implied object but no keys.
            arr = answers if isinstance(answers, list) else []
            out = {"value": arr}

    if not envelope:
        return out

    return {"ok": True, "data": out, "meta": {"elapsed_ms": elapsed_ms}}
