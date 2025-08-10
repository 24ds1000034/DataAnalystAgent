from __future__ import annotations
from typing import Dict, Any, List

def prepare_sources(
    uploaded_files: Dict[str, bytes],
    json_attachments: List[Dict[str, Any]],
    workdir: str,
    budget_ms: int,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "files": { filename: {"bytes": <bytes>, "size": int} },
        "notes": [...]
      }
    """
    files_meta: Dict[str, Any] = {}
    for name, b in (uploaded_files or {}).items():
        if not isinstance(b, (bytes, bytearray)):
            continue
        files_meta[name] = {"bytes": bytes(b), "size": len(b)}
    for j in (json_attachments or []):
        if not isinstance(j, dict):
            continue
        # If any JSON attachments include inline content
        if "filename" in j and "content" in j and isinstance(j["content"], (str, bytes)):
            content = j["content"].encode("utf-8") if isinstance(j["content"], str) else j["content"]
            files_meta[j["filename"]] = {"bytes": content, "size": len(content)}
    return {"files": files_meta, "notes": []}
