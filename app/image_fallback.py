from __future__ import annotations
import os, io, time, glob, base64
from typing import List, Dict, Any, Optional
from PIL import Image
import re
import json

_SCATTER_RX = re.compile(r"\bscatter\s*plot|\bscatterplot", re.I)

def _wants_scatter(q: str) -> bool:
    return bool(_SCATTER_RX.search(q or ""))

def _outdir(base: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(base, f"image_fallback_run_{ts}")
    os.makedirs(d, exist_ok=True)
    return d

def _find_pngs(workdir: str) -> List[str]:
    pats = [
        os.path.join(workdir, "codegen_run_*", "*.png"),
        os.path.join(workdir, "on_data_run_*", "*.png"),
        os.path.join(workdir, "deterministic_run_*", "**", "*.png"),
    ]
    files: List[str] = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    # prefer files that look like plots
    def score(path: str) -> tuple:
        name = os.path.basename(path).lower()
        prio = 0
        if "plot" in name or "scatter" in name:
            prio += 10
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            mtime = 0
        return (prio, mtime)
    return sorted(set(files), key=score, reverse=True)

def _png_to_under_100kb(path: str, max_bytes: int = 100_000) -> Optional[bytes]:
    try:
        with Image.open(path) as im0:
            im = im0.convert("P", palette=Image.ADAPTIVE, colors=128)
            buf = io.BytesIO()
            im.save(buf, format="PNG", optimize=True)
            data = buf.getvalue()
            while len(data) > max_bytes and im.size[0] > 320:
                w, h = im.size
                im = im.resize((int(w * 0.9), int(h * 0.9)))
                buf = io.BytesIO()
                im.save(buf, format="PNG", optimize=True)
                data = buf.getvalue()
            if len(data) <= max_bytes:
                return data
            return None
    except Exception:
        return None

def _to_data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()

async def image_base64_track(questions: List[str], workdir: str, budget_ms: int) -> Dict[str, Any]:
    out_dir = _outdir(workdir)
    pngs = _find_pngs(workdir)
    ans: List[Any] = []
    best_b64: Optional[str] = None

    # prepare a single best candidate b64 once (scatter questions get the same)
    for p in pngs:
        data = _png_to_under_100kb(p, 100_000)
        if data:
            best_b64 = _to_data_uri(data)
            break

    for q in questions:
        if _wants_scatter(q) and best_b64:
            ans.append(best_b64)
        else:
            ans.append(None)

    with open(os.path.join(out_dir, "image_fallback.json"), "w", encoding="utf-8") as f:
        json.dump({"array": ans, "picked_png": (pngs[0] if pngs else None)}, f, indent=2)
    return {"array": ans}
