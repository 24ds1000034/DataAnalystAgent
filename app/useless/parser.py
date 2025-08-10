from __future__ import annotations
import re
from typing import List

_NUM_RE = re.compile(r"^\s*(\d+)[\.\)]\s+(.*)$")

def parse_questions(text: str) -> List[str]:
    """
    Parse enumerated questions from a text document.
    - Lines like "1. ..." or "2) ...".
    - Continuations appended to the last numbered item.
    - Fallback to bullets (-,*,+) if no numbered lines.
    - Final fallback: whole text as one question.
    """
    lines = [ln.rstrip("\n") for ln in (text or "").splitlines()]

    items: List[str] = []
    in_enum = False

    for ln in lines:
        if not ln.strip():
            continue
        m = _NUM_RE.match(ln)
        if m:
            q_text = m.group(2).strip()
            items.append(q_text)
            in_enum = True
            continue
        if in_enum and items and not _NUM_RE.match(ln) and not ln.lstrip().startswith(("-", "*", "+")):
            items[-1] = (items[-1].rstrip() + " " + ln.strip()).strip()

    if items:
        return [it.strip() for it in items if it and it.strip()]

    bullets: List[str] = []
    for ln in lines:
        s = ln.strip()
        if s.startswith(("-", "*", "+")):
            bullets.append(s[1:].strip())
    if bullets:
        return bullets

    return [text.strip()] if text and text.strip() else []
