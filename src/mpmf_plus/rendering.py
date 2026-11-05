"""Block rendering: memory, thinking, prior-reasoning.

Ported 1:1 from mpmf_plus_user._recall_thinking_block / _recall_cognition_block
so the injected bytes are unchanged. No host imports.
"""
from __future__ import annotations

from typing import Any, Dict, List

from . import block as _block

LESSON_FRAMING = (
    "Past lessons that may or may not apply. Use one only if this "
    "situation clearly matches its trigger; otherwise ignore them all. "
    "Never mention them."
)


def render_memory(query: str, result: Any, memory_manager: Any, top_k: int = 8,
                  max_chars: int = 3500, elapsed: float = 0.0) -> str:
    body, _items = _block.render_memory_block(
        query, result, memory_manager, top_k, elapsed=elapsed, max_chars=max_chars)
    return body or ""


def thinking_block(results: List[Dict[str, Any]], max_chars: int = 800) -> str:
    """<thinking-context> block, byte-identical to the plugin's builder."""
    if not results:
        return ""
    lines = ["<thinking-context>",
             "Prior reasoning traces relevant to this task:"]
    used = 0
    n = 0
    for r in results:
        try:
            ctx = (r.get("context") or "").strip()
            concl = (r.get("conclusion") or "").strip()
            outcome = (r.get("outcome") or "unknown")
        except Exception:
            continue
        if not ctx and not concl:
            continue
        entry = f"- **{ctx[:140]}**"
        if concl:
            entry += f" → {concl[:200]}"
        if outcome not in ("unknown", ""):
            entry += f" *(outcome: {outcome})*"
        if used + len(entry) > max_chars:
            rem = max_chars - used
            if rem > 40:
                lines.append(entry[:rem] + "…")
            break
        lines.append(entry)
        used += len(entry)
        n += 1
    if n == 0:
        return ""
    lines.append("</thinking-context>")
    return "\n".join(lines)


def truncate_cognition_block(block: str, max_chars: int) -> str:
    OPEN, CLOSE = "<prior-reasoning>", "</prior-reasoning>"
    MARK = "\n[... prior-reasoning truncated ...]"
    if block.startswith(OPEN) and block.endswith(CLOSE):
        inner = block[len(OPEN):-len(CLOSE)]
        keep = max(0, max_chars - len(OPEN) - len(CLOSE) - len(MARK))
        return f"{OPEN}{inner[:keep]}{MARK}{CLOSE}"
    return block[:max_chars] + MARK + CLOSE


def with_lesson_framing(block: str) -> str:
    if not block or not block.strip() or LESSON_FRAMING in block:
        return block
    lines = block.split("\n")
    for i, ln in enumerate(lines):
        head = ln.strip().split(".", 1)[0]
        if head.isdigit() and "." in ln.strip():
            lines.insert(i, LESSON_FRAMING)
            return "\n".join(lines)
    return block


def cognition_block(block: str, max_chars: int = 700) -> str:
    if not block or not block.strip():
        return ""
    if len(block) > max_chars:
        block = truncate_cognition_block(block, max_chars)
    return with_lesson_framing(block)


def compose(memory: str = "", thinking: str = "", cognition: str = "") -> str:
    parts = [p for p in (memory, thinking, cognition) if p]
    return "\n\n".join(parts)
