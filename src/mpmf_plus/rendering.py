"""Block rendering: memory, thinking, prior-reasoning.

Ported 1:1 from mpmf_plus_user._recall_thinking_block / _recall_cognition_block
so the injected bytes are unchanged. No host imports.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from . import block as _block

LESSON_FRAMING = (
    "Past lessons that may or may not apply. Use one only if this "
    "situation clearly matches its trigger; otherwise ignore them all. "
    "Never mention them."
)

# r103: a cognition item starts at a numbered line ("1. ", "12. ") — same shape the
# lesson framing function already looks for.
_COG_ITEM = re.compile(r"^\s*\d+\.\s")


def render_memory(query: str, result: Any, memory_manager: Any, top_k: int = 8,
                  max_chars: int = 3500, elapsed: float = 0.0, opts=None) -> str:
    body, _items = _block.render_memory_block(
        query, result, memory_manager, top_k, elapsed=elapsed, max_chars=max_chars,
        opts=opts)
    return body or ""


def render_memory_sections(query: str, result: Any, memory_manager: Any, top_k: int = 8,
                           max_chars: int = 3500, elapsed: float = 0.0, opts=None,
                           query_vec=None) -> Dict[str, str]:
    """r1: the memory part as {"facts": str, "history": str} (legacy: history replaces facts)."""
    secs, _items = _block.render_memory_sections(
        query, result, memory_manager, top_k, elapsed=elapsed, max_chars=max_chars, opts=opts,
        query_vec=query_vec)
    return secs


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
            # legacy path stays byte-compatible with the pinned golden; the no-mid-text-cut
            # rule is enforced in the tagged renderers (tagged.prior_reasoning/lessons) and
            # in truncate_cognition_block. This branch must BREAK: without it the truncated
            # entry was followed by the full one, so the cap was exceeded (r103 regression).
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


def _split_cog_items(inner: str):
    """(header_lines, [item, ...]) for a cognition block body.

    An item starts at a numbered line ("1. ", "12. ") and OWNS every following
    non-numbered line (continuations), so dropping an item drops its whole text.
    """
    head, items = [], []
    for ln in inner.split("\n"):
        if _COG_ITEM.match(ln):
            items.append([ln])
        elif items:
            items[-1].append(ln)
        else:
            head.append(ln)
    return head, ["\n".join(it) for it in items]


def truncate_cognition_block(block: str, max_chars: int) -> str:
    """Cap a <prior-reasoning> block by DROPPING WHOLE ITEMS — never slicing one.

    r103: the old body was ``inner[:keep]``, which cut mid-item — a block ending
    "## 1. Ov", i.e. half a sentence injected as if it were a lesson. Now the header
    stays, items are added whole while they fit, and the first item that does not fit
    is dropped (with a marker). If not even one item fits, the result is empty: a
    header with no lessons is noise, and a cut lesson is misinformation.
    """
    OPEN, CLOSE = "<prior-reasoning>", "</prior-reasoning>"
    MARK = "\n[... prior-reasoning truncated ...]"
    inner = block[len(OPEN):-len(CLOSE)] if block.startswith(OPEN) and block.endswith(CLOSE) else block
    head, items = _split_cog_items(inner)
    budget = max_chars - len(OPEN) - len(CLOSE) - len(MARK) - 1
    head_txt = "\n".join(head)
    kept, used = [], (len(head_txt) + 1 if head else 0)
    for it in items:
        if used + len(it) + 1 > budget:
            continue                    # a later, shorter item may still fit
        kept.append(it)
        used += len(it) + 1
    if not kept:
        return ""
    body = "\n".join(([head_txt] if head else []) + kept)
    dropped = len(kept) < len(items)
    return OPEN + body + (MARK if dropped else "") + CLOSE


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
