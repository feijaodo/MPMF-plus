"""Second-pass retrieval (P18, RESEARCH): let the model ask MPMF for more WHILE it reasons.

WHY
---
Prefetch runs once, before the turn, with the user's raw message as the query. When that
query is vague ("what did I say about that car thing?") or the answer sits behind a
different wording, the block misses it and the model can only guess or decline. The model
is the one party that knows, after reading the block, WHAT is missing.

WHAT (flag ``second_pass``, default OFF)
----------------------------------------
Flag off: ``get_tool_schemas`` / ``handle_tool_call`` are exactly what they were before.
Flag on:
* the ``mpmf_recall`` tool description tells the model to call it when the injected context
  lacks what it needs, with a specific re-worded query (names, dates, topic);
* a query call returns the SAME rendering as prefetch (same caps, same ``## [id] name``
  headings, history ordering for temporal queries) instead of 300-char snippets;
* memories already shown in this turn's prefetch block (and in earlier second-pass calls of
  the turn) are never returned again, so every call can only ADD;
* at most ``second_pass_max_calls`` (2) query calls per turn (reset by the next prefetch);
  past that the tool says so and returns nothing.
``node_id`` lookups are unchanged.

Config (recall.config.json or ``MPMF_<KEY>``): second_pass (false), second_pass_max_calls (2),
second_pass_k (6).
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, Set

from . import paths as _paths

#: info about the most recent call (read by the eval harness; never used for logic)
LAST: Dict[str, Any] = {}

DESCRIPTION = (
    "Second-pass lookup against the MPMF+ memory graph. The <memory-context> injected at the "
    "start of the turn came from your user's raw message only. If it does not contain what you "
    "need (a name, a date, a number, an earlier decision), call this BEFORE answering, declining "
    "or guessing, with a specific query in different words than the user used (names, dates, "
    "topic). It returns only memories not already in your context. Use node_id to open one "
    "memory in full.")


def settings() -> Dict[str, Any]:
    c = _paths.load_config()
    return {
        "on": bool(c.get("second_pass", False)),
        "max_calls": max(1, int(c.get("second_pass_max_calls", 2) or 2)),
        "k": max(1, int(c.get("second_pass_k", 6) or 6)),
    }


def enabled() -> bool:
    try:
        return bool(settings()["on"])
    except Exception:
        return False


class TurnState:
    """Per-provider state for one turn: nodes already shown + calls made."""

    def __init__(self) -> None:
        self.seen: Set[Any] = set()
        self.calls = 0

    def reset(self, shown=()) -> None:
        self.seen = set(shown)
        self.calls = 0


def query(state: TurnState, text: str, top_k: Optional[int] = None, *, provider_mod: Any,
          max_chars: int = 3500) -> str:
    """Run one second-pass query; returns the rendered block (or a short notice)."""
    from . import block as _block
    from . import rendering as _rendering
    st = settings()
    text = (text or "").strip()
    if not text:
        return json.dumps({"error": "query is required"})
    if state.calls >= st["max_calls"]:
        LAST.clear()
        LAST.update({"refused": True, "calls": state.calls})
        return ("Second-pass limit reached (%d lookups this turn). Answer from what you have."
                % st["max_calls"])
    state.calls += 1
    t0 = time.perf_counter()
    k = max(1, min(int(top_k or st["k"]), 12))
    qvec = None
    try:
        qvec = provider_mod.embed_query(text)
    except Exception:
        qvec = None
    shim = _block.wrapper_recall(text, min(k + len(state.seen), 30), provider_mod, query_vec=qvec)
    if shim is None:
        raise RuntimeError("second-pass recall failed")
    fresh = [n for n in shim.top_nodes if n not in state.seen]
    shim.top_nodes = fresh
    mem = None
    try:
        mem = provider_mod._open().mem
    except Exception:
        mem = None
    body = _rendering.render_memory(text, shim, mem, k, max_chars=max_chars) if fresh else ""
    shown = list(fresh[:k]) if body else []
    state.seen.update(shown)
    LAST.clear()
    LAST.update({"refused": False, "calls": state.calls, "returned": len(shown),
                 "candidates": len(shim.top_nodes), "ms": (time.perf_counter() - t0) * 1000.0})
    if not body:
        return "No new memories for that query beyond those already in your context."
    return body
