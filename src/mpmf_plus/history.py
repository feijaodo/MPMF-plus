"""History mode: entity -> dated-fact index, relevance-then-date rendering.

WHY
---
Temporal questions ("what did I say first about X", "what changed between A
and B") fail when facts arrive in relevance order: the top-k cut drops the
early or late members of a sequence, so the model sees a fragment and answers
in the wrong order. Keeping facts in date order fixes the ordering; keeping
only the facts *relevant to the question* before sorting keeps the 1,500-char
cap from being spent on unrelated history.

DESIGN
------
* ``HistoryIndex`` is built once at prewarm from the store (entity, date, id)
  and appended on every write, so a temporal query is a dict lookup plus a
  sort — well under the 300 ms budget.
* ``is_temporal()`` gates the whole path. When it is false the caller renders
  exactly as before, which is what keeps non-temporal blocks byte-identical.
* Selection is relevance-first: candidates are scored against the question,
  the best few are kept, and only then are they sorted by date.

Off by default: ``history_mode`` in recall.config.json or ``MPMF_HISTORY_MODE=1``.
"""
from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

TEMPORAL = re.compile(
    r"\b(when|before|after|first|last|earliest|latest|earlier|later|"
    r"order|sequence|chronolog\w*|timeline|initial\w*|final\w*|time|"
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|"
    r"sept|oct|nov|dec|change[sd]?|evolv\w*|progress|used to|originally|"
    r"anymore|still|history)\b", re.I)

STOP = set("""a an the and or but if then than that this these those i you he she it we they
me my your his her its our their of to in on at for with about from by as is are was were be
been being do does did have has had what which who whom how why when where any some all
""".split())


def enabled() -> bool:
    """history_mode: env wins, then recall.config.json, default OFF."""
    v = os.environ.get("MPMF_HISTORY_MODE")
    if v is not None:
        return v.strip().lower() in ("1", "true", "yes", "on")
    try:
        import json as _json
        from . import block as _b
        with open(_b.RECALL_CFG_PATH) as f:
            cfg = _json.load(f) or {}
        return bool(cfg.get("history_mode"))
    except Exception:
        return False


def is_temporal(query: str) -> bool:
    return bool(TEMPORAL.search(query or ""))


def fact_date(wiki: Dict[str, Any]) -> str:
    """Best available date for a fact, as YYYY-MM-DD ('' when unknown)."""
    for k in ("_fact_date", "_said_at", "_conv_date"):
        v = str((wiki or {}).get(k) or "")
        if len(v) >= 10:
            return v[:10]
    return ""


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 2 and w not in STOP}


class HistoryIndex:
    """entity -> [(date, node_id)] plus a reverse map for cheap appends."""

    def __init__(self) -> None:
        self.by_entity: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.entity_of: Dict[int, str] = {}
        self.built_ms: float = 0.0
        self.built_at: float = 0.0

    # -- build / maintain ------------------------------------------------
    def build(self, memory_manager: Any, node_count: Optional[int] = None) -> "HistoryIndex":
        t0 = time.perf_counter()
        n = node_count or getattr(getattr(memory_manager, "header", None), "node_count", 0)
        for nid in range(1, int(n) + 1):
            self.add(memory_manager, nid)
        self.built_ms = (time.perf_counter() - t0) * 1000
        self.built_at = time.time()
        return self

    def add(self, memory_manager: Any, nid: int) -> None:
        try:
            wiki = memory_manager.get_wiki(nid) or {}
        except Exception:
            return
        ent = str(wiki.get("entity") or "").strip().lower()
        if not ent:
            return
        d = fact_date(wiki)
        if not d:
            return
        self.entity_of[nid] = ent
        self.by_entity[ent].append((d, nid))

    def ensure(self, memory_manager: Any) -> "HistoryIndex":
        """Lazily build once, then keep up with new nodes."""
        n = getattr(getattr(memory_manager, "header", None), "node_count", 0)
        if not self.built_at:
            return self.build(memory_manager, n)
        known = max(self.entity_of) if self.entity_of else 0
        for nid in range(known + 1, int(n) + 1):
            self.add(memory_manager, nid)
        return self

    # -- query -----------------------------------------------------------
    def facts_for(self, entity: str) -> List[Tuple[str, int]]:
        return self.by_entity.get((entity or "").strip().lower(), [])


_INDEX: Optional[HistoryIndex] = None


def index() -> HistoryIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = HistoryIndex()
    return _INDEX


def reset() -> None:
    global _INDEX
    _INDEX = None


def select_relevant(query: str, candidates: Iterable[Tuple[str, int]], memory_manager: Any,
                    keep: int = 12) -> List[Tuple[str, int]]:
    """Score facts against the question, keep the best, THEN sort by date."""
    qt = _tokens(query)
    scored = []
    for order, (d, nid) in enumerate(candidates):
        try:
            text = memory_manager.get_text(nid) or ""
        except Exception:
            text = ""
        overlap = len(qt & _tokens(text))
        # ties keep recall order, which is the model's own ranking
        scored.append((overlap, -order, d, nid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked = [(d, nid) for _s, _o, d, nid in scored[:keep]]
    picked.sort(key=lambda x: (x[0], x[1]))          # date asc, then id
    return picked


def render_history_block(query: str, result: Any, memory_manager: Any,
                         idx: HistoryIndex, top_k: int = 8,
                         max_chars: int = 1500, elapsed: float = 0.0) -> str:
    """Date-ordered block for a temporal query, capped at max_chars."""
    top = list(getattr(result, "top_nodes", None) or [])[:top_k]
    if not top or memory_manager is None:
        return ""
    ents, seen = [], set()
    for nid in top:
        try:
            ent = str((memory_manager.get_wiki(nid) or {}).get("entity") or "")
        except Exception:
            ent = ""
        if ent and ent.lower() not in seen:
            seen.add(ent.lower())
            ents.append(ent)

    lines = [f"# MPMF+ recall for: {query[:120]!r}",
             f"(history · {len(ents)} entities · {elapsed:.2f}s)", ""]
    used, n = len("\n".join(lines)), 0
    for ent in ents[:4]:
        cands = idx.facts_for(ent)
        if not cands:
            continue
        picked = select_relevant(query, cands, memory_manager, keep=10)
        head = f"## {ent}"
        if used + len(head) + 1 > max_chars:
            break
        lines.append(head)
        used += len(head) + 1
        for d, nid in picked:
            try:
                text = (memory_manager.get_text(nid) or "").strip().replace("\n", " ")
            except Exception:
                text = ""
            if len(text) < 5:
                continue
            entry = f"- {d} · {text[:180]}"
            if used + len(entry) + 1 > max_chars:
                lines[-1] = lines[-1]
                break
            lines.append(entry)
            used += len(entry) + 1
            n += 1
        if n >= top_k:
            break
    if n == 0:
        return ""
    return "\n".join(lines)
