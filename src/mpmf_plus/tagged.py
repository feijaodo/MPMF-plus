"""r1: ``render_format = "tagged"`` renderers (flag-gated; the default ``"legacy"`` never calls this).

Sections, in injection order, and the tags they use::

    <memory_facts query=".." hits="N">   one <fact ...>text</fact> per recalled fact
    <history entity="..">                dated list, one element per entity (history mode)
    <verbatim>                           the user's own raw messages (raw_layer)
    <prior_reasoning>                    reasoning traces (what legacy calls <thinking-context>)
    <lessons>                            distilled lessons (what legacy calls <prior-reasoning>)

Every section is well-formed XML fragment text: tags closed, attribute values escaped, text
escaped (``& < >``), characters illegal in XML 1.0 dropped. Each section is fitted to its OWN
character budget by dropping whole items (never cutting through a tag); a section that dropped
items says so with ``truncated="true"``.

Fact metadata is attached only where the store has it; a missing field is omitted, never
invented:  said (``_said_at``), said_src (``_said_at_source``, only when stored), fact_date
(``_fact_date``), kind (``_life_kind`` else ``kind``), mode (``_life_mode`` else ``_fact_mode``),
score (the recall score of this hit). ``source`` (user/assistant) is attached only for the
verbatim section, which by construction holds the user's own messages; the fact store has no
speaker field, so facts carry no ``source``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.sax.saxutils import escape

_ILLEGAL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")
_ATTR_ESC = {'"': "&quot;", "\n": "&#10;", "\r": "&#13;", "\t": "&#9;"}
TRUNC = ' truncated="true"'

LESSON_ITEM = re.compile(r"^(\d+)\.(?: \[([^\]]+)\])? (.*)$")


def clean(s: Any) -> str:
    return _ILLEGAL.sub("", str(s))


def esc(s: Any) -> str:
    """Element text."""
    return escape(clean(s))


def attr(s: Any) -> str:
    """Attribute value (to be placed inside double quotes)."""
    return escape(clean(s), _ATTR_ESC)


def attrs(pairs: Iterable[Tuple[str, Any]]) -> str:
    return "".join(' %s="%s"' % (k, attr(v)) for k, v in pairs if v not in (None, ""))


def fact_pairs(wiki: Dict[str, Any], score: Optional[float] = None) -> List[Tuple[str, Any]]:
    """Stored metadata of one fact, in a fixed order; absent fields are absent."""
    w = wiki or {}
    said = str(w.get("_said_at") or "")[:10]
    fdate = str(w.get("_fact_date") or "")[:10]
    kind = w.get("_life_kind") or w.get("kind") or ""
    mode = w.get("_life_mode") or w.get("_fact_mode") or ""
    src = w.get("_said_at_source") or ""
    sc = ""
    if score is not None:
        try:
            sc = "%.4f" % float(score)
        except Exception:
            sc = ""
    return [("said", said), ("said_src", src), ("fact_date", fdate),
            ("kind", kind), ("mode", mode), ("score", sc)]


def _wrap(open_tag: str, close_tag: str, lines: List[str], dropped: bool) -> str:
    if dropped:
        open_tag = open_tag[:-1] + TRUNC + ">"
    return "\n".join([open_tag] + lines + [close_tag])


# -- facts -------------------------------------------------------------------------------------
def facts(query: str, entries: List[Dict[str, Any]], n_hits: int, cap: int) -> str:
    """entries: dicts with nid, name, text, wiki, score (already item-capped). '' if none fit."""
    head = '<memory_facts query="%s" hits="%d">' % (attr(query[:120]), n_hits)
    close = "</memory_facts>"
    used = len(head) + len(TRUNC) + len(close) + 2
    lines: List[str] = []
    dropped = False
    for e in entries:
        pairs = [("id", e.get("nid")), ("name", e.get("name") or "")] + fact_pairs(
            e.get("wiki") or {}, e.get("score"))
        line = "<fact%s>%s</fact>" % (attrs(pairs), esc(e.get("text") or ""))
        if used + len(line) + 1 > cap:
            dropped = True
            break
        lines.append(line)
        used += len(line) + 1
    if not lines:
        return ""
    return _wrap(head, close, lines, dropped or len(entries) > len(lines))


# -- history -----------------------------------------------------------------------------------
def history(query: str, result: Any, memory_manager: Any, idx: Any, top_k: int, cap: int) -> str:
    """Tagged twin of history.render_history_block: same entity pick, same relevance-then-date
    selection, same 4-entity / 10-fact / 180-char limits; one <history entity> per entity."""
    from . import history as H
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
    parts: List[str] = []
    used, n = 0, 0
    for ent in ents[:4]:
        cands = idx.facts_for(ent)
        if not cands:
            continue
        picked = H.select_relevant(query, cands, memory_manager, keep=10)
        head = '<history entity="%s">' % attr(ent)
        close = "</history>"
        u = used + len(head) + len(TRUNC) + len(close) + 2
        if u > cap:
            break
        lines: List[str] = []
        dropped = False
        for d, nid in picked:
            try:
                text = (memory_manager.get_text(nid) or "").strip().replace("\n", " ")
            except Exception:
                text = ""
            if len(text) < 5:
                continue
            try:
                w = memory_manager.get_wiki(nid) or {}
            except Exception:
                w = {}
            line = "<fact%s>%s</fact>" % (
                attrs([("date", d)] + fact_pairs(w)), esc(text[:180]))
            if u + len(line) + 1 > cap:
                dropped = True
                break
            lines.append(line)
            u += len(line) + 1
        if lines:
            parts.append(_wrap(head, close, lines, dropped))
            used = u
            n += len(lines)
        if n >= top_k:
            break
    return "\n".join(parts) if n else ""


# -- verbatim ----------------------------------------------------------------------------------
def verbatim(hits: List[Dict[str, Any]], item_chars: int, cap: int) -> str:
    """The user's own raw messages (source is 'user' by construction of the raw layer)."""
    if not hits:
        return ""
    head, close = "<verbatim>", "</verbatim>"
    used = len(head) + len(TRUNC) + len(close) + 2
    lines: List[str] = []
    dropped = False
    for h in hits:
        text = " ".join((h.get("text") or "").split())
        if len(text) > item_chars:
            text = text[:item_chars] + "…"
        rr = h.get("reranker_score")
        pairs = [("said", h.get("date") or ""), ("source", "user"),
                 ("cos", "%.4f" % float(h["cos"]) if h.get("cos") is not None else ""),
                 ("rerank", "%.4f" % float(rr) if rr is not None else "")]
        line = "<message%s>%s</message>" % (attrs(pairs), esc(text))
        if used + len(line) + 1 > cap:
            dropped = True
            break
        lines.append(line)
        used += len(line) + 1
    return _wrap(head, close, lines, dropped or len(hits) > len(lines)) if lines else ""


# -- thinking -> <prior_reasoning> ---------------------------------------------------------------
def prior_reasoning(results: List[Dict[str, Any]], cap: int) -> str:
    if not results:
        return ""
    head, close = "<prior_reasoning>", "</prior_reasoning>"
    used = len(head) + len(TRUNC) + len(close) + 2
    lines: List[str] = []
    dropped = False
    for r in results:
        try:
            ctx = (r.get("context") or "").strip()
            concl = (r.get("conclusion") or "").strip()
            outcome = r.get("outcome") or "unknown"
            score = r.get("score")
        except Exception:
            continue
        if not ctx and not concl:
            continue
        pairs = [("outcome", outcome if outcome not in ("unknown", "") else ""),
                 ("score", "%.4f" % float(score) if isinstance(score, (int, float)) else "")]
        line = "<trace%s><context>%s</context>%s</trace>" % (
            attrs(pairs), esc(ctx[:140]),
            "<conclusion>%s</conclusion>" % esc(concl[:200]) if concl else "")
        if used + len(line) + 1 > cap:
            dropped = True
            break
        lines.append(line)
        used += len(line) + 1
    return _wrap(head, close, lines, dropped or len(results) > len(lines)) if lines else ""


# -- lessons -----------------------------------------------------------------------------------
def lessons(raw_block: str, cap: int, framing: str) -> str:
    """Re-tag the provider's numbered <prior-reasoning> lesson block as <lessons>."""
    if not raw_block or not raw_block.strip():
        return ""
    OPEN, CLOSE = "<prior-reasoning>", "</prior-reasoning>"
    inner = raw_block.strip()
    if inner.startswith(OPEN):
        inner = inner[len(OPEN):]
    if inner.endswith(CLOSE):
        inner = inner[:-len(CLOSE)]
    pre: List[str] = []
    items: List[List[Any]] = []          # [n, status, [text lines]]
    for ln in inner.strip("\n").split("\n"):
        m = LESSON_ITEM.match(ln)
        if m and int(m.group(1)) == len(items) + 1:
            items.append([int(m.group(1)), m.group(2) or "", [m.group(3)]])
        elif items:
            items[-1][2].append(ln)
        elif ln.strip():
            pre.append(ln.strip())
    if not items:
        return ""
    head, close = "<lessons>", "</lessons>"
    preamble = [esc(p) for p in pre if framing not in p] + [esc(framing)]
    used = len(head) + len(TRUNC) + len(close) + 2 + sum(len(p) + 1 for p in preamble)
    lines: List[str] = []
    dropped = False
    for n, status, tl in items:
        text = esc("\n".join(tl).strip())
        line = "<lesson%s>%s</lesson>" % (attrs([("n", n), ("status", status)]), text)
        if used + len(line) + 1 > cap:
            dropped = True
            room = cap - used - len('<lesson n="1">') - len("</lesson>") - 2
            if not lines and room > 60:      # keep one (cut) lesson rather than none
                cut = esc("\n".join(tl).strip())[:room].rstrip()
                cut = re.sub(r"&[a-z#0-9]*$", "", cut)   # never cut inside an entity
                lines.append("<lesson%s>%s…</lesson>" % (attrs([("n", n), ("status", status)]), cut))
            break
        lines.append(line)
        used += len(line) + 1
    if not lines:
        return ""
    return _wrap(head, close, preamble + lines, dropped)
