"""Block builder — ported from the plugin's render path (r59d).

Source: the mpmf_plus_user plugin (pre-r89)
  _render_cfg        lines 728-757  (verbatim, self._max_chars -> max_chars arg)
  _wrapper_recall    lines 509-560  (verbatim, module lookup -> explicit arg)
  memory render      lines 827-880  (verbatim, self.* -> explicit args)

No logic changes: same caps, same "…" item truncation, same block-cap marker,
same "## [id] name" heading format, same "…" fallback to "(node)", same order.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import paths as _paths
from .errors import ConfigError


def recall_cfg_path() -> Path:
    """The recall config this package reads — resolved on EVERY call (r103).

    There is exactly one resolver for it (``paths.config_path()``), so
    ``MPMF_RECALL_CONFIG`` / ``MPMF_HOME`` changes are picked up without a
    restart, and no module can hold a stale import-time copy.
    """
    return _paths.config_path()


class RecallShim:
    """Same shape the plugin's _wrapper_recall returns."""

    def __init__(self) -> None:
        self.top_nodes: List[Any] = []
        self.raw_texts: Dict[Any, str] = {}
        self.summaries: Dict[Any, str] = {}
        self.scores: Dict[Any, Any] = {}      # r1: recall score per node (tagged format only)


def render_cfg(max_chars: int = 3500) -> Dict[str, Any]:
    """Round 56 render mode, read on EVERY call. Unreadable/invalid -> ConfigError."""
    # r59d: an unreadable/invalid config RAISES. It does not silently degrade to
    # "summary" — that fallback is exactly what hid the r56 render switch for a day.
    # r103: the read itself goes through paths.load_config(strict=True), the ONE reader.
    # A missing file still means "package defaults" (portable install).
    cfg = _paths.load_config(strict=True)
    if not isinstance(cfg, dict):                      # defensive: load_config guarantees a dict
        raise ConfigError(f"recall config is not an object: {recall_cfg_path()}")
    mode = str(cfg.get("memory_render", "summary")).strip().lower()
    if mode not in ("summary", "full"):
        raise ConfigError(f"memory_render must be 'summary' or 'full', got {mode!r}")
    d_item = 1200 if mode == "full" else 600
    d_block = 3000 if mode == "full" else max_chars
    try:
        item_cap = int(cfg.get("memory_item_chars", d_item))
    except Exception:
        item_cap = d_item
    try:
        block_cap = int(cfg.get("memory_block_chars", d_block))
    except Exception:
        block_cap = d_block
    return {"mode": mode, "item_cap": item_cap, "block_cap": block_cap,
            # r112: this key was read at line ~194 but never put in the dict, so the
            # switch did nothing. Strings are parsed ("false" is not True).
            "show_said_date": (
                _paths._as_bool(cfg["show_said_date"])
                if isinstance(cfg.get("show_said_date"), str)
                else bool(cfg.get("show_said_date", False)))}


def wrapper_recall(query: str, top_k: int, memory_module: Any,
                   query_vec: Optional[List[float]] = None,
                   budget_start: Optional[float] = None) -> Optional[RecallShim]:
    """Namespace-aware recall in the top_nodes/raw_texts/summaries shape.

    An EMPTY result means "nothing relevant" and is returned as an empty shim —
    it is never treated as an error and never falls back to anything else.
    A raise propagates to the caller as None.
    """
    try:
        _kw = {"query_vec": query_vec} if query_vec is not None else {}
        if budget_start is not None:
            _kw["budget_start"] = budget_start
        rows = memory_module.recall(query, top_n=top_k, **_kw) or []
        shim = RecallShim()
        shim.top_nodes = [r.get("node_id") for r in rows]
        shim.raw_texts = {r.get("node_id"): (r.get("text") or "") for r in rows}
        shim.summaries = {r.get("node_id"): (r.get("summary") or r.get("text") or "")
                          for r in rows}
        shim.scores = {r.get("node_id"): r.get("score") for r in rows}
        return shim
    except Exception:
        return None


_FORMATS = ("legacy", "tagged")


def _opt_int(v: Any) -> Optional[int]:
    """A per-section budget: None / '' / junk / <= 0 means 'unset' (= the legacy cap)."""
    try:
        if v is None or v == "":
            return None
        n = int(v)
        return n if n > 0 else None
    except Exception:
        return None


def _opt_float(v: Any) -> Optional[float]:
    """A cosine floor: None / '' / junk / <= 0 means 'unset' (no filtering)."""
    try:
        if v is None or v == "":
            return None
        f = float(v)
        return f if f > 0 else None
    except Exception:
        return None


def injection_opts(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """r1 flags from a config dict (default: the recall config file, read on every call).

    Never raises on a missing/unreadable file (history never did); an INVALID render_format
    raises ConfigError, like memory_render does.
    """
    if cfg is None:
        cfg = _paths.load_config()
    fmt = str(cfg.get("render_format", "legacy") or "legacy").strip().lower()
    if fmt not in _FORMATS:
        raise ConfigError(f"render_format must be one of {_FORMATS}, got {fmt!r}")
    h = _opt_int(cfg.get("history_max_chars"))
    if isinstance(cfg.get("history_additive"), str):
        add = _paths._as_bool(cfg["history_additive"])
    else:
        # r99: default ON (additive). A missing key means "facts + dated list", never
        # "dated list instead of the facts" — the replace mode is what lost DMR/BEAM.
        add = bool(cfg.get("history_additive", True))
    return {"render_format": fmt, "history_additive": add,
            "facts_max_chars": _opt_int(cfg.get("facts_max_chars")),
            "history_max_chars": h if h is not None else 1500,
            "verbatim_max_chars": _opt_int(cfg.get("verbatim_max_chars")),
            "lessons_max_chars": _opt_int(cfg.get("lessons_max_chars")),
            # r101 (0.3.2), all default OFF/None: defaults unchanged.
            # r112: a string "false" was read through bool() and became True.
            "history_grounded": (_paths._as_bool(cfg["history_grounded"])
                                 if isinstance(cfg.get("history_grounded"), str)
                                 else bool(cfg.get("history_grounded", False))),
            "facts_dedupe": (_paths._as_bool(cfg["facts_dedupe"])
                             if isinstance(cfg.get("facts_dedupe"), str)
                             else bool(cfg.get("facts_dedupe", False))),
            "facts_min_sim": _opt_float(cfg.get("facts_min_sim"))}


def _facts_section(query: str, result: Any, memory_manager: Any, top_k: int,
                   elapsed: float, max_chars: int, tagged: bool,
                   facts_cap: Optional[int]):
    """The facts block. Returns (body, items). Legacy bytes are unchanged."""
    items: List[Dict[str, Any]] = []
    entries: List[Dict[str, Any]] = []
    lines: List[str] = []
    lines.append(f"# MPMF+ recall for: {query[:120]!r}")
    lines.append(f"({len(result.top_nodes)} top hits, {elapsed:.2f}s)")
    lines.append("")
    raw = getattr(result, "raw_texts", {}) or {}
    summaries = getattr(result, "summaries", {}) or {}
    scores = getattr(result, "scores", {}) or {}
    _rc = render_cfg(max_chars)
    _full = _rc["mode"] == "full"
    n_added = 0
    for nid in result.top_nodes:
        if n_added >= top_k:
            break
        summary = summaries.get(nid) or raw.get(nid) or ""
        if _full:
            summary = raw.get(nid) or summary
        if not summary or len(summary.strip()) < 5:
            continue
        name = ""
        try:
            if memory_manager is not None:
                name = memory_manager.get_name(nid) or ""
        except Exception:
            name = ""
        entry = summary.strip()
        if len(entry) > _rc["item_cap"]:
            entry = entry[: _rc["item_cap"]] + "…"
        # r65: show when it was said. A memory with no known saying time says
        # so rather than borrowing its ingest date, which would read as recent.
        said, said_src = "", ""
        _w: Dict[str, Any] = {}
        try:
            if memory_manager is not None:
                _w = memory_manager.get_wiki(nid) or {}
                said = str(_w.get("_said_at") or "")
                said_src = str(_w.get("_said_at_source") or "")
        except Exception:
            said, said_src = "", ""
            _w = {}
        head = f"## [{nid}] {name or '(node)'}"
        # r89c parity: the date suffix is behind a flag, default OFF, so the
        # package renderer is byte-identical to the pre89 plugin. Turning it on
        # is a prompt change and needs its own parity run.
        if said and bool(_rc.get("show_said_date", False)):
            head += f" · said {said[:10]}"
            if said_src == "ingest":
                head += " (ingest date, not when it was said)"
        lines.append(head)
        lines.append(entry)
        lines.append("")
        n_added += 1
        items.append({"node_id": nid, "name": name, "text": entry})
        entries.append({"nid": nid, "name": name, "text": entry, "wiki": _w,
                        "score": scores.get(nid)})
    if n_added == 0:
        return "", items
    if tagged:
        from . import tagged as _tg
        return _tg.facts(query, entries, len(result.top_nodes),
                         facts_cap if facts_cap is not None else _rc["block_cap"]), items
    body = "\n".join(lines).strip()
    cap = facts_cap if facts_cap is not None else _rc["block_cap"]
    if len(body) > cap:
        body = body[: cap] + "\n\n[... MPMF+ recall truncated at char cap ...]"
    return body, items


def _cosine_map(result: Any, memory_manager: Any, query_vec: Any) -> Dict[Any, float]:
    """r102: tagged 'score' is the fused recall score (rank-like); show cosine instead."""
    import math
    out: Dict[Any, float] = {}
    qn = math.sqrt(sum(a * a for a in query_vec)) or 1.0
    for nid in list(getattr(result, "top_nodes", None) or []):
        try:
            v = memory_manager.get_vector_list(nid)
        except Exception:
            v = None
        if not v:
            continue
        dot = sum(a * b for a, b in zip(query_vec, v))
        vn = math.sqrt(sum(b * b for b in v)) or 1.0
        out[nid] = dot / (qn * vn)
    return out


def _filter_min_sim(result: Any, memory_manager: Any, query_vec: Any, floor: float) -> Any:
    """r101: keep only nodes whose cosine(query, node) >= floor; a copy, never a mutation."""
    import copy
    import math
    keep = []
    qn = math.sqrt(sum(a * a for a in query_vec)) or 1.0
    for nid in list(getattr(result, "top_nodes", None) or []):
        try:
            v = memory_manager.get_vector_list(nid)
        except Exception:
            v = None
        if not v:
            continue
        dot = sum(a * b for a, b in zip(query_vec, v))
        vn = math.sqrt(sum(b * b for b in v)) or 1.0
        if dot / (qn * vn) >= floor:
            keep.append(nid)
    r = copy.copy(result)
    try:
        r.top_nodes = keep
    except Exception:
        return result
    return r


def render_memory_sections(query: str, result: Any, memory_manager: Any, top_k: int,
                           elapsed: float = 0.0, max_chars: int = 3500,
                           opts: Optional[Dict[str, Any]] = None,
                           query_vec: Optional[Any] = None):
    """Return ({"facts": str, "history": str}, items): the memory part, one string per section.

    Legacy (all r1/r101 flags at their defaults): a temporal question that fires history gets
    {"facts": "", "history": <block>} (history REPLACES the facts block), exactly as before.
    history_additive: facts are rendered exactly as with history off, history is added.
    r101: history_grounded (only entities named in the query), facts_dedupe (a fact already
    in the facts section is not repeated in history), facts_min_sim (cosine floor).
    """
    secs = {"facts": "", "history": ""}
    items: List[Dict[str, Any]] = []
    if not result or not getattr(result, "top_nodes", None):
        return secs, items
    o = opts if opts is not None else injection_opts()
    # r101: cosine floor on the recalled nodes, before anything is rendered.
    if memory_manager is not None and o.get("facts_min_sim") and query_vec is not None:
        result = _filter_min_sim(result, memory_manager, query_vec, float(o["facts_min_sim"]))
        if not getattr(result, "top_nodes", None):
            return secs, items
    if query_vec is not None and o["render_format"] == "tagged":
        # r102: the tagged "score" attribute is the fused recall score, not a similarity.
        _cs = _cosine_map(result, memory_manager, query_vec)
        if _cs:
            import copy as _copy
            result = _copy.copy(result)
            result.scores = _cs
    fires, _idx = False, None
    if memory_manager is not None:
        try:
            from . import history as _hist
            fires = bool(_hist.enabled() and _hist.is_temporal(query))
            if fires:
                _idx = _hist.index().ensure(memory_manager)
        except Exception:
            fires, _idx = False, None
    # Replace layout only when nothing r101 needs the facts section first.
    replace_mode = fires and not o["history_additive"] and o["render_format"] != "tagged" \
        and not o["history_grounded"]
    if replace_mode:
        try:
            from . import history as _hist
            body = _hist.render_history_block(query, result, memory_manager, _idx, top_k=top_k,
                                              max_chars=o["history_max_chars"], elapsed=elapsed)
            if body:
                secs["history"] = body
                return secs, items
        except Exception:
            pass
    body, items = _facts_section(query, result, memory_manager, top_k, elapsed, max_chars,
                                 o["render_format"] == "tagged", o["facts_max_chars"])
    secs["facts"] = body
    if fires:
        exclude = None
        if o["facts_dedupe"]:
            exclude = {re.sub(r"\W+", " ", it["text"].lower()).strip()[:120] for it in items}
        try:
            from . import history as _hist
            if o["render_format"] == "tagged":
                from . import tagged as _tg
                hbody = _tg.history(query, result, memory_manager, _idx, top_k,
                                    o["history_max_chars"], grounded=o["history_grounded"],
                                    exclude=exclude)
            else:
                hbody = _hist.render_history_block(
                    query, result, memory_manager, _idx, top_k=top_k,
                    max_chars=o["history_max_chars"], elapsed=elapsed,
                    title=(f"# MPMF+ history for: {query[:120]!r}" if o["history_additive"] else None),
                    grounded=o["history_grounded"], exclude=exclude)
            if hbody:
                secs["history"] = hbody
        except Exception:
            pass
    return secs, items


def render_memory_block(query: str, result: Any, memory_manager: Any, top_k: int,
                        elapsed: float = 0.0, max_chars: int = 3500,
                        opts: Optional[Dict[str, Any]] = None):
    """Return (body, items). Faithful port of prefetch() lines 827-880 (+ r1 flags)."""
    secs, items = render_memory_sections(query, result, memory_manager, top_k,
                                         elapsed=elapsed, max_chars=max_chars, opts=opts)
    return "\n\n".join(p for p in (secs["facts"], secs["history"]) if p), items
