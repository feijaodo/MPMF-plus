"""Block builder — ported from the plugin's render path (r59d).

Source: the mpmf_plus_user plugin (pre-r89)
  _render_cfg        lines 728-757  (verbatim, self._max_chars -> max_chars arg)
  _wrapper_recall    lines 509-560  (verbatim, module lookup -> explicit arg)
  memory render      lines 827-880  (verbatim, self.* -> explicit args)

No logic changes: same caps, same "…" item truncation, same block-cap marker,
same "## [id] name" heading format, same "…" fallback to "(node)", same order.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import ConfigError

RECALL_CFG_PATH = Path(os.environ.get("MPMF_RECALL_CONFIG")
                       or str(Path(os.environ.get("MPMF_HOME") or "~/.mpmf").expanduser()
                              / "recall.config.json"))


class RecallShim:
    """Same shape the plugin's _wrapper_recall returns."""

    def __init__(self) -> None:
        self.top_nodes: List[Any] = []
        self.raw_texts: Dict[Any, str] = {}
        self.summaries: Dict[Any, str] = {}
        self.scores: Dict[Any, Any] = {}      # r1: recall score per node (tagged format only)


def render_cfg(max_chars: int = 3500) -> Dict[str, Any]:
    """Round 56 render mode, read on EVERY call. Unknown/unreadable -> summary."""
    # r59d: a missing/unreadable/invalid config RAISES. It does not silently
    # degrade to "summary" — that fallback is exactly what hid the r56 render
    # switch for a day.
    try:
        with open(RECALL_CFG_PATH) as f:
            cfg = json.load(f) or {}
    except FileNotFoundError:
        # Portable install: no config file yet -> package defaults from paths.
        from . import paths as _paths
        cfg = dict(_paths.DEFAULTS)
    except Exception as e:
        raise ConfigError(f"recall config unreadable: {RECALL_CFG_PATH}: {e}") from e
    if not isinstance(cfg, dict):
        raise ConfigError(f"recall config is not an object: {RECALL_CFG_PATH}")
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
    return {"mode": mode, "item_cap": item_cap, "block_cap": block_cap}


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


def injection_opts(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """r1 flags from a config dict (default: the recall config file, read on every call).

    Never raises on a missing/unreadable file (history never did); an INVALID render_format
    raises ConfigError, like memory_render does.
    """
    if cfg is None:
        from . import paths as _paths
        cfg = _paths.load_config(RECALL_CFG_PATH)
    fmt = str(cfg.get("render_format", "legacy") or "legacy").strip().lower()
    if fmt not in _FORMATS:
        raise ConfigError(f"render_format must be one of {_FORMATS}, got {fmt!r}")
    h = _opt_int(cfg.get("history_max_chars"))
    from . import paths as _paths
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
            "lessons_max_chars": _opt_int(cfg.get("lessons_max_chars"))}


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


def render_memory_sections(query: str, result: Any, memory_manager: Any, top_k: int,
                           elapsed: float = 0.0, max_chars: int = 3500,
                           opts: Optional[Dict[str, Any]] = None):
    """Return ({"facts": str, "history": str}, items): the memory part, one string per section.

    Legacy (all r1 flags at their defaults): a temporal question that fires history gets
    {"facts": "", "history": <block>} (history REPLACES the facts block), exactly as before.
    history_additive: facts are rendered exactly as with history off, history is added.
    """
    secs = {"facts": "", "history": ""}
    items: List[Dict[str, Any]] = []
    if not result or not getattr(result, "top_nodes", None):
        return secs, items
    o = opts
    # history_mode (default ON since g2): temporal questions get date-ordered facts.
    # The r1 options are only read once history has actually fired / for the facts block, so a
    # config that history tolerated before is tolerated now.
    if memory_manager is not None:
        try:
            from . import history as _hist
            if _hist.enabled() and _hist.is_temporal(query):
                _idx = _hist.index().ensure(memory_manager)
                if o is None:
                    o = injection_opts()
                if o["render_format"] == "tagged":
                    from . import tagged as _tg
                    _body = _tg.history(query, result, memory_manager, _idx, top_k,
                                        o["history_max_chars"])
                elif o["history_additive"]:
                    _body = _hist.render_history_block(
                        query, result, memory_manager, _idx, top_k=top_k,
                        max_chars=o["history_max_chars"], elapsed=elapsed,
                        title=f"# MPMF+ history for: {query[:120]!r}")
                else:
                    _body = _hist.render_history_block(
                        query, result, memory_manager, _idx, top_k=top_k,
                        max_chars=o["history_max_chars"], elapsed=elapsed)
                if _body:
                    secs["history"] = _body
                    if not o["history_additive"]:
                        return secs, items
        except Exception:
            pass
    if o is None:
        o = injection_opts()
    body, items = _facts_section(query, result, memory_manager, top_k, elapsed, max_chars,
                                 o["render_format"] == "tagged", o["facts_max_chars"])
    secs["facts"] = body
    return secs, items


def render_memory_block(query: str, result: Any, memory_manager: Any, top_k: int,
                        elapsed: float = 0.0, max_chars: int = 3500,
                        opts: Optional[Dict[str, Any]] = None):
    """Return (body, items). Faithful port of prefetch() lines 827-880 (+ r1 flags)."""
    secs, items = render_memory_sections(query, result, memory_manager, top_k,
                                         elapsed=elapsed, max_chars=max_chars, opts=opts)
    return "\n\n".join(p for p in (secs["facts"], secs["history"]) if p), items
