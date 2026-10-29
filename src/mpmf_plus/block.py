"""Block builder — ported from the plugin's render path (r59d).

Source: /root/.hermes/plugins/mpmf_plus_user/__init__.py
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


def render_cfg(max_chars: int = 3500) -> Dict[str, Any]:
    """Round 56 render mode, read on EVERY call. Unknown/unreadable -> summary."""
    # r59d: a missing/unreadable/invalid config RAISES. It does not silently
    # degrade to "summary" — that fallback is exactly what hid the r56 render
    # switch for a day.
    try:
        with open(RECALL_CFG_PATH) as f:
            cfg = json.load(f) or {}
    except FileNotFoundError:
        raise ConfigError(f"recall config not found: {RECALL_CFG_PATH}") from None
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
        return shim
    except Exception:
        return None


def render_memory_block(query: str, result: Any, memory_manager: Any, top_k: int,
                        elapsed: float = 0.0, max_chars: int = 3500):
    """Return (body, items). Faithful port of prefetch() lines 827-880."""
    items: List[Dict[str, Any]] = []
    if not result or not getattr(result, "top_nodes", None):
        return "", items
    lines: List[str] = []
    lines.append(f"# MPMF+ recall for: {query[:120]!r}")
    lines.append(f"({len(result.top_nodes)} top hits, {elapsed:.2f}s)")
    lines.append("")
    raw = getattr(result, "raw_texts", {}) or {}
    summaries = getattr(result, "summaries", {}) or {}
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
        lines.append(f"## [{nid}] {name or '(node)'}")
        lines.append(entry)
        lines.append("")
        n_added += 1
        items.append({"node_id": nid, "name": name, "text": entry})
    if n_added == 0:
        return "", items
    body = "\n".join(lines).strip()
    if len(body) > _rc["block_cap"]:
        body = body[: _rc["block_cap"]] + "\n\n[... MPMF+ recall truncated at char cap ...]"
    return body, items
