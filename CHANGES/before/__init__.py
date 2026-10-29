"""MPMF+ memory provider for Hermes.

Loads the MPMF+ graph at <MPMF_HOME>/store.mpmf via the wrapper at
<MPMF_HOME>/live_wrapper.py, and exposes it as a drop-in MemoryProvider
plugin. On every turn:

  prefetch(query)       — recall top-K relevant nodes for the user message,
                          plus thinking-trace context and cognition lessons
  sync_turn(user, asst) — write new facts back to the graph
  system_prompt_block() — short status block so the agent knows it's live

The closed-loop injection (v1.0 spec Part B, now live):
- <thinking-context> block: raw reasoning traces from live-thinking.mpmf,
  retrieved via recall_thinking(). Shows past-session context → conclusion
  for judgments with similar problem structure.
- <prior-reasoning> block: distilled cognition lessons (provisional +
  verified) from the main store.mpmf store, retrieved via recall_cognition().
  Compact, actionable rules of thumb extracted from past session thinking.
Both are fail-soft: if either store is unavailable or returns no hits,
the turn proceeds with whatever else is available.

Designed to be a TRUE active-memory replacement for the static
MEMORY.md + USER.md injection. The agent no longer needs the static
2,200-char MEMORY.md / 1,375-char USER.md blocks in the system prompt —
relevant context is recalled dynamically per turn.

Configuration (config.yaml):
  memory:
    provider: mpmf_plus
  plugins:
    mpmf_plus:
      store_path: <MPMF_HOME>/store.mpmf
      wrapper_module: <MPMF_HOME>/live_wrapper.py
      top_k: 8                # nodes returned per prefetch
      max_chars: 3500         # hard cap on injected context size
      min_score: 0.0          # filter low-relevance hits
      auto_write: true        # write back user/assistant turns
      namespaces:             # optional namespace filter (empty = all)
        - memory
        - user
        - rumination
        - fact
      pre_tool_causal: true    # query causal plane + rumination nodes before tool
                               # calls. Blocks the tool with relevant context so
                               # the agent re-evaluates before executing.

"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)

# Round 56: where the render/arm switches live. A module-level constant (not a
# literal inside the function) so tests can point it at a temp file without
# touching the live config.
RECALL_CFG_PATH = "<MPMF_HOME>/recall.config.json"

# Mirrors memory_manager._EXTERNAL_PREFETCH_TIMEOUT_S — the core join budget.
# The agent runs prefetch() in a worker thread and joins it for this many
# seconds; if we exceed it the whole prefetch result is DROPPED for that turn.
# We can't observe the core-side timeout from here, but we CAN log our own
# elapsed time when it crosses this line, so a dropped-injection turn is
# diagnosable from the logs alone.
_PREFETCH_JOIN_BUDGET_S = 8.0

# Query truncation for the embed paths (item 1, 2026-10-10). Mirrors
# live_memory._truncate_for_embed so the cognition/thinking paths (which
# have their OWN embedders, not the wrapper pipeline) also receive a capped
# query. mxbai-embed-large's window is 512 tokens; >1200-char inputs are
# head+tail capped to avoid the Ollama "input length exceeds the context
# length" 500. Text <= 1200 chars passes through UNCHANGED.
_QUERY_TRUNC_THRESHOLD_DEFAULT = 1200


def _embed_query_cap() -> int:
    """Embed head+tail cap (chars) from <MPMF_HOME>/recall.config.json.

    Same key + same formula as live_memory._truncate_for_embed, so a query
    capped here is byte-identical to one capped there (shared vector cache still
    hits). Default 1200 when the key is absent. Read per call — no restart needed.
    """
    try:
        with open("<MPMF_HOME>/recall.config.json") as f:
            cap = int(json.load(f).get("embed_query_cap_chars"))
        return cap if cap > 0 else _QUERY_TRUNC_THRESHOLD_DEFAULT
    except Exception:
        return _QUERY_TRUNC_THRESHOLD_DEFAULT


def _recall_cfg() -> Dict[str, Any]:
    """The whole <MPMF_HOME>/recall.config.json as a dict ({} on failure).

    Round 12 item 3: shared reader for the recall-side keys (embed cap, prefetch
    budget, thinking floor / max-items). Unknown keys are simply absent.
    """
    try:
        with open("<MPMF_HOME>/recall.config.json") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _shared_query_vec(query: str):
    """Embed the query once for the whole prefetch. (vec, seconds) or (None, 0).

    Round 45: extracted from prefetch() so the no-vector failure path is
    testable. Returns None on ANY embed failure — callers then treat cosine as
    failed (Jaccard fallback) rather than as "no match".
    """
    import time as _t
    try:
        mod = _get_wrapper_module()
        if mod is not None and hasattr(mod, "embed_query"):
            _s = _t.time()
            vec = mod.embed_query(_truncate_query(query))
            return (vec, _t.time() - _s)
    except Exception as e:
        try:
            logger.debug("MPMF+ shared query embed failed (per-path embed runs): %s", e)
        except Exception:
            pass
    return (None, 0.0)


def _cognition_empty_action(arm, block, meta):
    """Round 45 policy: what does an EMPTY cognition result mean?

    Returns (action, reason):
      ("ok", None)            -> there is a block; nothing to decide.
      ("empty", reason)       -> inject nothing. reason is "below_min_cos" /
                                 "filtered_out" for cosine, "no_match" for the
                                 Jaccard arm. This is the correct answer for a
                                 turn that simply has no relevant lesson.
      ("fallback", reason)    -> the cosine arm could not run (no vector,
                                 missing/stale index, no meta, exception), so
                                 the Jaccard fallback is legitimate.
    """
    if (block or "").strip():
        return ("ok", None)
    if arm != "cosine":
        return ("empty", "no_match")
    if meta is None:
        return ("fallback", "cosine_error:no_meta")
    cs = meta.get("cosine_status")
    if cs in ("no_query_vec", "sit_missing", "sit_stale", "no_index", "error"):
        return ("fallback", "cosine_error:" + str(cs))
    return ("empty", cs or "below_min_cos")


def _truncate_query(text: str) -> str:
    cap = _embed_query_cap()
    if not text or len(text) <= cap:
        return text
    half = cap // 2
    return text[:half] + " " + text[-half:]


# Synthetic system messages (item 1, 2026-10-12). The core injects these as
# USER turns for role alternation (see process_registry.format_process_notification
# and turn_context), so prefetch_all() runs the full three-path prefetch on them.
# They are not user asks: recall can never help, and on long ones (a completion
# notification carrying command output) the prefetch costs 5-7 s for nothing.
# Match is anchored at the START of the (lstripped) message, so a genuine user
# message that merely CONTAINS one of these phrases mid-text is unaffected.
_SYNTHETIC_PREFIXES = (
    "[IMPORTANT: Background process",
    "[IMPORTANT: You are running as a scheduled cron job",
    "[CONTEXT COMPACTION",
    "[Your active task list",
    "[ASYNC DELEGATION",
)


def _is_synthetic_message(text: str) -> bool:
    """True if `text` is a system-injected synthetic turn, not a user ask."""
    if not text:
        return False
    s = text.lstrip()
    return any(s.startswith(p) for p in _SYNTHETIC_PREFIXES)

# -----------------------------------------------------------------------
# Module-level state — load wrapper once, share across agent instances.
# -----------------------------------------------------------------------

_WRAPPER_LOCK = threading.Lock()
_WRAPPER: Any = None
_WRAPPER_PATH: Optional[str] = None
_WRAPPER_LOAD_ERROR: Optional[str] = None
_CONFIG_CACHE: Optional[Dict[str, Any]] = None

# Thinking-trace store (live-thinking.mpmf) — loaded lazily, cached.
_THINKING_LOCK = threading.Lock()
_THINKING: Any = None
_THINKING_LOAD_ERROR: Optional[str] = None


def _load_plugin_config() -> Dict[str, Any]:
    """Read plugins.mpmf_plus_user (preferred) or legacy plugins.mpmf_plus from config.yaml."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        # Prefer the update-safe user-plugin block; fall back to legacy name.
        _CONFIG_CACHE = (
            cfg_get(cfg, "plugins", "mpmf_plus_user")
            or cfg_get(cfg, "plugins", "mpmf_plus")
            or {}
        )
    except Exception:
        _CONFIG_CACHE = {}
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = {}
    return _CONFIG_CACHE


def _load_wrapper(wrapper_path: str) -> Any:
    """Import the MPMF+ wrapper as a module. Cached after first success.

    CRITICAL: ``importlib.util.module_from_spec`` does NOT register the
    module in ``sys.modules``. Callers that later do
    ``sys.modules.get("live_memory_user")`` (``_tool_store``,
    ``sync_turn``) will get None and report ``store() unavailable`` even
    though the module object is alive on ``_WRAPPER``. Always register
    under the canonical name so module-level ``store()`` stays reachable.
    """
    global _WRAPPER, _WRAPPER_PATH, _WRAPPER_LOAD_ERROR
    with _WRAPPER_LOCK:
        if _WRAPPER is not None and _WRAPPER_PATH == wrapper_path:
            # Re-bind to sys.modules in case a reload cleared it.
            if "live_memory_user" not in sys.modules:
                sys.modules["live_memory_user"] = _WRAPPER
            return _WRAPPER
        try:
            p = Path(wrapper_path)
            if not p.is_file():
                raise FileNotFoundError(f"wrapper not found: {wrapper_path}")
            spec = importlib.util.spec_from_file_location(
                "live_memory_user", str(p)
            )
            if not spec or not spec.loader:
                raise RuntimeError("could not build import spec")
            mod = importlib.util.module_from_spec(spec)
            # Register BEFORE exec_module so intra-module relative imports
            # (and any code that inspects sys.modules during load) work.
            sys.modules["live_memory_user"] = mod
            # Add the wrapper's parent dir to sys.path so its internal
            # imports (e.g. `from mpmf.memory import MemoryManager`) resolve.
            parent = str(p.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            try:
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
            except Exception:
                # Don't leave a half-loaded module registered.
                sys.modules.pop("live_memory_user", None)
                raise
            _WRAPPER = mod
            _WRAPPER_PATH = wrapper_path
            _WRAPPER_LOAD_ERROR = None
            return mod
        except Exception as e:
            _WRAPPER_LOAD_ERROR = f"{type(e).__name__}: {e}"
            logger.warning("MPMF+ wrapper load failed: %s", _WRAPPER_LOAD_ERROR)
            return None


def _get_wrapper_module() -> Any:
    """Resolve the loaded live_memory wrapper, preferring the cache.

    Prefer ``_WRAPPER`` (set by ``_load_wrapper``) over ``sys.modules`` alone
    so store/sync still work if something clears sys.modules mid-process.
    Reloads the wrapper from ``plugins.mpmf_plus`` config if neither is set.
    """
    mod = _WRAPPER or sys.modules.get("live_memory_user")
    if mod is not None and hasattr(mod, "store"):
        return mod
    cfg = _load_plugin_config()
    wrapper_path = cfg.get("wrapper_module", "<MPMF_HOME>/live_wrapper.py")
    return _load_wrapper(wrapper_path)


# -----------------------------------------------------------------------
# Thinking-trace store loader (live-thinking.mpmf)
# -----------------------------------------------------------------------

def _load_thinking() -> Any:
    """Import the thinking-trace wrapper module (live_thinking.py), cached.

    Returns the module object (exposing recall_thinking / get_stats) or None
    on any failure. Fail-soft: the caller must tolerate None.
    """
    global _THINKING, _THINKING_LOAD_ERROR
    with _THINKING_LOCK:
        if _THINKING is not None:
            return _THINKING
        try:
            p = Path("<MPMF_HOME>/live_thinking.py")
            if not p.is_file():
                raise FileNotFoundError(f"thinking module not found: {p}")
            spec = importlib.util.spec_from_file_location("live_thinking_user", str(p))
            if not spec or not spec.loader:
                raise RuntimeError("could not build thinking import spec")
            mod = importlib.util.module_from_spec(spec)
            sys.modules["live_thinking_user"] = mod
            parent = str(p.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            try:
                spec.loader.exec_module(mod)
            except Exception:
                sys.modules.pop("live_thinking_user", None)
                raise
            _THINKING = mod
            _THINKING_LOAD_ERROR = None
            return mod
        except Exception as e:
            _THINKING_LOAD_ERROR = f"{type(e).__name__}: {e}"
            logger.warning("MPMF+ thinking loader failed: %s", _THINKING_LOAD_ERROR)
            return None


# -----------------------------------------------------------------------
# Provider
# -----------------------------------------------------------------------


class _RecallShim:
    """Adapts the wrapper recall()'s list-of-dicts return into the
    top_nodes / raw_texts / summaries shape the formatter below expects."""

    __slots__ = ("top_nodes", "raw_texts", "summaries")

    def __init__(self) -> None:
        self.top_nodes: List[Any] = []
        self.raw_texts: Dict[Any, str] = {}
        self.summaries: Dict[Any, str] = {}


class MPMFPlusProvider(MemoryProvider):
    """MPMF+ graph as Hermes memory backend."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or _load_plugin_config()
        self._store_path = self._config.get("store_path", "<MPMF_HOME>/store.mpmf")
        self._wrapper_path = self._config.get("wrapper_module", "<MPMF_HOME>/live_wrapper.py")
        self._top_k = int(self._config.get("top_k", 8))
        self._max_chars = int(self._config.get("max_chars", 3500))
        self._min_score = float(self._config.get("min_score", 0.0))
        self._auto_write = bool(self._config.get("auto_write", True))
        self._pre_tool_causal = bool(self._config.get("pre_tool_causal", True))
        # Thinking-trace + cognition injection (wired into prefetch)
        self._thinking_top_k = int(self._config.get("thinking_top_k", 3))
        self._thinking_max_chars = int(self._config.get("thinking_max_chars", 800))
        # Round 12 item 3: optional thinking floor / max-items, read from
        # recall.config.json. ABSENT key -> None -> no filtering = current
        # behaviour (floor off, item cap = thinking_top_k). Live values untouched.
        _rcfg = _recall_cfg()
        _tms = _rcfg.get("thinking_min_score")
        _tmi = _rcfg.get("thinking_max_items")
        self._thinking_min_score = float(_tms) if _tms is not None else None
        self._thinking_max_items = int(_tmi) if _tmi is not None else None
        self._cognition_top_k = int(self._config.get("cognition_top_k", 3))
        self._cognition_max_chars = int(self._config.get("cognition_max_chars", 700))
        self._inject_thinking = bool(self._config.get("inject_thinking", True))
        self._inject_cognition = bool(self._config.get("inject_cognition", True))
        # Cache per-thread for pre-tool causal recall
        self._pre_tool_cache = threading.local()
        self._pre_tool_cache.buffer = []
        # Cached per-process state
        self._memory = None  # MPMFMemory instance
        self._session_id: str = ""
        # Last query result for debugging
        self._last_prefetch: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "mpmf_plus_user"

    # -- lifecycle --------------------------------------------------------

    def is_available(self) -> bool:
        """Cheap check — does the file exist? Don't open it here."""
        try:
            return Path(self._store_path).is_file() and Path(self._wrapper_path).is_file()
        except Exception:
            return False

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Load the MPMF+ graph. Heavy but one-shot."""
        self._session_id = session_id
        mod = _load_wrapper(self._wrapper_path)
        if mod is None:
            logger.error(
                "MPMF+ provider init failed: wrapper unavailable. "
                "Reason: %s",
                _WRAPPER_LOAD_ERROR or "unknown",
            )
            return
        try:
            self._memory = mod.MPMFMemory(
                self._store_path,
                vector_dimensions=getattr(mod, "VECTOR_DIM", 1024),
                embed_model=getattr(mod, "EMBED_MODEL", "mxbai-embed-large"),
            )
            logger.info(
                "MPMF+ provider loaded %d nodes from %s",
                len(self._memory._nodes) if hasattr(self._memory, "_nodes") else "?",
                self._store_path,
            )
        except Exception as e:
            logger.error("MPMF+ provider init failed: %s", e)
            self._memory = None

        # Preload the cross-encoder reranker in a BACKGROUND thread so the
        # FIRST turn after a (re)start doesn't pay the ~6.7 s model-load
        # cost on the critical path. Fail-soft: the recall path loads it
        # lazily anyway if this thread hasn't finished yet.
        try:
            if hasattr(mod, "_get_reranker"):
                def _preload_reranker() -> None:
                    try:
                        mod._get_reranker()
                        logger.info("MPMF+ reranker preloaded (background)")
                    except Exception as e:
                        logger.warning("MPMF+ reranker preload failed: %s", e)
                threading.Thread(
                    target=_preload_reranker, name="mpmf-reranker-preload",
                    daemon=True,
                ).start()
        except Exception as e:
            logger.warning("MPMF+ reranker preload spawn failed: %s", e)

        # Round 8 item 3: ONE full-path dummy recall (embed + retrieval +
        # rerank) in a background thread, so the first REAL turn after a
        # (re)start doesn't pay the whole cold-start cost. Flag
        # warmup_on_start (default true). Fail-soft.
        try:
            _warm = bool(self._config.get("warmup_on_start", True))
        except Exception:
            _warm = True
        if _warm:
            def _warmup() -> None:
                _t = time.time()
                try:
                    self._wrapper_recall("warmup memory recall", 3)
                    logger.info("warmup done %d ms", int((time.time() - _t) * 1000))
                except Exception as e:
                    logger.warning("MPMF+ warmup failed: %s", e)
            threading.Thread(
                target=_warmup, name="mpmf-warmup", daemon=True,
            ).start()

    def shutdown(self) -> None:
        # Nothing to flush — the wrapper writes synchronously on every store()
        self._memory = None

    # -- static system-prompt block --------------------------------------

    def system_prompt_block(self) -> str:
        """One-line status, plus usage instructions for the agent."""
        if self._memory is None:
            return (
                "# MPMF+ Memory\n"
                "INACTIVE — wrapper failed to load. "
                "Falling back to static MEMORY.md/USER.md injection.\n"
                f"Reason: {_WRAPPER_LOAD_ERROR or 'unknown'}"
            )
        # Get real node/edge counts from the underlying MemoryManager
        n_nodes = "?"
        n_edges = "?"
        try:
            inner = getattr(self._memory, "mem", None) or getattr(self._memory, "graph", None)
            if inner is not None and hasattr(inner, "all_node_ids"):
                n_nodes = len(inner.all_node_ids())
            # edges are stored on the wrapper; sum across nodes if accessible
            mm = getattr(self._memory, "mem", None)
            if mm is not None and hasattr(mm, "_edges"):
                n_edges = len(mm._edges)
            elif hasattr(self._memory, "_edges"):
                n_edges = len(self._memory._edges)
        except Exception:
            pass
        return (
            "# MPMF+ Memory (ACTIVE)\n"
            f"Graph loaded: {n_nodes} nodes, {n_edges} edges from {self._store_path}.\n"
            "Relevant context for each user message is recalled dynamically and "
            "injected as <memory-context> blocks before the user message — "
            "treat them as authoritative reference, not new input.\n"
            "Use the `mpmf_recall` tool for ad-hoc deep lookups (entity, plane-filtered, "
            "or wider top_k). Use `mpmf_store` to write durable facts back to the graph."
        )

    def _wrapper_recall(self, query: str, top_k: int,
                        query_vec: Optional[List[float]] = None,
                        budget_start: Optional[float] = None) -> Optional["_RecallShim"]:
        """Namespace-aware recall via the wrapper (1024-d mxbai), in the
        top_nodes/raw_texts/summaries shape. Falls back to the package recall
        ONLY when the wrapper raises; returns None if that also raises.

        The wrapper's module-level recall() is the single source of truth for
        the store's embedding dimension (VECTOR_DIM=1024). The package
        MPMFMemory is built with the store's real dims (see initialize()), so
        the fallback is dimension-safe — but it is all-namespaces / no filter,
        and is therefore only a last resort. An EMPTY wrapper result means
        nothing relevant and is returned as an empty shim (never a fallback).
        """
        query = _truncate_query(query)
        mod = _get_wrapper_module()
        if mod is not None and hasattr(mod, "recall"):
            try:
                _kw = {"query_vec": query_vec} if query_vec is not None else {}
                if budget_start is not None:
                    _kw["budget_start"] = budget_start
                rows = mod.recall(query, top_n=top_k, **_kw) or []
                shim = _RecallShim()
                shim.top_nodes = [r.get("node_id") for r in rows]
                shim.raw_texts = {r.get("node_id"): (r.get("text") or "") for r in rows}
                shim.summaries = {r.get("node_id"): (r.get("summary") or r.get("text") or "")
                                  for r in rows}
                return shim
            except Exception as e:
                self._last_recall_error = type(e).__name__   # Round 21: audit only
                logger.warning(
                    "MPMF+ wrapper recall raised, falling back to package recall: %s", e
                )
        try:
            res = self._memory.recall(query, top_k=top_k, top_n=top_k)
            shim = _RecallShim()
            top = list(getattr(res, "top_nodes", []) or [])
            raw = getattr(res, "raw_texts", {}) or {}
            summ = getattr(res, "summaries", {}) or {}
            shim.top_nodes = top
            shim.raw_texts = {nid: (raw.get(nid) or "") for nid in top}
            shim.summaries = {nid: (summ.get(nid) or "") for nid in top}
            return shim
        except Exception as e:
            self._last_recall_error = type(e).__name__   # Round 21: audit only
            logger.warning("MPMF+ package recall raised (no injection): %s", e)
            return None

    #: Append-only JSONL coverage log, one line per prefetch attempt (real or
    #: synthetic skip). 2026-10-15 (round 4, item 3): deliberately NOT written
    #: through the rotating log handler, so coverage history is never lost.
    _COVERAGE_PATH = "<MPMF_HOME>/health/prefetch_coverage.jsonl"

    def _write_coverage(self, *, query: str = "", hits: int = 0,
                        total_ms: float = 0.0, phases: Optional[Dict] = None,
                        nsfilter: Optional[str] = None,
                        synthetic_skipped: bool = False,
                        reason: str = "") -> None:
        """Append one JSON line to prefetch_coverage.jsonl (fail-soft).

        Fields: ts (UTC ISO), query_len, hits, total_ms, phases{}, nsfilter,
        synthetic_skipped, reason. `synthetic_skipped` is True for turns that
        were skipped rather than delivered (synthetic system message, or an
        empty query / unloaded memory), so a coverage audit can tell real
        deliverable turns apart from skips.
        """
        try:
            from datetime import datetime, timezone
            rec = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "query_len": len(query or ""),
                "hits": int(hits),
                "total_ms": round(float(total_ms), 1),
                "phases": phases or {},
                "nsfilter": nsfilter,
                "synthetic_skipped": bool(synthetic_skipped),
            }
            if reason:
                rec["reason"] = reason
            d = os.path.dirname(self._COVERAGE_PATH)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self._COVERAGE_PATH, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:  # never break a turn over telemetry
            logger.debug("MPMF+ coverage log write failed: %s", e)

    # -- injection audit log (Round 21, item 1: LOGGING ONLY) --------------
    #: Append-only JSONL, ONE line per prefetch attempt (real, empty, or
    #: synthetic skip). Fail-soft: telemetry must never break a turn. This does
    #: NOT change which blocks run/when, nor any block's text, order or budget.
    _INJECTION_LOG = "<MPMF_HOME>/health/injection_log.jsonl"

    def _write_injection_log(self, rec: Dict[str, Any]) -> None:
        """Append one JSON line to injection_log.jsonl. Never raises."""
        try:
            d = os.path.dirname(self._INJECTION_LOG)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self._INJECTION_LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:
            try:
                logger.debug("MPMF+ injection log write failed: %s", e)
            except Exception:
                pass

    @staticmethod
    def _session_start_epoch(sid: Optional[str]) -> Optional[float]:
        """Best-effort session start (epoch) parsed from a 'YYYYMMDD_HHMMSS'
        session-id prefix. None when unparsable."""
        try:
            import datetime as _dt
            return _dt.datetime.strptime((sid or "")[:15], "%Y%m%d_%H%M%S").replace(
                tzinfo=_dt.timezone.utc).timestamp()
        except Exception:
            return None

    def _node_meta(self, nid) -> Dict[str, Any]:
        """Best-effort {name, ns, status, source_session} for a node id via the
        MemoryManager. Never raises; unknown -> None."""
        out: Dict[str, Any] = {"name": None, "ns": None, "status": None,
                               "source_session": None}
        try:
            _mm = getattr(self._memory, "mem", None)
            if _mm is None:
                return out
            out["name"] = _mm.get_name(nid)
            wiki = _mm.get_wiki(nid) or {}
            out["ns"] = wiki.get("_ns") or wiki.get("namespace")
            out["status"] = wiki.get("_status")
            out["source_session"] = wiki.get("_source_session")
        except Exception:
            pass
        return out

    @staticmethod
    def _same_session(src, name, sid, start_epoch) -> Optional[bool]:
        """True if the node belongs to the current session, else False; None if
        the session is unknown. A live_<epoch> node counts as same-session when
        its epoch is >= this session's start time."""
        if not sid:
            return None
        if src:
            return bool(src == sid)
        nm = name or ""
        if nm.startswith("live_"):
            try:
                ep = float(nm[5:].split("_")[0])
            except Exception:
                return None
            if start_epoch is None:
                return None
            return bool(ep >= start_epoch)
        return False

    def _mem_item(self, nid, name, text, sid, start_epoch) -> Dict[str, Any]:
        """One audited memory-block item (logging only)."""
        meta = self._node_meta(nid)
        return {
            "name": name or meta.get("name"),
            "ns": meta.get("ns"),
            "status": meta.get("status"),
            "score": None,  # memory recall exposes no per-item score
            "chars": len(text or ""),
            "source_session": meta.get("source_session"),
            "same_session": self._same_session(meta.get("source_session"),
                                               name or meta.get("name"), sid, start_epoch),
        }

    def _read_last_cognition_items(self):
        """Best-effort (n_returned, items) from the LAST cognition_inject.log
        line written by recall_cognition. Read-only tail read; logging only."""
        try:
            p = "<MPMF_HOME>/health/cognition_inject.log"
            with open(p, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 4096))
                data = f.read().decode("utf-8", "replace").strip().splitlines()
            if not data:
                return None, []
            last = json.loads(data[-1])
            self._last_cognition_retrieval = last.get("retrieval")  # round 35: 'jaccard'|'cosine'
            names = last.get("names") or []
            stats = last.get("statuses") or []
            scores = last.get("scores") or []
            items = []
            for i, nm in enumerate(names):
                items.append({
                    "name": nm, "ns": "cognition",
                    "status": (stats[i] if i < len(stats) else None),
                    "score": (scores[i] if i < len(scores) else None),
                    "chars": None, "source_session": None, "same_session": None,
                })
            return last.get("n_returned"), items
        except Exception:
            return None, []

    def _skip_rec(self, query, sid, skip_reason) -> Dict[str, Any]:
        """Minimal injection-log record for a turn where no block ran."""
        try:
            import datetime as _dt
            q = query or ""
            blocks = {b: {"enabled": (b != "memory"), "ran": False,
                          "skipped": skip_reason, "n_candidates": None,
                          "items": [], "block_chars": 0, "ms": 0.0}
                      for b in ("memory", "thinking", "cognition")}
            return {"ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                    "session_id": sid or None, "q_len": len(q),
                    "q_sha1": hashlib.sha1(q.encode("utf-8", "replace")).hexdigest()[:12],
                    "q_head": q[:80], "budget_ms": None,
                    "elapsed_ms_before_optional": None, "over_budget": None,
                    "blocks": blocks, "total_block_chars": 0}
        except Exception:
            return {}

    # -- per-turn prefetch (THE HOOK) -------------------------------------

    def _render_cfg(self):
        """Round 56: memory-block render mode, read on EVERY prefetch call.

        memory_render: "summary" (default) | "full".
          summary -> the node's short summary, 600-char item cap, block cap
                     self._max_chars (3,500) — byte-identical to pre-r56 output.
          full    -> the node's raw text, 1,200-char item cap, 3,000-char block.
        Caps are overridable with memory_item_chars / memory_block_chars.
        Any unknown value or unreadable config falls back to "summary".
        """
        cfg = {}
        try:
            with open(RECALL_CFG_PATH) as f:
                cfg = json.load(f) or {}
        except Exception:
            cfg = {}
        mode = str(cfg.get("memory_render", "summary")).strip().lower()
        if mode not in ("summary", "full"):
            mode = "summary"
        d_item = 1200 if mode == "full" else 600
        d_block = 3000 if mode == "full" else self._max_chars
        try:
            item_cap = int(cfg.get("memory_item_chars", d_item))
        except Exception:
            item_cap = d_item
        try:
            block_cap = int(cfg.get("memory_block_chars", d_block))
        except Exception:
            block_cap = d_block
        return {"mode": mode, "item_cap": item_cap, "block_cap": block_cap}

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall top-K relevant nodes for the upcoming turn.

        Returns formatted markdown wrapped in fenced block. Capped at
        self._max_chars. Returns "" on any failure (fail-soft).
        """
        if self._memory is None or not query or not query.strip():
            logger.warning(
                "MPMF+ prefetch skipped (no injection): memory_loaded=%s "
                "query_empty=%s",
                self._memory is not None,
                not (query and query.strip()),
            )
            self._write_coverage(query=query or "", synthetic_skipped=True,
                                 reason="no_memory_or_empty_query")
            self._write_injection_log(
                self._skip_rec(query or "", session_id or self._session_id, "disabled"))
            return ""
        # Item 1 (2026-10-12): skip synthetic system turns outright (see
        # _is_synthetic_message). One log line records the skip and the prefix.
        if _is_synthetic_message(query):
            logger.info(
                "MPMF+ prefetch skipped (synthetic system message, prefix=%r)",
                query[:48],
            )
            self._write_coverage(query=query, synthetic_skipped=True,
                                 reason="synthetic_system_message")
            self._write_injection_log(
                self._skip_rec(query, session_id or self._session_id, "synthetic"))
            return ""
        t0 = time.time()
        t0_pc = time.perf_counter()   # round 8 item 1: the prefetch deadline anchor
        # Round 21 (item 1): injection-audit context — logging only, no effect
        # on which blocks run, their order, their text, or the budget logic.
        _sid = session_id or self._session_id or None
        _start_epoch = self._session_start_epoch(self._session_id or session_id)
        # Item 2 (2026-10-12): embed the query ONCE and share the vector with
        # both memory recall and thinking recall. Both use the same mxbai model
        # and the same head+tail truncation, so one Ollama round-trip replaces
        # the two the prefetch used to make. On any failure the vector stays
        # None and each path embeds itself (unchanged behaviour).
        shared_qvec = None
        _t_shared_embed = 0.0
        shared_qvec, _t_shared_embed = _shared_query_vec(query)
        result = None
        # Preferred: the wrapper's namespace-aware recall() (default namespaces;
        # also writes health/recall_trace.log). The wrapper returns a list of
        # dicts, so shim it into the top_nodes/raw_texts/summaries shape used
        # by the formatter below. On ANY failure, warn and fall back to the
        # raw package recall (all namespaces, no namespace filter).
        mod = _get_wrapper_module()
        # Namespace-aware wrapper recall (1024-d). Item 3 (2026-10-10): an
        # EMPTY wrapper result means "nothing relevant" — do NOT fall back to
        # the all-namespaces package recall for it. Only a wrapper EXCEPTION
        # falls back; if that also raises, inject nothing.
        result = self._wrapper_recall(query, self._top_k, query_vec=shared_qvec,
                                      budget_start=t0_pc)
        if result is None:
            return ""
        elapsed = time.time() - t0
        _lt = dict(getattr(mod, "_LAST_TIMINGS", {}) or {})

        mem_items: List[Dict[str, Any]] = []   # Round 21: audit items (logging only)
        if not result or not getattr(result, "top_nodes", None):
            self._last_prefetch = {"query": query, "hits": 0, "elapsed": elapsed, "memory_hits": 0}
            body = ""
            n_added = 0
        else:
            # Build a compact context block
            lines: List[str] = []
            lines.append(f"# MPMF+ recall for: {query[:120]!r}")
            lines.append(f"({len(result.top_nodes)} top hits, {elapsed:.2f}s)")
            lines.append("")

            # raw_texts/summaries are dicts[node_id -> text]
            raw = getattr(result, "raw_texts", {}) or {}
            summaries = getattr(result, "summaries", {}) or {}
            _rc = self._render_cfg()          # Round 56: render mode, per call
            _full = _rc["mode"] == "full"

            n_added = 0
            for nid in result.top_nodes:
                if n_added >= self._top_k:
                    break
                summary = summaries.get(nid) or raw.get(nid) or ""
                if _full:
                    # Full-text mode: prefer the node's raw text.
                    summary = raw.get(nid) or summary
                if not summary or len(summary.strip()) < 5:
                    continue
                # Node name via the MemoryManager (.mem) — item 3, 2026-10-10.
                # self._memory (MPMFMemory) has NO get_node/get_text/... : those
                # live on self._memory.mem. The old missing-method call always
                # fell through to "" — which is exactly why every hit rendered
                # as "(node)" instead of its real name.
                name = ""
                try:
                    _mm = getattr(self._memory, "mem", None)
                    if _mm is not None:
                        name = _mm.get_name(nid) or ""
                except Exception:
                    name = ""

                # Trim overly long entries
                entry = summary.strip()
                if len(entry) > _rc["item_cap"]:
                    entry = entry[: _rc["item_cap"]] + "…"
                lines.append(f"## [{nid}] {name or '(node)'}")
                lines.append(entry)
                lines.append("")
                n_added += 1
                try:
                    mem_items.append(self._mem_item(nid, name, entry, _sid, _start_epoch))
                except Exception:
                    pass

            if n_added == 0:
                body = ""
            else:
                body = "\n".join(lines).strip()
                if len(body) > _rc["block_cap"]:
                    body = body[: _rc["block_cap"]] + "\n\n[... MPMF+ recall truncated at char cap ...]"

        # ── Append thinking-trace + cognition context (the closed loop) ──
        extra = []
        _t_think = 0.0
        _t_cog = 0.0
        _block_before_optional = len(body)   # Round 21: memory-only chars (logging only)
        _tb = ""
        _cb = ""
        _think_skipped: Optional[str] = None   # Round 21: audit only
        _cog_skipped: Optional[str] = None
        # Round 7 item 2c: if the memory phase already blew the prefetch budget,
        # skip the optional cognition/thinking phases (degrade, don't drop).
        _elapsed_ms = (time.time() - t0) * 1000.0
        try:
            with open("<MPMF_HOME>/recall.config.json") as _pf:
                _budget_ms = float(json.load(_pf).get("prefetch_budget_ms", 6500) or 6500)
        except Exception:
            _budget_ms = 6500.0
        _over_budget = _elapsed_ms >= _budget_ms
        if self._inject_thinking and not _over_budget:
            self._last_thinking_meta = {"n_candidates": None, "items": [], "block_chars": 0}
            _s = time.time()
            tb = self._recall_thinking_block(query, query_vec=shared_qvec)
            _t_think = time.time() - _s
            _tb = tb or ""
            if tb:
                extra.append(tb)
        elif not self._inject_thinking:
            _think_skipped = "disabled"
        else:
            _think_skipped = "over_budget"
        _cog_meta = {"effective": None, "champion": None, "pinned": None,
                     "arm": None, "items": [], "shadow_arm": None,
                     "shadow_items": [], "shadow_skipped": None, "fallback": None,
                     "pool_size": None, "sit_missing": None, "sit_stale": None,
                     "retrieval": None, "n_candidates": None, "code_rev": None,
                     "shadow_ms": None, "empty_reason": None,
                     "cosine_status": None, "shadow_empty_reason": None}
        self._last_cognition_meta = _cog_meta
        if self._inject_cognition and not _over_budget:
            _mod = _get_wrapper_module()
            _st = {}
            try:
                _st = _mod.cognition_state() if (_mod and hasattr(_mod, "cognition_state")) else {}
            except Exception:
                _st = {}
            _cog_meta["effective"] = _st.get("effective")
            _cog_meta["champion"] = _st.get("champion")
            _cog_meta["pinned"] = _st.get("pinned")
            _cog_meta["code_rev"] = getattr(_mod, "CODE_REV", None)
            _chosen = _st.get("effective") or "jaccard"
            _COG_ARM_BUDGET_MS = 250.0
            _s = time.time()
            _cb, _cmeta = "", None
            # Round 45 policy: an empty COSINE result is only a failure when the
            # arm could not run (no query vector, missing/stale index, no meta).
            # A healthy cosine that simply found nothing above min_cos is the
            # correct answer for that turn — inject nothing, do NOT re-run
            # Jaccard, and say why in empty_reason.
            _pre_fail = ("cosine_error:no_query_vec"
                         if (_chosen == "cosine" and not shared_qvec) else None)
            try:
                if _pre_fail:
                    # recall_cognition() would silently serve the Jaccard arm
                    # here, so make that fallback explicit rather than invisible.
                    _cog_meta["cosine_status"] = "no_query_vec"
                    _cog_meta["fallback"] = _pre_fail
                    _cb, _cmeta = self._recall_cognition_block(
                        query, query_vec=None, arm="jaccard", want_meta=True)
                    _cog_meta["arm"] = "jaccard"
                else:
                    _cb, _cmeta = self._recall_cognition_block(
                        query, query_vec=shared_qvec, arm=_chosen, want_meta=True)
                    _cog_meta["arm"] = _chosen
                    if _chosen == "cosine":
                        _cog_meta["cosine_status"] = (_cmeta or {}).get("cosine_status")
                    _cog_meta["best_score"] = (_cmeta or {}).get("best_score")
                    _act, _why = _cognition_empty_action(_chosen, _cb, _cmeta)
                    if _act == "empty":
                        _cog_meta["empty_reason"] = _why
                    elif _act == "fallback":
                        _cog_meta["fallback"] = _why
                        _cb, _cmeta = self._recall_cognition_block(
                            query, query_vec=shared_qvec, arm="jaccard", want_meta=True)
                        _cog_meta["arm"] = "jaccard"
            except Exception as _e:
                _cog_meta["fallback"] = "cosine_error:" + type(_e).__name__
                try:
                    _cb, _cmeta = self._recall_cognition_block(
                        query, query_vec=shared_qvec, arm="jaccard", want_meta=True)
                    _cog_meta["arm"] = "jaccard"
                except Exception:
                    _cb, _cmeta = "", None
            _t_cog = time.time() - _s
            _cb = _cb or ""
            _cog_meta["items"] = (_cmeta or {}).get("items") or []
            _cog_meta["pool_size"] = (_cmeta or {}).get("pool_size")
            _cog_meta["sit_missing"] = (_cmeta or {}).get("sit_missing")
            _cog_meta["sit_stale"] = (_cmeta or {}).get("sit_stale")
            _cog_meta["retrieval"] = (_cmeta or {}).get("retrieval")
            # per-arm time budget: if cosine blew the cap, use jaccard this turn
            if _cog_meta["arm"] == "cosine" and _t_cog * 1000.0 > _COG_ARM_BUDGET_MS:
                _cog_meta["fallback"] = "time:%.0fms" % (_t_cog * 1000.0)
                _cb, _cmeta = self._recall_cognition_block(
                    query, query_vec=shared_qvec, arm="jaccard", want_meta=True)
                _cb = _cb or ""
                _cog_meta["arm"] = "jaccard"
                _cog_meta["items"] = (_cmeta or {}).get("items") or []
                _cog_meta["pool_size"] = (_cmeta or {}).get("pool_size")
                _cog_meta["sit_missing"] = (_cmeta or {}).get("sit_missing")
                _cog_meta["sit_stale"] = (_cmeta or {}).get("sit_stale")
                _cog_meta["retrieval"] = (_cmeta or {}).get("retrieval")
            if _cb:
                extra.append(_cb)
            # round 43: vet observability for the injected arm (last call wins)
            for _vk in ("vet_mode", "vet_state", "vet_dropped_reject",
                        "vet_dropped_unvetted", "vet_entries"):
                _cog_meta[_vk] = (_cmeta or {}).get(_vk)
            # shadow: the arm NOT chosen. Never injects, never bumps, never
            # logs. Skipped entirely when we've crossed the join budget.
            _shadow = "jaccard" if _cog_meta["arm"] == "cosine" else "cosine"
            _cog_meta["shadow_arm"] = _shadow
            if (time.time() - t0) * 1000.0 < _budget_ms:
                try:
                    _ts_sh = time.time()
                    _sb, _smeta = self._recall_cognition_block(
                        query, query_vec=shared_qvec, arm=_shadow, want_meta=True, shadow=True)
                    _cog_meta["shadow_ms"] = round(1000.0 * (time.time() - _ts_sh), 1)
                    _cog_meta["shadow_items"] = (_smeta or {}).get("items") or []
                    # Round 45: same empty distinction, observation only.
                    _ssc = (_smeta or {}).get("cosine_status")
                    _cog_meta["shadow_cosine_status"] = _ssc
                    if not _cog_meta["shadow_items"]:
                        _cog_meta["shadow_empty_reason"] = (
                            None if _smeta is not None and _ssc is None
                            else ("no_meta" if _smeta is None
                                  else (_ssc or "below_min_cos")))
                    for _sk in ("vet_mode", "vet_state", "vet_dropped_reject",
                                "vet_dropped_unvetted", "vet_entries"):
                        _cog_meta["shadow_" + _sk] = (_smeta or {}).get(_sk)
                    if _cog_meta.get("pool_size") is None:
                        _cog_meta["pool_size"] = (_smeta or {}).get("pool_size")
                        _cog_meta["sit_missing"] = (_smeta or {}).get("sit_missing")
                        _cog_meta["sit_stale"] = (_smeta or {}).get("sit_stale")
                except Exception as _se:
                    _cog_meta["shadow_skipped"] = "error:" + type(_se).__name__
            else:
                _cog_meta["shadow_skipped"] = "over_budget"
        elif not self._inject_cognition:
            _cog_skipped = "disabled"
        else:
            _cog_skipped = "over_budget"
        if extra:
            body = (body + "\n\n" if body else "") + "\n\n".join(extra)

        self._last_prefetch = {
            "query": query,
            "hits": n_added,
            "elapsed": elapsed,
            "bytes": len(body),
            "memory_hits": n_added,
        }
        # Observability (item 5, 2026-10-10): ONE line per prefetch with the
        # total wall time and per-phase breakdown, emitted ALWAYS (not only on
        # slow/empty turns). Per-phase numbers come from the wrapper recall()'s
        # timings (embed / retrieval / rerank) plus our own thinking/cognition
        # block timers. Escalated to WARNING when we cross the core join budget
        # (where the agent DISCARDS the whole block).
        total = time.time() - t0
        def _ms(x):
            try:
                return 1000.0 * float(x)
            except Exception:
                return 0.0
        slow = total >= _PREFETCH_JOIN_BUDGET_S
        logger.log(
            logging.WARNING if slow else logging.INFO,
            "MPMF+ prefetch: total=%.0fms embed=%.0fms retrieval=%.0fms "
            "rerank=%.0fms cognition=%.0fms thinking=%.0fms open=%.0fms "
            "shared_embed=%.0fms nsfilter=%s rerank=%s "
            "hits=%d bytes=%d empty=%s%s query=%r",
            _ms(total), _ms(_lt.get("embed")), _ms(_lt.get("retrieval")),
            _ms(_lt.get("rerank")), _ms(_t_cog), _ms(_t_think), _ms(_lt.get("open")),
            _ms(_t_shared_embed), (_lt.get("nsfilter") or "?"),
            (_lt.get("rerank_status") or "?"),
            n_added, len(body), (not body),
            (" SLOW>=%.0fs" % _PREFETCH_JOIN_BUDGET_S) if slow else "",
            query[:80],
        )
        # Round 4 item 3: one append-only JSONL coverage line per real prefetch.
        self._write_coverage(
            query=query, hits=n_added, total_ms=total * 1000.0,
            phases={"embed": _ms(_lt.get("embed")),
                    "retrieval": _ms(_lt.get("retrieval")),
                    "rerank": _ms(_lt.get("rerank")),
                    "cognition": _ms(_t_cog), "thinking": _ms(_t_think),
                    "open": _ms(_lt.get("open")),
                    "shared_embed": _ms(_t_shared_embed),
                    "rerank_status": (_lt.get("rerank_status") or "?")},
            nsfilter=(_lt.get("nsfilter") or "?"), synthetic_skipped=False,
        )
        # ── Round 21 item 1: ONE injection-audit line per prefetch ──────────
        # LOGGING ONLY, fail-soft (never raises, never influences injection).
        try:
            import datetime as _dtm
            # round 39: prefer the arm's own returned meta; the cognition_inject
            # log tail-read is only a fallback when the arm returned no meta.
            _cog_items = _cog_meta.get("items") or []
            _cog_retrieval = _cog_meta.get("retrieval")
            _cn = len(_cog_items) if _cog_items else None
            if _cb and not _cog_items:
                try:
                    _cn, _cog_items = self._read_last_cognition_items()
                    _cog_retrieval = _cog_retrieval or getattr(
                        self, "_last_cognition_retrieval", None)
                except Exception:
                    _cn, _cog_items = None, []
            elif not _cb and self._inject_cognition and not _over_budget:
                _cn = 0
            _tmeta = getattr(self, "_last_thinking_meta", None) or {}
            _think_items = _tmeta.get("items") or []
            _tn = _tmeta.get("n_candidates")
            # Round 45 item 4: per-stage timings straight from the module's own
            # _LAST_TIMINGS (no extra work, no extra embed). Null when recall
            # did not run at all (skipped / failed), so a row is self-describing.
            _st = dict(_lt or {})
            _stages = None
            if result is not None and _st:
                _stages = {
                    # the pipeline reuses the shared vector, so its own embed
                    # time is 0; the real Ollama cost is the shared embed.
                    "embed_ms": _ms(max(_st.get("embed") or 0.0, _t_shared_embed)),
                    "open_ms": _ms(_st.get("open")),
                    "open_cache_hit": _st.get("open_cache_hit"),
                    "ns_scan_ms": _ms(_st.get("ns_scan")),
                    "retrieval_ms": _ms(_st.get("retrieval")),
                    "rerank_ms": _ms(_st.get("rerank")),
                    "rerank_timed_out": _st.get("rerank_timed_out"),
                    "n_pre": _st.get("n_pre"),
                    "n_post": _st.get("n_post"),
                    "query_len": len(query or ""),
                    "torch_threads": _st.get("torch_threads"),
                    "warm_ms": _st.get("warm_ms"),
                    "warm_count": _st.get("warm_count"),
                    # r46d/r46f: warm query cost and RSS are ALREADY in ms / kB —
                    # they come from live_memory in final units, so no _ms() here
                    # (that wrapper multiplies SECONDS by 1000; applying it to an
                    # ms value inflated every r46d row by 1000x).
                    "warm_query_ms": _st.get("warm_query_ms"),
                    "rss_kb": _st.get("rss_kb"),
                    "prewarm_state": _st.get("prewarm_state"),
                }
            _rec = {
                "ts": _dtm.datetime.now(_dtm.timezone.utc).isoformat(),
                "session_id": _sid,
                "q_len": len(query or ""),
                "q_sha1": hashlib.sha1((query or "").encode("utf-8", "replace")).hexdigest()[:12],
                "q_head": (query or "")[:80],
                "budget_ms": _budget_ms,
                "elapsed_ms_before_optional": round(_elapsed_ms, 1),
                "over_budget": bool(_over_budget),
                "blocks": {
                    "memory": {
                        "enabled": True,
                        "ran": result is not None,
                        "skipped": (None if result is not None else
                                    "error:" + (getattr(self, "_last_recall_error", None) or "RecallError")),
                        "n_candidates": len(getattr(result, "top_nodes", []) or []),
                        "items": mem_items,
                        "block_chars": _block_before_optional,
                        "ms": round(1000.0 * elapsed, 1),
                    },
                    "thinking": {
                        "enabled": bool(self._inject_thinking),
                        "ran": bool(_tb),
                        "skipped": _think_skipped,
                        "n_candidates": _tn,
                        "items": _think_items,
                        "block_chars": len(_tb),
                        "ms": round(1000.0 * _t_think, 1),
                    },
                    "cognition": {
                        "enabled": bool(self._inject_cognition),
                        # Round 46: "ran" means the arm EXECUTED, not that it
                        # produced a block. An empty cosine answer is a correct
                        # result, so block presence is the wrong signal.
                        "ran": (_cog_meta.get("arm") in ("cosine", "jaccard")),
                        "retrieval": _cog_retrieval,
                        "effective": _cog_meta.get("effective"),
                        "champion": _cog_meta.get("champion"),
                        "pinned": _cog_meta.get("pinned"),
                        "arm": _cog_meta.get("arm"),
                        "fallback_reason": _cog_meta.get("fallback"),
                        "skipped": _cog_skipped,
                        "n_candidates": _cn,
                        "items": _cog_items,
                        "shadow_arm": _cog_meta.get("shadow_arm"),
                        "shadow_items": _cog_meta.get("shadow_items") or [],
                        "shadow_skipped": _cog_meta.get("shadow_skipped"),
                        "vet_mode": _cog_meta.get("vet_mode"),
                        "vet_state": _cog_meta.get("vet_state"),
                        "vet_dropped_reject": _cog_meta.get("vet_dropped_reject"),
                        "vet_dropped_unvetted": _cog_meta.get("vet_dropped_unvetted"),
                        "vet_entries": _cog_meta.get("vet_entries"),
                        "shadow_vet_mode": _cog_meta.get("shadow_vet_mode"),
                        "shadow_vet_state": _cog_meta.get("shadow_vet_state"),
                        "shadow_vet_dropped_reject": _cog_meta.get("shadow_vet_dropped_reject"),
                        "shadow_vet_dropped_unvetted": _cog_meta.get("shadow_vet_dropped_unvetted"),
                        "pool_size": _cog_meta.get("pool_size"),
                        "sit_missing": _cog_meta.get("sit_missing"),
                        "sit_stale": _cog_meta.get("sit_stale"),
                        "code_rev": _cog_meta.get("code_rev"),
                        "shadow_ms": _cog_meta.get("shadow_ms"),
                        "cosine_status": _cog_meta.get("cosine_status"),
                        "best_score": _cog_meta.get("best_score"),
                        "empty_reason": _cog_meta.get("empty_reason"),
                        "shadow_cosine_status": _cog_meta.get("shadow_cosine_status"),
                        "shadow_empty_reason": _cog_meta.get("shadow_empty_reason"),
                        "block_chars": len(_cb),
                        "ms": round(1000.0 * _t_cog, 1),
                    },
                },
                "stages": _stages,
                "total_block_chars": len(body),
            }
            self._write_injection_log(_rec)
        except Exception as _e:
            try:
                logger.debug("MPMF+ injection log build failed: %s", _e)
            except Exception:
                pass
        return body

    # -- thinking-trace + cognition recall (closed loop) ------------------

    def _recall_thinking_block(self, query: str,
                               query_vec: Optional[List[float]] = None) -> str:
        """Recall relevant thinking traces from live-thinking.mpmf.

        Returns a labeled markdown block, or "" if nothing / on failure
        (fail-soft — never blocks the turn).
        """
        tm = _load_thinking()
        if tm is None or not hasattr(tm, "recall_thinking"):
            return ""
        query = _truncate_query(query)
        try:
            _kw = {"query_vec": query_vec} if query_vec is not None else {}
            results = tm.recall_thinking(query, top_n=self._thinking_top_k, **_kw)
        except Exception as e:
            logger.debug("MPMF+ thinking recall failed: %s", e)
            return ""
        if not results:
            return ""
        # Round 12 item 3: optional floor + max-items. Both default to None
        # (no filtering) so behaviour is identical unless a key is present.
        if self._thinking_min_score is not None:
            results = [r for r in results
                       if float(r.get("score") or 0.0) >= self._thinking_min_score]
        if self._thinking_max_items is not None:
            results = results[: self._thinking_max_items]
        if not results:
            return ""
        # Round 21 (item 1): audit diagnostics — logging only, no behaviour change.
        try:
            _mstart = self._session_start_epoch(self._session_id)
            self._last_thinking_meta = {
                "n_candidates": len(results),
                "items": [{
                    "name": r.get("name"),
                    "ns": None,
                    "status": None,
                    "score": r.get("score"),
                    "chars": len((r.get("context") or "") + (r.get("conclusion") or "")),
                    "source_session": None,
                    "same_session": self._same_session(None, r.get("name"),
                                                       self._session_id or None, _mstart),
                } for r in results],
                "block_chars": 0,
            }
        except Exception:
            try:
                self._last_thinking_meta = {"n_candidates": len(results),
                                            "items": [], "block_chars": 0}
            except Exception:
                pass

        lines = ["<thinking-context>",
                 "Prior reasoning traces relevant to this task:"]
        cap = self._thinking_max_chars
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
            # Trim to char budget
            if used + len(entry) > cap:
                rem = cap - used
                if rem > 40:
                    lines.append(entry[:rem] + "…")
                break
            lines.append(entry)
            used += len(entry)
            n += 1
        if n == 0:
            return ""
        lines.append("</thinking-context>")
        _blk = "\n".join(lines)
        try:
            self._last_thinking_meta["block_chars"] = len(_blk)
        except Exception:
            pass
        return _blk

    def _recall_cognition_block(self, query: str, query_vec=None, arm=None,
                                want_meta=False, shadow=False):
        """Recall distilled cognition lessons via recall_cognition().

        arm=None -> the module resolves the arm from the state file.
        arm="jaccard"|"cosine" -> force that arm (used for the shadow).
        shadow=True -> never write an injection-log line (the injected arm does).
        Returns the <prior-reasoning> block, or "" on no hits / failure.
        want_meta=True -> returns (block, meta_dict) where meta_dict carries
        {retrieval, items:[{id,name,score,status}]}; on failure meta is None.
        """
        mod = _get_wrapper_module()
        if mod is None or not hasattr(mod, "recall_cognition"):
            return ("", None) if want_meta else ""
        query = _truncate_query(query)
        try:
            kw = {}
            if arm in ("jaccard", "cosine"):
                kw["arm"] = arm
            if want_meta:
                kw["return_meta"] = True
            kw["log_inject"] = not shadow  # round 39: only the SHADOW is silent
            kw["vet"] = True               # round 43: both arms honour the sidecar
            out = mod.recall_cognition(
                query, top_k=self._cognition_top_k, mode="all", bump_counts=False,
                query_vec=query_vec, **kw,
            )
        except Exception as e:
            logger.debug("MPMF+ cognition recall failed (arm=%s): %s", arm, e)
            return ("", None) if want_meta else ""
        meta = None
        if want_meta:
            if isinstance(out, dict):
                block, meta = out.get("block") or "", out
            else:
                block = out or ""
        else:
            block = out or ""
        if not block or not block.strip():
            return ("", meta) if want_meta else ""
        if len(block) > self._cognition_max_chars:
            block = self._truncate_cognition_block(block)
        # Round 50: framing line (the r49b C arm). Applied AFTER the cap so
        # the lesson body stays byte-identical to what the cap yields today.
        block = self._with_lesson_framing(block)
        return (block, meta) if want_meta else block

    # Round 50 (r49b C arm): the one framing line that goes immediately
    # before the numbered lessons. Kept byte-identical to the arm that was
    # judged; do not reword without re-running the gauntlet.
    LESSON_FRAMING = (
        "Past lessons that may or may not apply. Use one only if this "
        "situation clearly matches its trigger; otherwise ignore them all. "
        "Never mention them."
    )

    def _with_lesson_framing(self, block: str) -> str:
        """Insert LESSON_FRAMING once, immediately before the first lesson.

        No-op on an empty block (0 lessons -> the block is omitted upstream)
        and idempotent if the line is already present. Tags, lesson order and
        the truncation cap are untouched: the line goes in after capping, so
        the numbered lesson body is byte-identical to the pre-round-50 output.
        """
        if not block or not block.strip():
            return block
        if self.LESSON_FRAMING in block:
            return block
        lines = block.split("\n")
        idx = None
        for i, ln in enumerate(lines):
            head = ln.strip().split(".", 1)[0]
            if head.isdigit() and "." in ln.strip():
                idx = i
                break
        if idx is None:  # no numbered lessons in the body: leave it alone
            return block
        lines.insert(idx, self.LESSON_FRAMING)
        return "\n".join(lines)

    def _truncate_cognition_block(self, block: str) -> str:
        """Cap a <prior-reasoning> block at _cognition_max_chars, keeping the
        closing tag so the injected fence stays well-formed.

        recall_cognition() returns '<prior-reasoning>\\n...\\n</prior-reasoning>'.
        We trim the inner body to fit the char budget and re-attach OPEN/MARK/
        CLOSE, so the total stays <= _cognition_max_chars whenever the budget
        can hold the tags themselves. Never returns an unterminated fence.
        """
        OPEN = "<prior-reasoning>"
        CLOSE = "</prior-reasoning>"
        MARK = "\n[... prior-reasoning truncated ...]"
        if block.startswith(OPEN) and block.endswith(CLOSE):
            inner = block[len(OPEN):-len(CLOSE)]
            keep = max(0, self._cognition_max_chars - len(OPEN) - len(CLOSE) - len(MARK))
            return f"{OPEN}{inner[:keep]}{MARK}{CLOSE}"
        # Unknown shape: still guarantee closure so the fence can't leak open.
        return block[: self._cognition_max_chars] + MARK + CLOSE

    # -- per-turn write-back ---------------------------------------------

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        """Persist the just-completed turn to the graph.

        Lightweight: only writes a compact summary if both sides are non-trivial.
        Heuristic avoids flooding the graph on every exchange.
        """
        if not self._auto_write or self._memory is None:
            return
        if not user_content or not assistant_content:
            return
        if len(user_content) + len(assistant_content) < 120:
            return  # too short to be worth a node

        # The wrapper's MPMFMemory instance has no `store` method — store()
        # is module-level. Resolve via _get_wrapper_module so store works
        # even when sys.modules lost the dynamic import entry.
        mod = _get_wrapper_module()
        if mod is None or not hasattr(mod, "store"):
            return

        # Build a composite "fact" from the turn — a digest the next turn
        # can recall. Tag with session_id so per-session traces stay grouped.
        digest = (
            f"[session={self._session_id or session_id or '-'} t={int(time.time())}]\n"
            f"USER: {user_content[:600]}\n"
            f"ASSISTANT: {assistant_content[:600]}"
        )
        name = f"turn_{self._session_id or 'adhoc'}_{int(time.time() * 1000)}"
        try:
            # auto_extract=True — LLM produces 5-plane payload (associative,
            # causal, temporal, functional, priority) so the planes stay
            # meaningful even on auto-writes. Per-turn LLM cost, but the
            # extraction is cached in <MPMF_HOME>/cache/extractions.json so
            # repeat facts are free.
            #
            # auto_link=False — edge rebuild is currently O(N) duplicates
            # (rumination 2026-06-09: 72.9% duplicate rate). Only run
            # auto_link when the user explicitly requests it.
            mod.store(
                digest,
                name=name,
                auto_extract=True,
                auto_link=False,
                namespace="transcript",
            )
        except Exception as e:
            logger.debug("MPMF+ auto-write failed (non-fatal): %s", e)

    # -- tool surface (for ad-hoc deep lookups) ---------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "mpmf_recall",
                "description": (
                    "Ad-hoc deep lookup against the MPMF+ memory graph. "
                    "Use this when prefetched context isn't enough — e.g. to "
                    "look up an entity by name, fetch a node by ID, or run "
                    "a wider recall than the auto-injected block."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (free-form text).",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max nodes to return (default 10).",
                        },
                        "node_id": {
                            "type": "integer",
                            "description": "If set, fetch this specific node by ID instead of running a recall.",
                        },
                    },
                },
            },
            {
                "name": "mpmf_store",
                "description": (
                    "Write a durable fact into the MPMF+ graph. "
                    "Use for things the user would expect to be remembered "
                    "across sessions: preferences, decisions, project state, "
                    "key facts discovered during work."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Fact content to store.",
                        },
                        "name": {
                            "type": "string",
                            "description": "Short human-readable name for the fact.",
                        },
                    },
                    "required": ["text"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        if self._memory is None:
            return json.dumps({"error": f"MPMF+ unavailable: {_WRAPPER_LOAD_ERROR or 'unknown'}"})
        try:
            if tool_name == "mpmf_recall":
                return self._tool_recall(args)
            if tool_name == "mpmf_store":
                return self._tool_store(args)
            return json.dumps({"error": f"unknown tool: {tool_name}"})
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    # -- pre-tool causal + meta hook ------------------------------------
    # This is registered separately in register() so it fires before every tool.
    # It queries the MPMF graph for causal-plane and rumination/meta nodes
    # relevant to the tool being called. If context is found, it blocks the
    # tool with the context so the agent re-evaluates the call.

    SIDE_EFFECT_TOOLS = frozenset({
        "write_file", "patch", "terminal", "delegate_task",
        "image_generate", "send_message", "cronjob",
        "skill_manage", "memory", "mpmf_store",
    })

    def on_pre_tool_call(self, *, tool_name: str = "", args: Any = None,
                          task_id: str = "") -> Optional[Dict[str, Any]]:
        """Pre-tool execution hook: query causal + meta nodes for this tool call.

        Returns {"action": "block", "message": str} with relevant context if
        found, or None to let the tool proceed silently.
        """
        if not self._pre_tool_causal or self._memory is None:
            return None

        # For non-side-effect tools, check if there's a hit — but only
        # block for side-effect tools to avoid excess round-trips on
        # read-only calls.
        is_side_effect = tool_name in self.SIDE_EFFECT_TOOLS

        # Build a query from tool name + args
        args_str = ""
        if isinstance(args, dict):
            # Extract key args for the query — skip file content/large bodies
            query_parts = [tool_name]
            for k, v in args.items():
                if isinstance(v, str) and len(v) < 150 and not v.startswith("MEDIA:"):
                    query_parts.append(f"{k}={v}")
                elif k in ("path", "name", "pattern", "command", "url", "goal"):
                    # Key context-bearing args
                    sv = str(v)[:120]
                    query_parts.append(f"{k}={sv}")
            args_str = " ".join(query_parts[:8])
        else:
            args_str = f"{tool_name} {str(args)[:150]}"

        if not args_str.strip():
            return None

        try:
            # Query MPMF for relevant content (namespace-aware wrapper path)
            results = self._wrapper_recall(args_str, 5)
        except Exception as e:
            logger.debug("MPMF+ pre-tool recall failed: %s", e)
            return None

        if not results or not getattr(results, "top_nodes", None):
            return None

        # Separate causal-plane nodes from rumination/meta nodes
        causal_lines = []
        meta_lines = []
        seen = set()

        for nid in results.top_nodes:
            if nid in seen:
                continue
            seen.add(nid)

            # Reads go through the MemoryManager (.mem) — item 3, 2026-10-10.
            _mm = getattr(self._memory, "mem", None)
            if _mm is None:
                continue
            try:
                node = _mm.get_node(nid)
                text = (_mm.get_text(nid) or "")[:250]
                if not text.strip():
                    text = (_mm.get_summary(nid) or "")[:250]
                if not text.strip():
                    continue
            except Exception:
                continue

            flags = node.plane_flags if hasattr(node, "plane_flags") else 0
            try:
                wiki = _mm.get_wiki(nid) or {}
            except Exception:
                wiki = {}
            ns = wiki.get("_ns", "") or wiki.get("namespace", "") or ""

            is_causal = bool(flags & 2)  # Plane.CAUSAL.value
            is_rumination = ns in ("ruminate", "meta") or "ruminate" in str(
                wiki.get("topic", "") or wiki.get("process", "") or ""
            )

            if is_causal:
                causal_lines.append(text[:200])
            if is_rumination:
                meta_lines.append(text[:200])

        if not causal_lines and not meta_lines:
            return None

        # Only block if there's something meaningful OR it's a side-effect tool
        if not is_side_effect and not causal_lines and not meta_lines:
            return None

        # Build the context block
        parts = []
        if causal_lines:
            parts.append("<causal-memory>")
            for l in causal_lines[:3]:
                parts.append(f"  - {l}")
            parts.append("</causal-memory>")
        if meta_lines:
            parts.append("<meta-lessons>")
            for l in meta_lines[:3]:
                parts.append(f"  - {l}")
            parts.append("</meta-lessons>")

        context = "\n".join(parts)

        msg = (
            f"[Pre-Tool MPMF Causal Check]\n"
            f"Before calling {tool_name}, the MPMF memory system found "
            f"relevant knowledge:\n\n"
            f"{context}\n\n"
            f"You were about to call: {tool_name}({args_str})\n"
            f"Evaluate this context and proceed if appropriate — "
            f"or adjust your approach if the context suggests a better one."
        )

        # Cap message length so the agent's context window isn't flooded
        if len(msg) > 1200:
            msg = msg[:1200] + "\n\n[... truncated ...]"

        return {"action": "block", "message": msg}

    def _tool_recall(self, args: Dict[str, Any]) -> str:
        node_id = args.get("node_id")
        if node_id is not None:
            try:
                # Reads go through the MemoryManager (.mem) — item 3, 2026-10-10.
                _mm = getattr(self._memory, "mem", None)
                if _mm is None:
                    return json.dumps({"error": "memory manager unavailable"})
                n = _mm.get_node(int(node_id))
                if n is None:
                    return json.dumps({"found": False, "node_id": node_id})
                # Best-effort text dump
                txt = ""
                try:
                    txt = _mm.get_text(int(node_id)) or ""
                except Exception:
                    pass
                return json.dumps(
                    {
                        "found": True,
                        "node_id": node_id,
                        "text": txt[:2000],
                    },
                    ensure_ascii=False,
                )
            except Exception as e:
                return json.dumps({"error": f"node_id lookup failed: {e}"})

        query = args.get("query") or ""
        top_k = int(args.get("top_k") or 10)
        if not query.strip():
            return json.dumps({"error": "query is required when node_id is not set"})
        result = self._wrapper_recall(query, top_k)
        if result is None:
            return json.dumps({"error": "recall unavailable (wrapper and package both raised)"})
        hits = []
        raw = getattr(result, "raw_texts", {}) or {}
        summaries = getattr(result, "summaries", {}) or {}
        for nid in result.top_nodes or []:
            hits.append(
                {
                    "node_id": nid,
                    "summary": (summaries.get(nid) or "")[:600],
                    "text": (raw.get(nid) or "")[:1500],
                }
            )
        return json.dumps(
            {"query": query, "top_nodes": result.top_nodes, "hits": hits},
            ensure_ascii=False,
        )

    def _tool_store(self, args: Dict[str, Any]) -> str:
        text = args.get("text") or ""
        name = args.get("name") or f"fact_{int(time.time() * 1000)}"
        if not text.strip():
            return json.dumps({"error": "text is required"})
        # The wrapper exposes store() as a module-level function, not a method.
        # Use _get_wrapper_module (cache + sys.modules + reload) so this never
        # fails just because importlib didn't register in sys.modules.
        mod = _get_wrapper_module()
        if mod is None or not hasattr(mod, "store"):
            return json.dumps({
                "error": "MPMF+ store() unavailable",
                "reason": _WRAPPER_LOAD_ERROR or "wrapper missing store()",
            })
        try:
            # auto_extract=True — the LLM populates the 5 planes from the
            # fact, so they reflect real meaning rather than defaulting
            # to associative-only. auto_link=False — avoid the duplicate-
            # edge storm documented in rumination 2026-06-09.
            mod.store(text, name=name, auto_extract=True, auto_link=False)
            return json.dumps({"stored": True, "name": name, "chars": len(text)})
        except Exception as e:
            return json.dumps({"error": f"store failed: {e}"})


# -----------------------------------------------------------------------
# Plugin registration
# -----------------------------------------------------------------------


def register(ctx: Any) -> None:
    """Register the MPMF+ provider with the plugin context."""
    config = _load_plugin_config()
    provider = MPMFPlusProvider(config=config)
    if hasattr(ctx, "register_memory_provider"):
        ctx.register_memory_provider(provider)
    # Register the pre-tool causal+meta hook
    if hasattr(ctx, "register_hook"):
        ctx.register_hook("pre_tool_call", provider.on_pre_tool_call)


# Module-level instance for fallback discovery
_instance: Optional[MPMFPlusProvider] = None


def _get_instance() -> MPMFPlusProvider:
    global _instance
    if _instance is None:
        _instance = MPMFPlusProvider()
    return _instance
