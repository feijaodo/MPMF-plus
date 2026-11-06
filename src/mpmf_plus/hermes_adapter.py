"""Hermes adapter — the ONLY module in this package that knows Hermes exists.

R89: all rendering, thinking and cognition logic moved into the package
(rendering.py / thinking.py / provider.recall_cognition). The host plugin is a
thin shim that imports and registers this class.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

from . import block as _block
from . import paths as _paths
from . import rendering as _rendering

logger = logging.getLogger("mpmf_plus.adapter")


def _provider_base():
    try:
        from hermes.memory_provider import MemoryProvider  # type: ignore
        return MemoryProvider
    except Exception:
        return object


def _log_fail(where: str, exc: BaseException) -> str:
    """Append a failure + traceback to failures.log. Never debug-only.

    Round 90b: sync_turn swallowed its own TypeError into logger.warning, so
    three rounds of "memory writes are broken" produced no evidence anywhere
    a human looks.
    """
    import traceback
    msg = "%s: %s: %s" % (where, type(exc).__name__, exc)
    try:
        path = _paths.log_path("failures.log")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write("\n%s adapter FAIL %s\n%s\n"
                    % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg,
                       "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))))
    except Exception:
        logger.error("MPMF+ could not write failures.log: %s", msg, exc_info=True)
    logger.error("MPMF+ %s", msg, exc_info=True)
    return msg


class MPMFPlusProvider(_provider_base()):  # type: ignore[misc]
    """Thin host binding: recall + render + store, no memory logic of its own."""

    SIDE_EFFECT_TOOLS = {
        "terminal", "write_file", "patch", "execute_code", "send_message",
        "cronjob", "delegate_task", "browser_navigate", "browser_click",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = _paths.load_config()
        cfg.update(dict(config or {}))
        self._cfg = cfg
        self._top_k = int(cfg.get("top_k", 8))
        self._max_chars = int(cfg.get("max_chars", 3500))
        self._auto_write = bool(cfg.get("auto_write", True))
        self._turn_namespace = str(cfg.get("turn_namespace", "transcript"))
        self._pre_tool_causal = bool(cfg.get("pre_tool_causal", True))
        self._session_id = ""
        self.last_sync_error: Optional[str] = None

    @property
    def name(self) -> str:
        return "mpmf_plus"

    def is_available(self) -> bool:
        try:
            from . import provider as _p
            return bool(_p.stats())
        except Exception:
            return False

    def initialize(self, session_id: str = "", **kwargs: Any) -> None:
        self._session_id = session_id
        if not getattr(self, "_warmed", False):
            self._warmed = True
            # r90c: warm SYNCHRONOUSLY. As a thread it raced the first prefetch,
            # both paid the cold cost, and the host's 8 s ceiling still tripped
            # on every restart (01:37:51). Init is not on that clock.
            if os.environ.get("MPMF_WARM_ASYNC") == "1":
                import threading
                threading.Thread(target=self._warm_start, name="mpmf-warm",
                                 daemon=True).start()
            else:
                self._warm_start()

    def _warm_start(self) -> None:
        """Open the store and run one embed so the first prefetch is warm.

        Cold start cost ~9-10 s (store map + 1.5 GB embedder), which blew the
        host's 8 s prefetch ceiling on every restart.
        """
        try:
            from . import provider as _prov
            t0 = time.perf_counter()
            mem = _prov._open().mem
            try:
                _prov.embed_text("warm")
            except Exception:
                pass
            # A warm embed is not enough: the first recall also pays index
            # expansion and candidate scoring (~5.5 s on 24k nodes). Run one
            # full recall here, off the host's prefetch clock.
            try:
                _prov.recall("warm up query for the memory pipeline", top_n=3)
            except Exception:
                pass
            logger.info("MPMF+ warm start: store+embedder ready in %.1fs (nodes=%d)",
                        time.perf_counter() - t0, getattr(mem.header, "node_count", 0))
        except Exception as e:
            _log_fail("warm_start", e)

    def shutdown(self) -> None:
        return None

    def system_prompt_block(self) -> str:
        return (
            "# MPMF+ Memory (ACTIVE)\n"
            "Relevant context is recalled per turn and injected as <memory-context>; "
            "treat it as reference, not new input."
        )

    def _thinking(self, query: str) -> str:
        if not self._cfg.get("inject_thinking", True):
            return ""
        try:
            from . import thinking as _t
            results = _t.recall_thinking(
                query, top_n=int(self._cfg.get("thinking_top_k", 3)))
            return _rendering.thinking_block(
                results, max_chars=int(self._cfg.get("thinking_max_chars", 800)))
        except Exception as e:
            logger.debug("thinking recall failed: %s", e)
            return ""

    def _cognition(self, query: str, query_vec=None) -> str:
        if not self._cfg.get("inject_cognition", True):
            return ""
        try:
            from . import provider as _p
            # r89c parity: the pre89 plugin passed vet=True (sidecar-vetted
            # lessons only, round 43) and log_inject=True (round 39 injection
            # log). Omitting vet=True made the shim inject UNVETTED lessons:
            # 129/129 blocks carried <prior-reasoning> instead of 47/129.
            kw = {}
            if query_vec is None:
                # r89c parity: the pre89 plugin always passed the shared query
                # vector. Without it the cosine arm cannot run and the lexical
                # arm answers nearly every query (129/129 vs 47/129 blocks).
                try:
                    query_vec = _p.embed_query(query)
                except Exception:
                    query_vec = None
            if query_vec is not None:
                kw["query_vec"] = query_vec
            raw = _p.recall_cognition(
                query, top_k=int(self._cfg.get("cognition_top_k", 3)),
                mode="all", bump_counts=False, vet=True, log_inject=True, **kw)
            return _rendering.cognition_block(
                raw or "", max_chars=int(self._cfg.get("cognition_max_chars", 700)))
        except Exception as e:
            logger.debug("cognition recall failed: %s", e)
            return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query or not query.strip():
            return ""
        try:
            from . import provider as _p
            shim = _block.wrapper_recall(query, self._top_k, _p)
            if shim is None:
                return ""
            mem_mgr = None
            try:
                mem_mgr = _p._open().mem
            except Exception:
                mem_mgr = None
            body = _rendering.render_memory(
                query, shim, mem_mgr, self._top_k, max_chars=self._max_chars)
        except Exception as e:
            logger.warning("memory render failed: %s", e)
            body = ""
        return _rendering.compose(body, self._thinking(query), self._cognition(query))

    def on_pre_tool_call(self, *, tool_name: str = "", args: Any = None,
                         task_id: str = "") -> Optional[Dict[str, Any]]:
        """Pre-tool hook: surface causal + meta lessons for this tool call."""
        if not self._pre_tool_causal:
            return None
        is_side_effect = tool_name in self.SIDE_EFFECT_TOOLS
        args_str = ""
        if isinstance(args, dict):
            parts = [tool_name]
            for k, v in args.items():
                if isinstance(v, str) and len(v) < 150 and not v.startswith("MEDIA:"):
                    parts.append(f"{k}={v}")
                elif k in ("path", "name", "pattern", "command", "url", "goal"):
                    parts.append(f"{k}={str(v)[:120]}")
            args_str = " ".join(parts[:8])
        else:
            args_str = f"{tool_name} {str(args)[:150]}"
        if not args_str.strip():
            return None
        try:
            from . import block as _b
            from . import provider as _p
            res = _b.wrapper_recall(args_str, 5, _p)
            mm = _p._open().mem
        except Exception as e:
            logger.debug("pre-tool recall failed: %s", e)
            return None
        if not res or not getattr(res, "top_nodes", None) or mm is None:
            return None
        causal, meta, seen = [], [], set()
        for nid in res.top_nodes:
            if nid in seen:
                continue
            seen.add(nid)
            try:
                node = mm.get_node(nid)
                text = (mm.get_text(nid) or "")[:250]
                if not text.strip():
                    text = (mm.get_summary(nid) or "")[:250]
                if not text.strip():
                    continue
            except Exception:
                continue
            flags = node.plane_flags if hasattr(node, "plane_flags") else 0
            try:
                wiki = mm.get_wiki(nid) or {}
            except Exception:
                wiki = {}
            ns = wiki.get("_ns", "") or wiki.get("namespace", "") or ""
            if flags & 2:  # Plane.CAUSAL
                causal.append(text[:200])
            if ns in ("ruminate", "meta") or "ruminate" in str(
                    wiki.get("topic", "") or wiki.get("process", "") or ""):
                meta.append(text[:200])
        if not causal and not meta:
            return None
        blocks = []
        if causal:
            blocks.append("<causal-memory>")
            blocks += [f"  - {l}" for l in causal[:3]]
            blocks.append("</causal-memory>")
        if meta:
            blocks.append("<meta-lessons>")
            blocks += [f"  - {l}" for l in meta[:3]]
            blocks.append("</meta-lessons>")
        msg = (
            f"[Pre-Tool MPMF Causal Check]\n"
            f"Before calling {tool_name}, the MPMF memory system found "
            f"relevant knowledge:\n\n" + "\n".join(blocks) +
            f"\n\nYou were about to call: {tool_name}({args_str})\n"
            f"Evaluate this context and proceed if appropriate — "
            f"or adjust your approach if the context suggests a better one."
        )
        if len(msg) > 1200:
            msg = msg[:1200] + "\n\n[... truncated ...]"
        return {"action": "block", "message": msg}

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages: Optional[list] = None) -> None:
        """HOOK 2: persist the completed turn (namespace transcript, auto_link off)."""
        self.last_sync_error = None
        if not self._auto_write or not user_content or not assistant_content:
            return
        if len(user_content) + len(assistant_content) < 120:
            return
        sid = self._session_id or session_id or "-"
        digest = (
            f"[session={sid} t={int(time.time())}]\n"
            f"USER: {user_content[:600]}\n"
            f"ASSISTANT: {assistant_content[:600]}"
        )
        name = f"turn_{sid if sid != '-' else 'adhoc'}_{int(time.time() * 1000)}"
        try:
            # r90b: write through ingest, not memory.store. memory.store()
            # returned cleanly while persisting nothing for turn digests, which
            # is how three rounds of turns vanished without one error line.
            from . import provider as _prov
            # r90d: bind ingest to the Hermes session BEFORE the turn. Without
            # this, _ingest_turn falls back to conv=turn_<epoch>, so every turn
            # is its own conversation and history/sequence can never work.
            if sid and sid != "-":
                _prov.set_ingest_session(sid)
            _prov._ingest_turn(digest, name=name, summary="", wiki=None,
                               namespace=self._turn_namespace, importance=0.6)
        except Exception as e:
            self.last_sync_error = _log_fail("sync_turn", e)


    def on_session_end(self, *args: Any, **kwargs: Any) -> None:
        """Host calls this when a session closes.

        sync_turn already persisted the turn, so there is nothing to flush.
        Defined explicitly because the host logs a warning for every session
        that ends without the attribute (seen at 01:13:25 on 79c157d).
        """
        return None

    # -- host tool API (required at memory-provider init) -----------------
    def get_tool_schemas(self):
        return [
            {
                "name": "mpmf_recall",
                "description": ("Ad-hoc deep lookup against the MPMF+ memory graph. "
                                "Use when the injected context is not enough."),
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string"},
                    "node_id": {"type": "integer"},
                    "top_k": {"type": "integer", "default": 10}}},
            },
            {
                "name": "mpmf_store",
                "description": "Store a durable fact in MPMF+ memory.",
                "parameters": {"type": "object",
                               "properties": {"text": {"type": "string"},
                                              "name": {"type": "string"}},
                               "required": ["text"]},
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        import json
        try:
            from . import provider as _p
            if tool_name == "mpmf_recall":
                nid = args.get("node_id")
                if nid is not None:
                    mm = _p._open().mem
                    return json.dumps({"found": True, "node_id": nid,
                                       "text": (mm.get_text(int(nid)) or "")[:2000]},
                                      ensure_ascii=False)
                hits = _p.recall(args.get("query") or "", top_n=int(args.get("top_k") or 10)) or []
                out = []
                for h in hits[:12]:
                    if isinstance(h, dict):
                        out.append({"id": h.get("id") or h.get("node_id"),
                                    "text": str(h.get("text") or h.get("summary") or "")[:300]})
                    else:
                        out.append({"text": str(h)[:300]})
                return json.dumps({"hits": out}, ensure_ascii=False)
            if tool_name == "mpmf_store":
                text = (args.get("text") or "").strip()
                if not text:
                    return json.dumps({"error": "text is required"})
                nid = _p.store(text, args.get("name") or "fact_%d" % int(time.time() * 1000),
                               auto_extract=True, auto_link=False)
                return json.dumps({"ok": True, "id": nid})
            return json.dumps({"error": "unknown tool: %s" % tool_name})
        except Exception as e:
            return json.dumps({"error": "%s: %s" % (type(e).__name__, e)})


def register(ctx: Any) -> None:
    ctx.register_memory_provider(MPMFPlusProvider())
