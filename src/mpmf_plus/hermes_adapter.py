"""Hermes adapter — the ONLY module in this package that knows Hermes exists.

R89: all rendering, thinking and cognition logic moved into the package
(rendering.py / thinking.py / provider.recall_cognition). The host plugin is a
thin shim that imports and registers this class.
"""
from __future__ import annotations

import logging
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
            from . import memory as _memory
            _memory.store(digest, name=name, auto_extract=True, auto_link=False,
                          namespace=self._turn_namespace, said_at=time.time(),
                          conv_id=sid)
        except Exception as e:
            self.last_sync_error = f"{type(e).__name__}: {e}"
            logger.warning("MPMF+ turn write FAILED (turn not persisted): %s",
                           self.last_sync_error, exc_info=True)


def register(ctx: Any) -> None:
    ctx.register_memory_provider(MPMFPlusProvider())
