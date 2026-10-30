"""Hermes adapter — the ONLY module in this package that knows Hermes exists.

Everything above this file is host-agnostic. Import it only when running inside
Hermes (it needs hermes memory_provider.MemoryProvider).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from . import block as _block
from . import memory as _memory

logger = logging.getLogger("mpmf_plus.adapter")


def _provider_base():
    try:
        from hermes.memory_provider import MemoryProvider  # type: ignore
        return MemoryProvider
    except Exception:
        return object


class MPMFPlusProvider(_provider_base()):  # type: ignore[misc]
    """Thin host binding: recall + render + store, no memory logic of its own."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config or {})
        self._top_k = int(self._config.get("top_k", 8))
        self._max_chars = int(self._config.get("max_chars", 3500))
        self._auto_write = bool(self._config.get("auto_write", True))
        self._turn_namespace = str(self._config.get("turn_namespace", "transcript"))
        self._session_id = ""
        self.last_sync_error: Optional[str] = None

    @property
    def name(self) -> str:
        return "mpmf_plus"

    def is_available(self) -> bool:
        try:
            _memory.stats()
            return True
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

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query or not query.strip():
            return ""
        shim = _block.wrapper_recall(query, self._top_k, _memory)
        if shim is None:
            return ""
        mem_mgr = None
        try:
            from .memory import _open
            mem_mgr = _open().mem
        except Exception:
            mem_mgr = None
        body, _items = _block.render_memory_block(
            query, shim, mem_mgr, self._top_k, max_chars=self._max_chars)
        return body

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[list] = None,
    ) -> None:
        """HOOK 2 (after the turn): persist the completed turn.

        Ported from the measured plugin (mpmf_plus_user sync_turn, 1:1 shape):
        a compact USER/ASSISTANT digest, skipped when the turn is too short to
        be worth a node, stored with ``auto_extract=True`` (5-plane payload,
        cached by text hash) and ``auto_link=False`` (the O(N) edge rebuild was
        72.9% duplicates in practice). Namespace ``transcript`` keeps per-turn
        traces out of normal ``recall()``, which defaults to ``["memory"]``.

        A failure here must not kill the host's turn, but it must not vanish
        either: it is logged at WARNING with a traceback and kept in
        ``last_sync_error``.
        """
        self.last_sync_error = None
        if not self._auto_write:
            return
        if not user_content or not assistant_content:
            return
        if len(user_content) + len(assistant_content) < 120:
            return  # too short to be worth a node

        sid = self._session_id or session_id or "-"
        digest = (
            f"[session={sid} t={int(time.time())}]\n"
            f"USER: {user_content[:600]}\n"
            f"ASSISTANT: {assistant_content[:600]}"
        )
        name = f"turn_{sid if sid != '-' else 'adhoc'}_{int(time.time() * 1000)}"
        try:
            _memory.store(digest, name=name, auto_extract=True, auto_link=False,
                          namespace=self._turn_namespace)
        except Exception as e:
            self.last_sync_error = f"{type(e).__name__}: {e}"
            logger.warning("MPMF+ turn write FAILED (turn not persisted): %s",
                           self.last_sync_error, exc_info=True)


def register(ctx: Any) -> None:
    ctx.register_memory_provider(MPMFPlusProvider())
