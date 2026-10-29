"""Hermes adapter — the ONLY module in this package that knows Hermes exists.

Everything above this file is host-agnostic. Import it only when running inside
Hermes (it needs hermes memory_provider.MemoryProvider).
"""
from __future__ import annotations

import logging
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
        self._session_id = ""

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

    def sync_turn(self, *args: Any, **kwargs: Any) -> None:
        return None


def register(ctx: Any) -> None:
    ctx.register_memory_provider(MPMFPlusProvider())
