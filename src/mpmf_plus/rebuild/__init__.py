"""Rebuild a MPMF+ store from a conversation export.

Stages
------
1. ``extraction`` -- one LLM call per conversation using the frozen 67e prompt,
   with retry + explicit ``failed`` status (an unparseable reply is NEVER counted
   as "zero facts"), persisting ``_msg_index`` / ``_evidence`` / ``_fact_date``.
2. ``stage2``     -- kind/mode classification, hub states (newest ACTUAL
   change/have wins), categories, IS_A attached in the same transaction.
3. ``retag``      -- context-aware mode re-tag: re-reads each fact's source
   message plus the two before it and re-marks actual / hypothetical / unclear.

Entry point: ``mpmf-rebuild --input export.jsonl --out store.mpmf --cap 2.00``
"""

from .extraction import (  # noqa: F401
    PROMPT,
    ExtractionError,
    cap_exceeded,
    extract_conversation,
    load_export,
    persist_facts,
    run_extraction,
)
from .stage2 import build_states, classify_facts  # noqa: F401
from .retag import retag_modes  # noqa: F401

__all__ = [
    "PROMPT", "ExtractionError", "cap_exceeded", "classify_facts",
    "build_states", "extract_conversation", "load_export", "persist_facts",
    "retag_modes", "run_extraction",
]
