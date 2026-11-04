"""mpmf_plus.ingest — one turn in, facts out. Per-turn ingestion as part of MPMF.

Round 83. Everything the wrapper used to assemble by hand around `store()` lives
here now, in the package:

    turn text (+ the 2 messages before it)
        -> frozen 67e extraction prompt          (rebuild/prompt_67e.txt)
        -> facts with entity / evidence / fact_date / msg_index
        -> kind + mode decided with the context  (rebuild/stage2.classify_text)
        -> dedup on the float32 matrix path      (dedup_helpers.check_dedup)
        -> ONE write_txn, ONE store.log line

The embedder is the MPMF default (mxbai-embed-large, 1024-d) and the store's
stamp is enforced: a store stamped with a different embedder raises instead of
silently mixing vector spaces.

Public API:
    ingest(turn_text, said_at=None, conv_id=None, msg_index=0, context=None, ...)
        -> {"conv_id", "msg_index", "status", "written", "skipped", "store_ms"}

Nothing here calls the wrapper. Athena's `store()` and Ayu's harness both call
this one function.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from . import dedup_helpers as _dedup
from mpmf.embed_text import embed_text  # r83f: ONE embed rule
from . import memory as _mem
from . import store as _store
from .rebuild import extraction as _x
from .rebuild import stage2 as _stage2

DEFAULT_EMBED_MODEL = "mxbai-embed-large"
DEFAULT_EMBED_DIM = 1024
# No hardcoded root path in the package (test_no_hardcoded_root_paths_in_src):
# the caller passes its own log path; the env var is the fallback.
DEFAULT_STORE_LOG = os.environ.get("MPMF_STORE_LOG", "")

#: Persisted on every fact node. `kind`/`mode` are stage-2 outputs, the rest are
#: the 67e provenance schema (r70's lesson: parse it or lose it).
FACT_FIELDS = ("entity", "_msg_index", "_evidence", "_fact_date", "_said_at",
               "_conv_id", "kind", "_fact_mode", "_ns")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_call(body: str) -> str:
    """One extraction call through the package's own client (no wrapper code).

    Probed 2026-11-03 (r83b): _llm_complete returns {"content", "finish_reason",
    "model"}, and the upstream sometimes puts the whole answer in `reasoning`
    and leaves `content` as None (probe 1: content None / finish_reason "";
    probe 2, same body: content '{"facts":[]}' / finish_reason "stop").
    The old code did json.dumps(None) -> "null" -> the misleading
    "reply has no 'facts' list". Retry once on empty content, then fail loudly:
    a turn must never be silently zero.
    """
    r = _mem._llm_complete(body, temperature=0.0, max_tokens=2000, reasoning=False)
    for _ in range(2):
        c = r.get("content") if isinstance(r, dict) else r
        if isinstance(c, (dict, list)):
            return json.dumps(c)
        if isinstance(c, str) and c.strip():
            s = c.strip()
            if s.startswith("```"):                      # ```json fences
                s = s.split("\n", 1)[1] if "\n" in s else s[3:]
                if s.rstrip().endswith("```"):
                    s = s.rstrip()[:-3]
            return s.strip()
        r = _mem._llm_complete(body, temperature=0.0, max_tokens=2000, reasoning=False)
    fr = r.get("finish_reason") if isinstance(r, dict) else None
    raise _x.ExtractionError(
        "empty extraction content (finish_reason=%r) — provider returned reasoning only" % (fr,))


def _pack(vec: List[float], dim: int = DEFAULT_EMBED_DIM) -> bytes:
    if vec is None:
        raise ValueError("ingest: embedder returned None")
    if len(vec) != dim:
        raise ValueError("ingest: vector must be %d floats, got %d" % (dim, len(vec)))
    if not any(vec):
        raise ValueError("ingest: embedder returned an all-zero vector")
    return struct.pack("<%df" % dim, *vec)


def _store_log(line: str, path: Optional[str]) -> None:
    """One line per ingest. Append-only; never raises into the caller."""
    path = path if path is not None else DEFAULT_STORE_LOG
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _context_mode(sentence: str, context: Optional[List[Dict[str, str]]]) -> Dict[str, str]:
    """kind from the fact sentence; mode may be narrowed by the 2 messages before.

    Round 71: "I'd love a boat" is only hypothetical in context — the sentence
    alone says interest, the window says if/suppose/would.
    """
    cls = _stage2.classify_text(sentence)
    if context:
        window = " ".join((m.get("text") or "") for m in context[-2:]) + " " + sentence
        wcls = _stage2.classify_text(window)
        if wcls["mode"] in ("hypothetical", "temporary"):
            cls["mode"] = wcls["mode"]
    return cls


def ingest(turn_text: str,
           said_at: Optional[str] = None,
           conv_id: Optional[str] = None,
           msg_index: int = 0,
           context: Optional[List[Dict[str, str]]] = None,
           model: Optional[str] = None,
           namespace: str = "memory",
           store: Any = None,
           call_model: Optional[Callable[[str], str]] = None,
           embed_fn: Optional[Callable[[str], List[float]]] = None,
           log_path: Optional[str] = None,
           embed_model: str = DEFAULT_EMBED_MODEL,
           embed_dim: int = DEFAULT_EMBED_DIM) -> Dict[str, Any]:
    """Ingest ONE turn into facts. All writes happen in a single write_txn.

    Args:
        turn_text: the turn's text (user message; assistant text may follow).
        said_at: ISO timestamp; defaults to now UTC.
        conv_id: conversation id; defaults to a fresh `turn_<hex>`.
        msg_index: index of this message inside the conversation.
        context: the messages before this one (pass the last 2 for r71 parity).
        model: extraction model override.
        namespace: `_ns` for the facts (default "memory").
        store: an MPMFMemory-like object; defaults to the configured store.
        call_model: injection point for tests (body -> raw JSON string).
        embed_fn: injection point for tests (text -> list[float]).
        log_path: store.log path; "" disables.

    Returns a dict with status ("ok" | "empty"), the written fact ids and the
    dedup-skipped ones. Raises on extraction failure — a failed turn is never
    silently zero.
    """
    t0 = time.perf_counter()
    conv_id = conv_id or ("turn_" + uuid.uuid4().hex[:12])
    said_at = said_at or _utc()
    msg_index = int(msg_index or 0)
    messages = list(context or []) + [{"role": "user", "text": turn_text}]
    conv = {"id": conv_id, "start": said_at, "messages": messages}

    body = _x.PROMPT + "\n\nMESSAGES:\n" + _x._window(conv)
    raw = (call_model or _default_call)(body)
    facts = _x._parse_reply(raw)

    # r83c: the earlier messages are CONTEXT for extraction and mode only. A
    # fact the extractor attributes to one of them belongs to an earlier turn,
    # which has (or will have) its own ingest pass; writing it here stamps the
    # wrong _said_at/_conv_id/_msg_index on it. r83b's judge caught exactly
    # that: 3/20 facts were about the previous message but dated to this turn.
    # None = the extractor omitted msg_index; treated as the current message.
    cur_msg = len(messages) - 1
    n_parsed = len(facts)
    facts = [f for f in facts if f.get("msg_index") in (None, cur_msg)]
    dropped_ctx = n_parsed - len(facts)

    mem = store if store is not None else _mem._open_rw()
    core = getattr(mem, "mem", mem)
    txn = getattr(mem, "write_txn", None) or getattr(core, "write_txn", None)
    if txn is None:
        raise RuntimeError("ingest: store has no write_txn — refusing an unguarded write")

    # Embedder stamp: adopt mxbai on a virgin store, refuse a foreign stamp.
    try:
        _store.assert_embedder(core, embed_model, embed_dim)
        if _store.read_embedder(core) is None:
            _store.stamp_embedder(core, embed_model, embed_dim)
    except Exception:
        raise

    embed = embed_fn or (lambda t: mem.embed(t) if hasattr(mem, "embed")
                         else _mem._embed_floats_local(t, dim=embed_dim))

    written: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    # R87: facts are durable memory. A caller may pass namespace="transcript"
    # for the RAW turn node, but extracted facts must not inherit it — that made
    # every user fact invisible to recall (default namespaces=('memory',)).
    fact_ns = namespace if namespace in ("memory", "ruminate", "cognition",
                                         "meta", "cognition_staging") else "memory"
    with txn():
        for n, f in enumerate(facts):
            sentence = f["sentence"]
            action, kept = _dedup.check_dedup(sentence, fact_ns, core,
                                             lambda t, dim=embed_dim: embed(embed_text(t)))
            if action in ("exact_match", "near_dup"):
                skipped.append({"entity": f.get("entity"), "action": action,
                                "kept": int(kept) if kept is not None else None})
                continue
            cls = _context_mode(sentence, context)
            wiki = {
                "kind": cls["kind"],
                "_fact_mode": cls["mode"],
                "entity": f.get("entity") or "other",
                "_evidence": (f.get("evidence") or "")[:300],
                "_fact_date": f.get("fact_date") or said_at[:10],
                "_fact_date_src": "extractor" if f.get("fact_date") else "said_at",
                "_said_at": said_at,
                "_conv_id": conv_id,
                "_msg_index": str(msg_index),
                "_ns": fact_ns,
                "_source": "ingest",
                "_ingest_model": model or _mem.EXTRACT_MODEL,
                "_extraction_model": model or _mem.EXTRACT_MODEL,
            }
            missing = [k for k in FACT_FIELDS if wiki.get(k) in (None, "")]
            if missing:
                raise AssertionError("ingest: fact %d missing %s" % (n, missing))
            vec = _pack(embed(embed_text(sentence)), embed_dim)
            # R87: "memfact_" prefix — "turn_"/"sess_" would classify as
            # transcript and "fact_" as ruminate; this prefix classifies memory.
            name = "memfact_%s_%d_%d" % (conv_id[:10], msg_index, n)
            node = core.create_node(name=name, node_type=1, plane_flags=1,
                                    text=sentence, summary=sentence[:120],
                                    wiki=wiki, vector=vec,
                                    importance=float(0.5 + int(wiki["kind"] == "have") * 0.1))
            written.append({"name": name, "node_id": int(getattr(node, "node_id", node)),
                            "entity": wiki["entity"], "kind": cls["kind"],
                            "mode": cls["mode"], "text": sentence})

    ms = (time.perf_counter() - t0) * 1000.0
    status = "ok" if written else "empty"
    if log_path != "":
        _store_log("%s ingest_ms=%.0f facts=%d skipped=%d dropped_ctx=%d status=%s conv=%s msg=%d"
                   % (_utc(), ms, len(written), len(skipped), dropped_ctx, status, conv_id, msg_index),
                   log_path)
    return {"conv_id": conv_id, "msg_index": msg_index, "status": status,
            "written": written, "skipped": skipped, "dropped_ctx": dropped_ctx,
            "store_ms": round(ms, 1)}


__all__ = ["ingest", "FACT_FIELDS", "DEFAULT_EMBED_MODEL", "DEFAULT_EMBED_DIM"]
