"""Raw-message layer (P8, RESEARCH): every real user message as a searchable node.

WHY
---
Extraction keeps what the model judged worth a fact. Whatever it dropped, merged or
paraphrased away is unreachable: "never stored" misses. The raw layer keeps the user's
own words, verbatim, in a namespace of their own (``_ns == "raw"``) next to the facts.

DESIGN
------
* Nodes: name ``rawmsg_<sha1(text)[:20]>`` (so an exact repeat is found by name, O(1)),
  text = the message, vector = ``embed_text(message)`` like every other node.
  wiki: ``_ns=raw, _source=raw_layer, _said_at, _conv_id, _msg_index``. No ``entity``, so
  the history index and ``history_block`` never see these nodes.
* Isolation: the default recall asks for namespaces ``("memory",)``; raw nodes can never
  enter it. They are reached only through ``search()`` below.
* Fusion (flag ``raw_layer``, default OFF): the facts block is built exactly as before;
  ``prefetch_section`` adds a SEPARATE verbatim section after it, so the raw hits can
  neither be pushed out by the facts nor cut by the facts' block cap, and the facts can
  never be pushed out by the raw hits. Candidates: cosine over the raw rows only, floor
  ``raw_min_cos``, top 20 -> cross-encoder (same model, same timeout rule as d20) -> top
  ``raw_k``.
* Privacy: text that looks like a secret, an e-mail or a phone number, and host
  "[System ..." injections, are never stored.

Config (recall.config.json or ``MPMF_<KEY>`` env): raw_layer (false), raw_k (3),
raw_min_cos (0.5), raw_rerank (true), raw_item_chars (500), raw_block_chars (1500).
"""
from __future__ import annotations

import hashlib
import os
import re
import struct
import time
from typing import Any, Dict, Iterable, List, Optional

from . import paths as _paths

NS = "raw"
MIN_CHARS = 40
MAX_TEXT = 4000
RERANK_CHARS = 700
DIM = 1024

SECRET = [re.compile(p) for p in (
    r"sk-or-v1-[A-Za-z0-9]{20,}", r"sk-[A-Za-z0-9_\-]{32,}", r"gh[pousr]_[A-Za-z0-9]{30,}",
    r"AKIA[0-9A-Z]{16}", r"xox[abp]-[A-Za-z0-9-]{10,}", r"\b\d{8,10}:[A-Za-z0-9_\-]{35}\b",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"(?i)(api[_-]?key|secret|token|password|passwd)\s*[=:]\s*['\"]?[A-Za-z0-9_\-\.]{12,}",
    r"(?i)bearer\s+[A-Za-z0-9_\-\.]{24,}", r"hf_[A-Za-z0-9]{30,}",
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
    r"(?<!\d)\+?\d{1,2}[ .\-]?\(?\d{3}\)?[ .\-]\d{3}[ .\-]\d{4}(?!\d)")]
SYSTEM_PREFIX = ("[System", "[ASYNC", "[IMPORTANT", "[CONTEXT", "You've reached the maximum",
                 "[Note:", "[SYSTEM")

#: last search timings/hit info (read by the eval harness; never used for logic)
LAST: Dict[str, Any] = {}


# -- config -----------------------------------------------------------------
def settings() -> Dict[str, Any]:
    c = _paths.load_config()
    return {
        "on": bool(c.get("raw_layer", False)),
        "k": max(1, int(c.get("raw_k", 3) or 3)),
        "min_cos": float(c.get("raw_min_cos", 0.5) or 0.0),
        "rerank": bool(c.get("raw_rerank", True)),
        "item_chars": max(80, int(c.get("raw_item_chars", 500) or 500)),
        "block_chars": max(200, int(c.get("raw_block_chars", 1500) or 1500)),
    }


def enabled() -> bool:
    try:
        return bool(settings()["on"])
    except Exception:
        return False


# -- what may be stored -----------------------------------------------------
def storable(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < MIN_CHARS or t.startswith(SYSTEM_PREFIX):
        return False
    return not any(p.search(t) for p in SECRET)


def node_name(text: str) -> str:
    return "rawmsg_" + hashlib.sha1((text or "").strip().encode("utf-8", "replace")).hexdigest()[:20]


def _pack(vec) -> bytes:
    if vec is None or len(vec) != DIM or not any(vec):
        raise ValueError("raw_layer: embedder returned a bad vector")
    return struct.pack("<%df" % DIM, *vec)


def _create(core, text: str, vec: bytes, said_at: str, conv_id: str, msg_index: int) -> Optional[int]:
    """Create one raw node inside an open write txn. None if the exact text exists."""
    text = text.strip()[:MAX_TEXT]
    name = node_name(text)
    if core.get_node_id_by_name(name) is not None:
        return None
    wiki = {"_ns": NS, "_source": "raw_layer", "_said_at": said_at or "",
            "_conv_id": str(conv_id or ""), "_msg_index": str(int(msg_index or 0)),
            "kind": "raw"}
    node = core.create_node(name=name, node_type=1, plane_flags=1, text=text,
                            summary=text[:120], wiki=wiki, vector=vec, importance=0.4)
    return int(getattr(node, "node_id", node))


# -- write: live path (one turn) ---------------------------------------------
def add_turn(user_text: str, conv_id: str = "", msg_index: int = 0,
             said_at: Optional[str] = None, embed_fn=None) -> str:
    """Store ONE real user message. Returns 'stored' | 'dup' | 'skipped'. Raises on I/O."""
    if not storable(user_text):
        return "skipped"
    from . import provider as P
    said_at = said_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    from mpmf.embed_text import embed_text
    text = user_text.strip()[:MAX_TEXT]
    # embed BEFORE the write lock (same rule as ingest, r90e)
    vec = _pack((embed_fn or P._embed_floats_local)(embed_text(text)))
    with P._STORE_LOCK, P._write_txn_open() as m:
        nid = _create(m.mem, text, vec, said_at, conv_id, msg_index)   # write_txn saves on exit
    P._invalidate_open_cache()
    return "stored" if nid is not None else "dup"


# -- write: backfill (many messages, precomputed vectors) ---------------------
def backfill(items: Iterable[Dict[str, Any]], vectors, store_path: str,
             allow_live: bool = False, lock_timeout: float = 600.0) -> Dict[str, int]:
    """Write raw nodes for ``items`` (dicts text/date/conv/idx) with row-aligned ``vectors``.

    One write transaction. Refuses the live store unless allow_live (deploy decision).
    """
    import numpy as np
    from . import provider as P
    live = os.path.realpath(str(_paths.store_path()))
    if os.path.realpath(store_path) == live and not allow_live:
        raise PermissionError("raw_layer.backfill: %s is the live store; pass allow_live=True "
                              "(a deploy decision) to write to it" % store_path)
    items = list(items)
    if len(items) != len(vectors):
        raise ValueError("raw_layer.backfill: %d items vs %d vectors" % (len(items), len(vectors)))
    prev = P.configure(PATH=store_path)
    P._invalidate_open_cache()
    n_new = n_dup = n_skip = 0
    try:
        with P._STORE_LOCK, P._write_txn_open(timeout=lock_timeout) as m:
            core = m.mem
            for it, v in zip(items, vectors):
                if not storable(it["text"]):
                    n_skip += 1
                    continue
                vec = _pack(np.asarray(v, dtype=np.float32).tolist())
                said = (it.get("date") or "")[:10]
                said = said + "T00:00:00Z" if said else ""
                nid = _create(core, it["text"], vec, said, it.get("conv", ""), it.get("idx", 0))
                if nid is None:
                    n_dup += 1
                else:
                    n_new += 1
    finally:
        P.configure(**prev)
        P._invalidate_open_cache()
    return {"written": n_new, "dup": n_dup, "skipped": n_skip}


# -- read -------------------------------------------------------------------
_RAW_CACHE: Dict[str, Any] = {"key": None}


def _raw_matrix(mem):
    """(raw node ids, their float32 rows, norms) — rebuilt only when the store version moves."""
    import numpy as np
    from . import provider as P
    ids, M, norms, keep = P._d20_node_matrix(mem)
    key = (id(mem), P._D20_MATRIX.get("key"), len(ids))
    if _RAW_CACHE["key"] == key:
        return _RAW_CACHE["ids"], _RAW_CACHE["M"], _RAW_CACHE["norms"]
    rows = [i for i, nid in enumerate(ids) if (mem.get_wiki(nid) or {}).get("_ns") == NS]
    rows_a = np.asarray(rows, dtype=np.int64)
    _RAW_CACHE.update({"key": key, "ids": [ids[i] for i in rows],
                       "M": M[rows_a] if len(rows) else M[:0],
                       "norms": norms[rows_a] if len(rows) else norms[:0]})
    return _RAW_CACHE["ids"], _RAW_CACHE["M"], _RAW_CACHE["norms"]


def search(query: str, qvec: Optional[List[float]], mem=None, k: int = 3, min_cos: float = 0.5,
           rerank: bool = True, budget_start: Optional[float] = None,
           n_cand: int = 20) -> List[Dict[str, Any]]:
    """Top-k raw messages for the query (cosine floor -> optional cross-encoder)."""
    import numpy as np
    from . import provider as P
    t0 = time.perf_counter()
    bstart = budget_start if budget_start is not None else t0
    if mem is None:
        mem = P._open().mem
    if qvec is None:
        qvec = P.embed_query(query)
    ids, M, norms = _raw_matrix(mem)
    LAST.clear()
    LAST.update({"n_raw": len(ids), "rerank": "off", "hits": 0})
    if not ids:
        return []
    q = np.asarray(qvec, dtype=np.float32)
    sims = (M @ q) / (norms * (float(np.linalg.norm(q)) or 1.0) + 1e-9)
    order = np.argsort(-sims)[:n_cand]
    cands = []
    for r in order:
        if float(sims[r]) < min_cos:
            break
        nid = ids[int(r)]
        w = mem.get_wiki(nid) or {}
        cands.append({"rank": len(cands) + 1, "node_id": nid, "text": mem.get_text(nid) or "",
                      "cos": float(sims[r]), "date": str(w.get("_said_at") or "")[:10],
                      "conv": str(w.get("_conv_id") or ""), "name": mem.get_name(nid)})
    LAST["n_cand"] = len(cands)
    out = cands[:k]
    if rerank and len(cands) > 1:
        elapsed_ms = (time.perf_counter() - bstart) * 1000.0
        remaining_s = max((P._prefetch_budget_ms() - elapsed_ms) / 1000.0, 0.05)
        # the encoder sees what the model will see (the first RERANK_CHARS chars): rerank cost
        # grows with text length and a raw message can be thousands of characters
        by_id = {c["node_id"]: c for c in cands}
        short = [dict(c, text=c["text"][:RERANK_CHARS]) for c in cands]
        rr = P._rerank_with_timeout(query, short, len(short), remaining_s)
        if rr is None:
            LAST["rerank"] = "timeout"
        else:
            LAST["rerank"] = "done"
            out = [dict(by_id[r["node_id"]], reranker_score=r.get("reranker_score")) for r in rr[:k]]
    LAST.update({"hits": len(out), "ms": (time.perf_counter() - t0) * 1000.0,
                 "top_cos": cands[0]["cos"] if cands else None})
    return out


def render(hits: List[Dict[str, Any]], item_chars: int = 500, block_chars: int = 1500) -> str:
    if not hits:
        return ""
    lines = ["# Raw messages you wrote (verbatim; most relevant first)"]
    used = len(lines[0])
    for h in hits:
        text = " ".join((h.get("text") or "").split())
        if len(text) > item_chars:
            text = text[:item_chars] + "…"
        line = "- %s: %s" % (h.get("date") or "undated", text)
        if used + len(line) + 1 > block_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines) if len(lines) > 1 else ""


def prefetch_section(query: str, qvec: Optional[List[float]], budget_start: Optional[float] = None) -> str:
    """The extra block for prefetch(); "" when the flag is off or anything fails."""
    try:
        s = settings()
        if not s["on"]:
            return ""
        hits = search(query, qvec, k=s["k"], min_cos=s["min_cos"], rerank=s["rerank"],
                      budget_start=budget_start)
        LAST["hit_texts"] = [h.get("text", "") for h in hits]
        return render(hits, s["item_chars"], s["block_chars"])
    except Exception:
        return ""
