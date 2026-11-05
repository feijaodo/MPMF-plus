#!/usr/bin/env python3
"""
Thinking-Trace MPMF — stores reasoning paths, not facts.

Architecture parallel to athena.mpmf (fact store) but stores
LLM thinking/reasoning traces as first-class graph nodes.

Each thinking-trace node carries:
  - name: think_{session_id_hash}
  - text: The full reasoning trace
  - wiki: context, reasoning_steps, conclusion, outcome, key_concepts
  - edges: to related concept nodes and other thinking traces

Usage:
    import athena_thinking as atm
    atm.store_thinking(context, thinking_text, conclusion, concepts)
    atm.recall_thinking(problem_description, top_n=5)
"""

import sys, os, json, hashlib, time, threading

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

from mpmf.format import Plane
from mpmf.memory import MemoryManager, StringPool

from .paths import thinking_store

STORE_PATH = str(thinking_store())


def _txn(m):
    """write_txn on either a MemoryManager or an MPMFMemory wrapper."""
    t = getattr(m, "write_txn", None)
    return t() if t is not None else m.mem.write_txn()
VECTOR_DIM = 1024
LLM_MODEL = "deepseek/deepseek-v4.1-flash"

# ── Ollama embedding for vector search ──────────────────────────

def _embed(text: str) -> list:
    """Local embedding via Ollama (synchronous, lightweight)."""
    import requests
    r = requests.post("http://localhost:11434/api/embeddings",
                      json={"model": "mxbai-embed-large", "prompt": text},
                      timeout=30)
    return r.json().get("embedding", [0.0] * VECTOR_DIM)

def _cosine(a, b):
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0

def _thinking_matrix(m):
    """(ids, nodes, M, norms) for every thinking node that carries a vector.

    Item 2 (2026-10-12): the matrix is built ONCE and cached on the loaded
    MemoryManager instance (mirrors the memory store's mem._mpmf_vec_matrix),
    instead of re-scanning every node and re-collecting every vector on each
    recall. Invalidation is implicit and identical to the memory store: the
    read-snapshot cache (_open) is keyed on (path, mtime_ns, size) and is also
    dropped by _invalidate_open_cache(), which store_thinking() calls after
    every save — so the next _open() returns a FRESH instance with no cached
    matrix, forcing a rebuild that sees the new node.
    """
    cached = getattr(m, "_mpmf_thinking_matrix", None)
    if cached is not None:
        return cached
    ids = []
    nodes = []
    rows = []
    for nid in list(m._nodes):
        c = m._content.get(nid)
        if not (c and c.wiki and c.wiki.get("_ns") == "thinking"):
            continue
        vec = m.get_vector(nid)
        if vec is None:
            continue
        ids.append(nid)
        nodes.append(c)
        rows.append(vec)
    if _np is not None and rows:
        M = _np.asarray(rows, dtype=_np.float32)
        norms = _np.linalg.norm(M, axis=1)
    else:  # pragma: no cover - numpy fallback keeps row lists
        M, norms = rows, None
    out = (ids, nodes, M, norms)
    try:
        m._mpmf_thinking_matrix = out
    except Exception:
        pass
    return out

# ── Store management ───────────────────────────────────────────
# Read-snapshot cache keyed by (path, mtime_ns, size): recall reuses one loaded
# MemoryManager instead of re-loading the whole file every call. Writers take a
# private instance (_open_rw) so they never mutate the shared snapshot mid-read.

_OPEN_CACHE = {}
_OPEN_CACHE_LOCK = threading.Lock()

def _store_key(path):
    try:
        st = os.stat(path)
        return (path, st.st_mtime_ns, st.st_size)
    except OSError:
        return None

def _invalidate_open_cache():
    with _OPEN_CACHE_LOCK:
        _OPEN_CACHE.pop(STORE_PATH, None)

def _open():
    key = _store_key(STORE_PATH)
    if key is not None:
        with _OPEN_CACHE_LOCK:
            ent = _OPEN_CACHE.get(STORE_PATH)
            if ent is not None and ent[0] == key:
                return ent[1]
    mm = MemoryManager(STORE_PATH, vector_dimensions=VECTOR_DIM)
    mm.load()
    if key is not None:
        with _OPEN_CACHE_LOCK:
            _OPEN_CACHE[STORE_PATH] = (key, mm)
    return mm

def _open_rw():
    mm = MemoryManager(STORE_PATH, vector_dimensions=VECTOR_DIM)
    mm.load()
    return mm

def _vec_to_bytes(vec: list) -> bytes:
    import struct
    return struct.pack(f"<{VECTOR_DIM}f", *vec)

def _thin(text: str, max_len: int = 60) -> str:
    return text.strip()[:max_len].replace("\n", " ").strip()

# ── Main API ──────────────────────────────────────────────────


#: r83e: ONE canonical rule for "what string does a node's vector represent".
#: The writer, the audit and the daily health check all use this, so
#: cos(stored, embed(embed_text_of(text))) is exactly 1.0 by construction.
#: r83f: the canonical rule lives in the release core (mpmf.embed_text)
#: so writer, audit and daily check cannot drift apart again.
from mpmf.embed_text import EMBED_CHARS, embed_text as _embed_text  # noqa: E402


def store_thinking(context: str, thinking: str, conclusion: str,
                   concepts: list = None, session_id: str = "",
                   outcome: str = "unknown"):
    """Store a thinking trace node.

    Args:
        context: The problem/situation (what was being reasoned about)
        thinking: Full thinking trace text
        conclusion: What was concluded
        concepts: Key concept tags
        session_id: Source session ID
        outcome: Did the reasoning lead to a good outcome?
    """
    concepts = concepts or []
    
    # Unique name
    hash_key = hashlib.sha256(session_id.encode() if session_id else thinking.encode()).hexdigest()[:12]
    name = f"think_{hash_key}"
    
    wiki = {
        "_ns": "thinking",
        "context": _thin(context, 200),
        "reasoning_steps": _thin(thinking, 300),
        "conclusion": _thin(conclusion, 200),
        "outcome": outcome,
        "source_session": session_id,
        "concepts": json.dumps(concepts),
    }
    
    m = _open_rw()  # writer: private instance, never the shared read snapshot
    
    # Check for existing node with same name
    existing = m._name_index.get(name)
    if existing is not None:
        # Update wiki
        c = m._content.get(existing)
        if c:
            # r81: mutate under the lock — a bare periodic_save() here raced
            # other writers and raised StaleImageError, losing the update.
            with _txn(m):
                c2 = m._content.get(existing)
                if c2:
                    c2.wiki = wiki
            _invalidate_open_cache()
            return existing
    
    # r83e: the vector must be the embedding of EXACTLY the string we save as
    # text=thinking. It used to be _embed(context + " " + thinking[:512]) while
    # text was the whole trace, so every node's vector disagreed with its own
    # text (cos 0.87-0.95) and the store's vectors could not be re-derived.
    # The context stays in wiki["context"]; it is not part of the vector.
    vec = _embed(_embed_text(thinking))

    # Round 78: r69 guards are enforced by the release core — a create_node()
    # outside write_txn() raises instead of silently vanishing on reload.
    with _txn(m):
        nid = m._next_node_id
        m.create_node(
            name=name,
            node_type=0,  # CONCEPT
            plane_flags=1,  # ASSOCIATIVE
            text=thinking,
            summary=conclusion,
            wiki=wiki,
            vector=_vec_to_bytes(vec),
            importance=0.8,
        )
    
    # Create edges to related concept nodes (by name match in concepts)
    for concept in concepts:
        concept_lower = concept.lower()
        for other_nid in m.all_node_ids():
            if other_nid == nid:
                continue
            c = m._content.get(other_nid)
            if c and c.wiki and c.wiki.get("_ns") == "thinking":
                other_concepts = c.wiki.get("concepts", "[]")
                try:
                    if isinstance(other_concepts, str):
                        other_concepts = json.loads(other_concepts)
                except Exception:
                    other_concepts = []
                if concept_lower in [oc.lower() for oc in other_concepts]:
                    with _txn(m):
                        m.create_edge(nid, other_nid, plane=Plane.CAUSAL,
                                      relation_label="shares_concept", weight=0.6)
                        m.create_edge(other_nid, nid, plane=Plane.CAUSAL,
                                      relation_label="shares_concept", weight=0.6)
                    break
    
    # r81: no trailing save — create_node/create_edge each ran inside _txn(),
    # so the nodes are already durable. Saving again outside the lock raced
    # concurrent capture runs and raised StaleImageError after the fact.
    _invalidate_open_cache()
    return nid


def recall_thinking(query: str, top_n: int = 5, query_vec: list = None) -> list:
    """Retrieve thinking traces related to a problem context.

    Args:
        query: the problem context (used for the concept-overlap boost and,
               when query_vec is not supplied, to embed the query).
        top_n: max results.
        query_vec: item 2 (2026-10-12) — a pre-computed query vector to reuse
               instead of embedding here. The prefetch plugin computes ONE
               vector and shares it with memory recall, so this path does no
               second Ollama round-trip.

    Returns list of dicts with: name, context, thinking, conclusion,
    reasoning_steps, concepts, score.
    """
    m = _open()
    qvec = query_vec if query_vec is not None else _embed(query)

    # One cached numpy matmul over the prebuilt (N, dim) matrix — no per-call
    # node scan / vector collection (item 2). Falls back to per-row cosine when
    # numpy is unavailable.
    ids, nodes, M, norms = _thinking_matrix(m)
    if not ids:
        return []

    if _np is not None and norms is not None:
        q = _np.asarray(qvec, dtype=_np.float32)
        qn = float(_np.linalg.norm(q)) or 1.0
        sims = (M @ q) / (norms * qn + 1e-9)
    else:  # pragma: no cover
        sims = [_cosine(v, qvec) for v in M]

    ql = query.lower()
    scored = []
    for nid, c, s in zip(ids, nodes, sims):
        score = float(s)
        # Boost for concept overlap
        concept_text = c.wiki.get("concepts", "[]")
        try:
            concept_list = json.loads(concept_text) if isinstance(concept_text, str) else concept_text
        except Exception:
            concept_list = []
        for concept in concept_list:
            if concept.lower() in ql:
                score += 0.15
        scored.append((score, nid, c))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    results = []
    for score, nid, c in scored[:top_n]:
        wiki = c.wiki or {}
        results.append({
            "name": c.name,
            "node_id": nid,
            "context": wiki.get("context", ""),
            "thinking": c.text[:500] if c.text else "",
            "thinking_full": c.text or "",
            "conclusion": c.summary or wiki.get("conclusion", ""),
            "reasoning_steps": wiki.get("reasoning_steps", ""),
            "concepts": wiki.get("concepts", "[]"),
            "outcome": wiki.get("outcome", "unknown"),
            "score": round(score, 4),
        })
    
    return results


def get_stats() -> dict:
    m = _open()
    return {
        "nodes": m.header.node_count,
        "edges": m.header.edge_count,
    }


# ── CLI entry point ───────────────────────────────────────────
if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "stats":
        s = get_stats()
        print(f"Thinking store: {s['nodes']} nodes, {s['edges']} edges")
    elif len(_sys.argv) > 1 and _sys.argv[1] == "recall":
        results = recall_thinking(" ".join(_sys.argv[2:]), top_n=5)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['name']}")
            print(f"       Context: {r['context'][:80]}")
            print(f"       Conclusion: {r['conclusion'][:80]}")
            print()
    else:
        print("Usage:")
        print("  python3 athena_thinking.py stats")
        print("  python3 athena_thinking.py recall <query>")
