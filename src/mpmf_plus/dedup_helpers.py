# --- A2: Store-level dedup helpers (per MPMF+ v1.0 spec §A2) ---
# Implements:
#  A2.1  Exact-text short-circuit: normalize (strip/collapse whitespace), hash;
#        if a node with the same text-hash exists in the same namespace →
#        update its `modified_at`, return existing id.
#  A2.2  Near-dup check: embed, compare against existing vectors in the same
#        namespace; if cosine ≥ 0.95, merge into the existing node (update
#        timestamp, keep higher importance) instead of creating.
#  A2.3  Log every dedup action (kept_id, skipped_text_hash) to the store's
#        audit log.
import hashlib
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

# Per spec A2.2 — the long-standing rule: err conservative. Wrongly merging
# distinct facts is unrecoverable; a surviving near-dup is merely untidy.
DEDUP_COSINE_THRESHOLD = 0.95


def audit_log_path() -> Path:
    """Where dedup decisions are appended: ``$MPMF_HOME/health/dedup_audit.log``.

    r60c: this used to be the hardcoded ``/root/.mpmf/health/dedup_audit.log``,
    so a scratch store in a test or a second host wrote its audit trail into the
    author's live health directory (observed during the r60b stress run, where
    the scratch probes landed in the live log). Resolved from ``$MPMF_HOME`` on
    every call, exactly like the store path.
    """
    home = Path(os.environ.get("MPMF_HOME") or "~/.mpmf").expanduser()
    return home / "health" / "dedup_audit.log"


#: Deprecated alias kept for importers; resolved lazily via audit_log_path().
DEDUP_AUDIT_LOG = "dedup_audit.log"


def _normalize_for_hash(text: str) -> str:
    return " ".join((text or "").split())


def _text_hash(text: str) -> str:
    return hashlib.sha256(_normalize_for_hash(text).encode("utf-8")).hexdigest()[:16]


def _audit_dedup(action: str, kept_id, skipped_hash: str, namespace: str = "memory",
                 score: Optional[float] = None) -> None:
    """Append one line to the store's audit log. Best-effort, never raises."""
    try:
        log_path = audit_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            parts = [
                datetime.now(timezone.utc).isoformat(),
                f"action={action}",
                f"ns={namespace}",
                f"kept_id={kept_id}",
                f"skipped_hash={skipped_hash}",
            ]
            if score is not None:
                parts.append(f"score={score:.4f}")
            f.write("\t".join(parts) + "\n")
    except Exception as e:
        # Audit failure must never block a write. Log to stderr.
        print(f"[dedup-audit] write failed: {e}", file=sys.stderr)


try:  # numpy fast path: one matrix-vector product over the store's float32 matrix
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

#: Per-row candidate masks (namespace + not-superseded), cached per store version.
#: Building them is one O(N) wiki walk; the cosine scan then costs one matvec.
_NEAR_CACHE = {"key": None, "ns": None, "keep": None}


def _candidate_masks(mem, ids, version, namespace):
    key = (id(mem), version, namespace)
    if _NEAR_CACHE["key"] == key and _NEAR_CACHE["ns"] is not None:
        return _NEAR_CACHE["ns"], _NEAR_CACHE["keep"]
    n = len(ids)
    ns = _np.zeros(n, dtype=bool)
    keep = _np.ones(n, dtype=bool)
    for i, nid in enumerate(ids):
        w = mem.get_wiki(nid) or {}
        if w.get("_ns", "memory") == namespace:
            ns[i] = True
        if w.get("_superseded_by"):
            keep[i] = False            # never merge into an audit-trail row
    _NEAR_CACHE.update({"key": key, "ns": ns, "keep": keep})
    return ns, keep


def _cosine(a, b) -> float:
    """Cosine similarity between two equal-length float vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def check_dedup(text: str, namespace: str, mem, embed_fn) -> Tuple[str, Optional[int]]:
    """Decide whether `text` is a duplicate of an existing node in `mem`.

    Returns a tuple (action, kept_node_id):
      ('new', None)              — proceed with normal create
      ('exact_match', node_id)   — exact text-hash match found, will be returned
      ('near_dup',   node_id)    — cosine ≥ threshold to an existing node

    Per spec A2.3, every dedup action is audit-logged.

    `mem` is the underlying mpmf memory object (has get_text/summary/wiki/vector).
    `embed_fn(text, dim)` produces an embedding vector for `text` — passed in
    so we use the canonical embedder, not duplicate embed paths.

    Conservative: returns ('new', None) on ANY uncertainty (missing vector,
    missing wiki, error reading existing node). Better a duplicate survives
    than a distinct fact gets wrongly merged.

    Performance: A2.1 exact hash check runs BEFORE embedding — exact matches
    short-circuit and avoid an unnecessary LLM embed call.
    """
    new_hash = _text_hash(text)

    # A2.1: Walk all nodes, check exact-text hash FIRST (no embed needed).
    try:
        nids = list(mem.all_node_ids() if hasattr(mem, "all_node_ids")
                    else range(mem.header.node_count))
    except Exception:
        nids = []

    for nid in nids:
        try:
            existing_text = mem.get_text(nid) or mem.get_summary(nid) or ""
            existing_wiki = mem.get_wiki(nid) or {}
            existing_ns = existing_wiki.get("_ns", "memory")
        except Exception:
            continue

        if existing_ns != namespace:
            continue

        # A2.1 exact-text short-circuit — no embed needed for this path
        if _text_hash(existing_text) == new_hash:
            kept_id = mem.get_name(nid) if nid is not None else "?"
            _audit_dedup(action="exact_match", kept_id=kept_id, skipped_hash=new_hash,
                         namespace=namespace, score=1.0)
            return ("exact_match", nid)

    # A2.2 near-dup: embed the new text only after exact-match check passed
    new_vec = None
    try:
        new_vec = embed_fn(text, dim=1024)
    except Exception as e:
        _audit_dedup(action="dedup_skipped_embed_failed", kept_id=None,
                     skipped_hash=new_hash, namespace=namespace, score=None)
        return ("new", None)

    best_near: Tuple[int, float] = (None, -1.0)

    # Round 78 fast path: the store already holds every vector in one float32
    # matrix, so the near-dup scan is ONE matrix-vector product plus two masks,
    # not a Python cosine per node (measured 4.5 s per write on the live store).
    vector_rows = getattr(mem, "vector_rows", None)
    used_matrix = False
    if _np is not None and vector_rows is not None:
        try:
            ids, M, norms, version = vector_rows()
            if len(ids):
                q = _np.asarray(new_vec, dtype=_np.float32)
                qn = float(_np.linalg.norm(q)) or 1.0
                sims = (M @ q) / (norms * qn + 1e-9)
                ns_mask, keep = _candidate_masks(mem, ids, version, namespace)
                sims = _np.where(ns_mask & keep, sims, -1.0)
                best = int(_np.argmax(sims))
                score = float(sims[best])
                if score >= DEDUP_COSINE_THRESHOLD:
                    best_near = (int(ids[best]), score)
                used_matrix = True
        except Exception as e:      # any doubt -> conservative Python path
            print(f"[dedup] matrix path failed, falling back: {e}", file=sys.stderr)

    if not used_matrix:
        for nid in nids:
            try:
                existing_vec = mem.get_vector(nid)
                if existing_vec is None:
                    continue
                existing_wiki = mem.get_wiki(nid) or {}
                existing_ns = existing_wiki.get("_ns", "memory")
                if existing_ns != namespace:
                    continue
                score = _cosine(new_vec, list(existing_vec))
            except Exception:
                continue

            if score >= DEDUP_COSINE_THRESHOLD and score > best_near[1]:
                best_near = (nid, score)

    if best_near[0] is not None:
        kept_id = mem.get_name(best_near[0]) if best_near[0] is not None else "?"
        _audit_dedup(action="near_dup", kept_id=kept_id, skipped_hash=new_hash,
                     namespace=namespace, score=best_near[1])
        return ("near_dup", best_near[0])

    return ("new", None)
