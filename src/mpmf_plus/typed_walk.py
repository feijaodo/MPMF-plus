"""Typed-link walk (P12, RESEARCH): one hop over IS_A / DERIVED_FROM / supersedes edges.

WHY
---
Plain cosine retrieval ranks nodes one by one. A question whose answer needs two linked
facts (a fact and the state built from it, a statement and the one it supersedes) often
retrieves only one of them. The store already holds typed edges between such nodes.

WHAT (flag ``typed_walk``, default OFF)
---------------------------------------
After the d20 arm has produced its final list, take the top ``typed_walk_seeds`` hits,
follow their edges whose label is IS_A, DERIVED_FROM or supersedes (either direction, ONE
hop), rank the neighbours by cosine to the query, and let the best ``typed_walk_k`` of them
(cosine >= ``typed_walk_min_cos``) take the LAST slots of the list. The list length is
unchanged (same render budget); seeds are never evicted; superseded-marked nodes, nodes
outside the requested namespaces are never injected, and a neighbour must be at least
``typed_walk_min_cos`` similar to the query (so a big category hub only comes in when the
query is about it).

Config (recall.config.json or ``MPMF_<KEY>``): typed_walk (false), typed_walk_k (2),
typed_walk_seeds (4), typed_walk_min_cos (0.3).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from . import paths as _paths

LABELS = frozenset({"IS_A", "DERIVED_FROM", "supersedes"})
MIN_TEXT = 20

#: info about the most recent call (read by the eval harness)
LAST: Dict[str, Any] = {}
_INDEX: Dict[str, Any] = {"key": None, "adj": None, "ms": 0.0}


def settings() -> Dict[str, Any]:
    c = _paths.load_config()
    return {
        "on": bool(c.get("typed_walk", False)),
        "k": max(1, int(c.get("typed_walk_k", 2) or 2)),
        "seeds": max(1, int(c.get("typed_walk_seeds", 4) or 4)),
        "min_cos": float(c.get("typed_walk_min_cos", 0.3) or 0.0),
    }


def enabled() -> bool:
    try:
        return bool(settings()["on"])
    except Exception:
        return False


def adjacency(mem) -> Dict[int, List[Tuple[int, str]]]:
    """node -> [(neighbour, label)] for the three labels, both directions. Cached per
    (store object, edge count): a write that adds edges rebuilds it."""
    edges = getattr(mem, "_edges", [])
    key = (id(mem), len(edges))
    if _INDEX["key"] == key and _INDEX["adj"] is not None:
        return _INDEX["adj"]
    t = time.perf_counter()
    labels = getattr(mem, "_edge_labels", {})
    adj: Dict[int, List[Tuple[int, str]]] = {}
    for e in edges:
        lab = labels.get(e.edge_id)
        if lab in LABELS:
            adj.setdefault(e.source_node_id, []).append((e.target_node_id, lab))
            adj.setdefault(e.target_node_id, []).append((e.source_node_id, lab))
    _INDEX.update({"key": key, "adj": adj, "ms": (time.perf_counter() - t) * 1000.0})
    return adj


def _cos(a, b) -> float:
    import numpy as np
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    return float(x @ y / ((np.linalg.norm(x) * np.linalg.norm(y)) + 1e-9))


def expand(mem, result: List[dict], qvec, namespaces, node_ns: Dict[int, str]) -> List[dict]:
    """Return `result` with up to typed_walk_k tail slots replaced by 1-hop typed
    neighbours of the top hits. Same length in, same length out. Never raises."""
    LAST.clear()
    t0 = time.perf_counter()
    try:
        s = settings()
        if not s["on"] or not result or qvec is None:
            return result
        adj = adjacency(mem)
        have = {int(r["node_id"]) for r in result}
        seeds = [int(r["node_id"]) for r in result[: s["seeds"]]]
        cand: Dict[int, Tuple[str, int]] = {}
        for sd in seeds:
            for nb, lab in adj.get(sd, ()):
                nb = int(nb)
                if nb in have or nb in cand:
                    continue
                cand[nb] = (lab, sd)
        scored = []
        for nb, (lab, sd) in cand.items():
            if node_ns.get(nb, "memory") not in namespaces:
                continue
            w = mem.get_wiki(nb) or {}
            if w.get("_superseded_by"):
                continue
            text = mem.get_text(nb) or mem.get_summary(nb) or ""
            if len(text.strip()) < MIN_TEXT:
                continue
            v = mem.get_vector(nb)
            if v is None:
                continue
            c = _cos(qvec, v)
            if c >= s["min_cos"]:
                scored.append((c, nb, lab, sd, text))
        scored.sort(key=lambda x: -x[0])
        room = max(0, len(result) - len(seeds))
        pick = scored[: min(s["k"], room)]
        LAST.update({"n_cand": len(cand), "n_scored": len(scored), "added": len(pick),
                     "labels": [p[2] for p in pick], "ids": [p[1] for p in pick],
                     "index_ms": _INDEX["ms"]})
        if not pick:
            return result
        out = [dict(r) for r in result[: len(result) - len(pick)]]
        for c, nb, lab, sd, text in pick:
            out.append({"rank": len(out) + 1, "node_id": nb, "text": text,
                        "summary": mem.get_summary(nb), "name": mem.get_name(nb),
                        "wiki": mem.get_wiki(nb), "_ns": node_ns.get(nb, "memory"),
                        "score": 0.0, "via": lab, "via_seed": sd, "walk_cos": c})
        for i, r in enumerate(out, start=1):
            r["rank"] = i
        return out
    except Exception as e:                      # a walk failure must never break recall
        LAST["error"] = repr(e)[:200]
        return result
    finally:
        LAST["ms"] = (time.perf_counter() - t0) * 1000.0
