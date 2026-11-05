"""Evaluation helpers: frozen-question dry runs, accuracy, McNemar.

Moved out of the r85 scorecard/nightly scripts. No host paths; caller passes
the question file and store. Judging (LLM) stays in the caller — this module
only does the deterministic parts plus prefetch timing.
"""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def load_questions(path) -> List[Dict[str, Any]]:
    p = Path(path)
    with open(p) as fh:
        if p.suffix == ".jsonl":
            return [json.loads(l) for l in fh if l.strip()]
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("questions") or data.get("rows") or []
    return list(data)


def acc(rows: Iterable[Dict[str, Any]]) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    return (sum(1 for r in rows if r.get("verdict") == "correct")
            + 0.5 * sum(1 for r in rows if r.get("verdict") == "partly")) / len(rows)


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def flips(base: Dict[Any, str], cur: Dict[Any, str]):
    """(b, c) where b = base wrong -> cur correct."""
    ok = lambda v: v == "correct"
    common = [i for i in cur if i in base]
    b = sum(1 for i in common if ok(cur[i]) and not ok(base[i]))
    c = sum(1 for i in common if not ok(cur[i]) and ok(base[i]))
    return b, c, common


def prefetch_ids(query: str, top_k: int = 8, use_rerank: bool = True) -> List[int]:
    from . import provider
    res = provider.recall(query, top_n=top_k, use_rerank=use_rerank) or []
    out = []
    for r in res:
        nid = r.get("node_id") if isinstance(r, dict) else None
        if nid is not None:
            out.append(nid)
    return out


def dry_run(questions: Sequence[Dict[str, Any]], top_k: int = 8,
            limit: Optional[int] = None) -> Dict[str, Any]:
    """Deterministic: ids + timings only, no LLM, no writes."""
    qs = list(questions)[:limit] if limit else list(questions)
    ids, times = [], []
    for q in qs:
        text = q.get("q") or q.get("question") or ""
        t = time.perf_counter()
        ids.append(prefetch_ids(text, top_k=top_k))
        times.append((time.perf_counter() - t) * 1000.0)
    times.sort()
    return {
        "n": len(qs),
        "top_k": top_k,
        "ids": ids,
        "p50_ms": round(statistics.median(times), 1) if times else 0.0,
        "p90_ms": round(times[int(0.9 * len(times)) - 1], 1) if times else 0.0,
        "ids_sha": __import__("hashlib").sha256(
            json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
    }
