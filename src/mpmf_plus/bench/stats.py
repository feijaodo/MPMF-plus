"""b0 statistics: Wilson interval, exact McNemar, paired sign-flip permutation test, bootstrap CI.

Pure python + numpy. Every random procedure takes a fixed seed so a report can be reproduced.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PERMUTATIONS = 10000
SEED = 20261114


def wilson(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p from the discordant counts (binomial, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def paired_binary(a: Dict[str, Optional[int]], b: Dict[str, Optional[int]]) -> dict:
    """McNemar on ids scored (non-null) in both dicts. b_ = first fails & second passes ... see keys.

    ``b`` = first arm wrong, second arm right; ``c`` = first right, second wrong.
    """
    ids = [i for i in a if i in b and a[i] is not None and b[i] is not None]
    bb = sum(1 for i in ids if not a[i] and b[i])
    cc = sum(1 for i in ids if a[i] and not b[i])
    return {"test": "McNemar exact (two-sided)", "n": len(ids), "b": bb, "c": cc,
            "both_right": sum(1 for i in ids if a[i] and b[i]),
            "both_wrong": sum(1 for i in ids if not a[i] and not b[i]),
            "p": round(mcnemar_exact(bb, cc), 6)}


def signflip_p(diffs: Sequence[float], n_perm: int = PERMUTATIONS, seed: int = SEED) -> float:
    """Paired sign-flip permutation test, two-sided, on the mean of per-question differences."""
    import numpy as np
    d = np.asarray([x for x in diffs], dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0 or not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice((-1.0, 1.0), size=(n_perm, d.size))
    stat = np.abs((signs * d).mean(axis=1))
    return float((np.sum(stat >= obs - 1e-12) + 1) / (n_perm + 1))


def paired_scores(a: Dict[str, Optional[float]], b: Dict[str, Optional[float]]) -> dict:
    ids = [i for i in a if i in b and a[i] is not None and b[i] is not None]
    d = [b[i] - a[i] for i in ids]
    mean_d = (sum(d) / len(d)) if d else 0.0
    return {"test": "paired sign-flip permutation (%d perms, seed %d)" % (PERMUTATIONS, SEED),
            "n": len(ids), "mean_diff": round(mean_d, 6),
            "wins": sum(1 for x in d if x > 1e-12), "losses": sum(1 for x in d if x < -1e-12),
            "p": round(signflip_p(d), 6)}


def bootstrap_ci(x: Sequence[float], n_boot: int = PERMUTATIONS, seed: int = SEED) -> Tuple[float, float]:
    """95% percentile bootstrap interval on the mean."""
    import numpy as np
    a = np.asarray(list(x), dtype=float)
    if a.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    m = a[idx].mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def mean(x: Iterable[float]) -> float:
    x = list(x)
    return sum(x) / len(x) if x else 0.0
