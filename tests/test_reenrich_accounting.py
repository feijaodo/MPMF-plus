#!/usr/bin/env python3
"""r92 step 4: the re-enrich line must account for every pending node (497 -> 50 + 447)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mpmf_plus.memory import sweep_summary
cases = [
    ("497 pending, limit 50, all attempted enriched", {"pending": 497, "enriched": 50, "still_pending": 0, "deferred": 447}, 0, "447 deferred"),
    ("quiet day: nothing pending", {"pending": 0, "enriched": 0, "still_pending": 0, "deferred": 0}, 0, ""),
    ("2 attempted failed", {"pending": 10, "enriched": 6, "still_pending": 2, "deferred": 2}, 1, "2 failed"),
    ("sums must close", {"pending": 10, "enriched": 3, "still_pending": 0, "deferred": 0}, 1, "accounting error"),
]
bad = []
for name, r, rc_want, frag in cases:
    text, rc = sweep_summary(r)
    ok = rc == rc_want and frag in text
    print("[%s] %s -> rc=%d %r" % ("ok" if ok else "FAIL", name, rc, text.splitlines()[0] if text else ""))
    if not ok: bad.append(name)
print("FAIL: %s" % bad if bad else "OK test_reenrich_accounting")
sys.exit(1 if bad else 0)
