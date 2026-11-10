#!/usr/bin/env python3
"""P9: summarise rows_<mode>.json (judge means, paired counts). usage: p9_analyze.py <rows.json>..."""
import json, statistics as st, sys
from math import comb
rows = []
for p in sys.argv[1:]:
    rows += json.load(open(p))


def mean_j(r, arm, k):
    v = [j[k] for j in r["judge"][arm] if j]
    return st.mean(v) if v else None


def sign_p(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)


n = len(rows)
chg = [r for r in rows if r["changed"]]
print("turns", n, "retried", sum(1 for r in rows if r["kd"]["retried"]), "fixed-turns", len(chg),
      "facts_fixed", sum(r["kd"]["fixed"] for r in rows))
for k, lab in ((0, "specificity (1-5, 3 judges mean)"), (1, "unsupported facts per turn")):
    off = [mean_j(r, "off", k) for r in rows]; on = [mean_j(r, "on", k) for r in rows]
    pr = [(a, b) for a, b in zip(off, on) if a is not None and b is not None]
    up = sum(1 for a, b in pr if b > a); dn = sum(1 for a, b in pr if b < a)
    print("%-36s off %.3f  on %.3f  delta %+.3f   turns up %d / down %d / same %d  sign-p %.3f" % (
        lab, st.mean(a for a, _ in pr), st.mean(b for _, b in pr),
        st.mean(b - a for a, b in pr), up, dn, len(pr) - up - dn, sign_p(up, dn)))
    if chg:
        pc = [(mean_j(r, "off", k), mean_j(r, "on", k)) for r in chg]
        pc = [(a, b) for a, b in pc if a is not None and b is not None]
        print("   changed turns only (n=%d): off %.3f on %.3f delta %+.3f" % (
            len(pc), st.mean(a for a, _ in pc), st.mean(b for _, b in pc), st.mean(b - a for a, b in pc)))
print("specifics dropped (programmatic): off %d  on %d  (message had %d)" % (
    sum(r["dropped"]["off"] for r in rows), sum(r["dropped"]["on"] for r in rows),
    sum(r["specifics_in_msg"] for r in rows)))
print("facts written: off %d on %d" % (sum(len(r["facts"]["off"]) for r in rows), sum(len(r["facts"]["on"]) for r in rows)))
print("judge parse failures:", sum(1 for r in rows for a in ("off", "on") for j in r["judge"][a] if j is None))
