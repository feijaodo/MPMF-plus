#!/usr/bin/env python3
"""r92 evidence statistics (stdlib only). Reproduces the 0.3.0 README numbers.

  python3 r92_stats.py [--prefetch-log PATH] [--since 2026-11-06] > r92_stats.json

 * Wilson score interval (95%, z=1.96) for the strict-correct proportion of the
   order questions, history_mode off vs on (counts from r92_order_history.json).
   Wilson is used instead of the normal approximation because the 'off' count is 0.
 * Live prefetch latency: p50/p90/max of prefetch_ms from prefetch.log, real sessions
   only (session ids starting r85 / cron / e2e / harness are excluded), nearest-rank
   percentiles. Aggregates only; the log itself is not shipped.
"""
import argparse, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def pct(sorted_v, q):                       # nearest-rank
    return sorted_v[max(0, math.ceil(q * len(sorted_v)) - 1)] if sorted_v else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefetch-log", default=os.path.expanduser("~/.mpmf/health/prefetch.log"))
    ap.add_argument("--since", default="2026-11-06")
    a = ap.parse_args()
    ev = json.load(open(os.path.join(HERE, "r92_order_history.json")))
    out = {"method": "Wilson score interval, 95% (z=1.96)", "order_strict": {}}
    for arm in ("off", "on"):
        k, n = ev[arm]["correct"], ev[arm]["n"]
        lo, hi = wilson(k, n)
        out["order_strict"][arm] = {"correct": k, "n": n, "p": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    v = []
    try:
        for line in open(a.prefetch_log):
            m = re.match(r"^(\S+) prefetch_ms=(\d+) session=(\S+)", line)
            if m and m.group(1)[:10] >= a.since and not m.group(3).startswith(("r85", "cron", "e2e", "harness")):
                v.append(int(m.group(2)))
    except FileNotFoundError:
        out["prefetch_live"] = {"error": "log not found: %s" % a.prefetch_log}
        v = []
    if v:
        v.sort()
        out["prefetch_live"] = {"since": a.since, "n": len(v), "p50_ms": pct(v, .5), "p90_ms": pct(v, .9),
                                "max_ms": v[-1], "percentile": "nearest-rank", "sessions": "real only"}
    json.dump(out, sys.stdout, indent=1); print()

if __name__ == "__main__":
    main()
