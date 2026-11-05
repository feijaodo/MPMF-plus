"""R85 nightly scorecard + drift alert. Silent unless HOLDOUT really dropped.

Runs scorecard.py on TUNE+HOLDOUT, archives the per-question rows, then compares
HOLDOUT against the previous run: alert only if accuracy fell by >= 8 points AND
a McNemar test on the paired questions says it is not noise (p < 0.10).

no_agent cron: empty stdout = no message to the user.
"""
import json
import os
from .. import paths as _paths
_BASE = os.environ.get('MPMF_EVAL_DIR') or str(_paths.home() / "eval")
import math
import os
import shutil
import subprocess
import sys
import time

BASE = _BASE
RUNS = os.path.join(BASE, "runs")
SC = os.path.join(BASE, "scorecard.jsonl")
BUDGET = 0.10
DROP_PTS = 0.08
P_MAX = 0.10


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def acc(rows):
    n = len(rows)
    return (sum(1 for r in rows if r["verdict"] == "correct")
            + 0.5 * sum(1 for r in rows if r["verdict"] == "partly")) / max(1, n)


def main():
    os.makedirs(RUNS, exist_ok=True)
    prev_files = sorted(f for f in os.listdir(RUNS) if f.endswith(".jsonl"))[-1:]
    prev = None
    if prev_files:
        prev = [json.loads(l) for l in open(os.path.join(RUNS, prev_files[-1]))]

    r = subprocess.run([sys.executable, "-m", "mpmf_plus.eval", "scorecard",
                        "--split", "both", "--budget-usd", str(BUDGET)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("SCORECARD FAILED rc=%d" % r.returncode)
        print((r.stderr or "")[-500:])
        return

    rows = [json.loads(l) for l in open(SC)]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    shutil.copy2(SC, os.path.join(RUNS, "scorecard_%s.jsonl" % stamp))
    summ = json.load(open(os.path.join(BASE, "scorecard_summary.json")))
    cur_h = [x for x in rows if x["split"] == "holdout"]
    line = ("scorecard %s: TUNE acc %.3f (n=%d) | HOLDOUT acc %.3f (n=%d) | p90 %.0f ms | $%.3f"
            % (stamp,
               summ["splits"]["tune"]["accuracy"], summ["splits"]["tune"]["n"],
               summ["splits"]["holdout"]["accuracy"], summ["splits"]["holdout"]["n"],
               summ["splits"]["holdout"]["prefetch_p90_ms"] or 0, summ["cost_usd"]))
    open(os.path.join(BASE, "scorecard_last.txt"), "w").write(line + "\n")

    # --- r88c: auto-revert the reranker if HOLDOUT drops vs the pre-change baseline ---
    try:
        bl = json.load(open(os.path.join(BASE, "rerank_baseline.json")))
        base_rows = bl.get("rows") or {}
        cmap = {x["id"]: x["verdict"] for x in cur_h}
        common_b = [i for i in cmap if i in base_rows]
        if len(common_b) >= 32:
            a_base = acc([{"verdict": base_rows[i]} for i in common_b])
            a_cur = acc([{"verdict": cmap[i]} for i in common_b])
            ok_v = lambda v: v == "correct"
            b = sum(1 for i in common_b if ok_v(cmap[i]) and not ok_v(base_rows[i]))
            c = sum(1 for i in common_b if not ok_v(cmap[i]) and ok_v(base_rows[i]))
            p_b = mcnemar(b, c)
            drop_b = a_base - a_cur
            if drop_b >= 0.054 and p_b < 0.10:
                cfg_path = str(_paths.config_path())
                cfg = json.load(open(cfg_path))
                if cfg.get("rerank_skip_after_ms") == 0:
                    cfg["rerank_skip_after_ms"] = 4500
                    tmp = cfg_path + ".tmp"
                    with open(tmp, "w") as fh:
                        json.dump(cfg, fh, indent=1)
                    os.replace(tmp, cfg_path)
                    print("RERANK AUTO-REVERTED to 4500: HOLDOUT %.3f vs baseline %.3f "
                          "(-%.1f pts), McNemar p=%.3f" % (a_cur, a_base, drop_b * 100, p_b))
    except Exception:
        pass

    if not prev:
        return                      # first run: nothing to compare, stay silent
    prev_h = {x["id"]: x["verdict"] for x in prev if x.get("split") == "holdout"}
    cur_map = {x["id"]: x["verdict"] for x in cur_h}
    common = [i for i in cur_map if i in prev_h]
    if not common:
        return
    a_prev = acc([{"verdict": prev_h[i]} for i in common])
    a_cur = acc([{"verdict": cur_map[i]} for i in common])
    ok = lambda v: v == "correct"
    b = sum(1 for i in common if ok(cur_map[i]) and not ok(prev_h[i]))
    c = sum(1 for i in common if not ok(cur_map[i]) and ok(prev_h[i]))
    p = mcnemar(b, c)
    drop = a_prev - a_cur
    if drop >= DROP_PTS and p < P_MAX:
        print("HOLDOUT DROP: %.3f -> %.3f (-%.0f pts) over %d paired questions, McNemar p=%.3f (b=%d c=%d)"
              % (a_prev, a_cur, 100 * drop, len(common), p, b, c))
        print(line)
        worst = [x for x in cur_h if x["id"] in common and cur_map[x["id"]] in ("wrong", "declined")][:3]
        for w in worst:
            print("  [%s] %s -> %s" % (w["id"], (w.get("q") or "")[:70], w["verdict"]))


if __name__ == "__main__":
    main()
