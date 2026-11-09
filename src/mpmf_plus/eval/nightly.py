"""R85 nightly scorecard + drift alert. Silent unless HOLDOUT really dropped.

Runs scorecard.py on TUNE+HOLDOUT, archives the per-question rows, then compares
HOLDOUT against the previous run: alert only if accuracy fell by >= 8 points AND
a McNemar test on the paired questions says it is not noise (p < 0.10).

r91a fixes:
  * ``--split``/``--budget-usd`` are accepted here and forwarded to the eval CLI
    (before, the eval CLI's subparsers had no flags and the wrapper died rc=2).
  * ``--dry-run`` validates the exact subprocess command and the presence of
    both auto-revert baselines, without spending a cent.
  * both auto-revert checks now *log their verdict* to
    ``$MPMF_EVAL_DIR/autorevert_last.txt`` whether they fire or not, so
    "did the safety net run?" is answerable instead of a silent no-op.

no_agent cron: empty stdout = no message to the user.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import time

from .. import paths as _paths

BASE = os.environ.get("MPMF_EVAL_DIR") or str(_paths.home() / "eval")
RUNS = os.path.join(BASE, "runs")
SC = os.path.join(BASE, "scorecard.jsonl")
BUDGET = 0.10
DROP_PTS = 0.08
P_MAX = 0.10
AR_LOG = os.path.join(BASE, "autorevert_last.txt")


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


def _log_ar(name, msg):
    """Record the verdict of one auto-revert check (fired or not)."""
    try:
        with open(AR_LOG, "a") as fh:
            fh.write("%s %-8s %s\n" % (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), name, msg))
    except Exception:
        pass


def _scorecard_args(budget, split):
    return ["-m", "mpmf_plus.eval", "scorecard", "--split", split,
            "--budget-usd", str(budget)]


def _check_baselines():
    out = {}
    for name in ("rerank_baseline.json", "history_baseline.json"):
        p = os.path.join(BASE, name)
        try:
            bl = json.load(open(p))
            rows = bl.get("rows") or {}
            out[name] = (len(rows), bl.get("acc"), bl.get("ts"))
        except Exception as exc:
            out[name] = ("MISSING", repr(exc), None)
    return out


MIN_PAIRS = 32
REVERT_DROP = 0.054


def _pair(base_rows, cur_h):
    """Pair scorecard rows with baseline rows by question id.

    r92 step 2: ids are compared as STRINGS. The baselines are JSON objects (keys
    are always strings) while scorecard rows carry integer ids, so the old
    ``i in base_rows`` matched 0 of 64 and both checks were silently SKIPPED.
    Returns ({id: current_verdict}, {id: baseline_verdict}, [paired ids]).
    """
    cmap = {str(x["id"]): x["verdict"] for x in cur_h}
    bmap = {str(k): v for k, v in (base_rows or {}).items()}
    return cmap, bmap, [i for i in cmap if i in bmap]


def _check(name, base_file, cur_h, revert_fn):
    """One auto-revert check. Returns (rc, message). rc 2 = could not run."""
    try:
        bl = json.load(open(os.path.join(BASE, base_file)))
    except Exception as exc:
        msg = "%s: baseline unreadable (%r) -> CHECK NOT RUN" % (name, exc)
        _log_ar(name, msg)
        return 2, "AUTOREVERT ALERT " + msg
    cmap, bmap, common = _pair(bl.get("rows"), cur_h)
    if len(common) < MIN_PAIRS:
        msg = ("n=%d < %d paired (holdout rows %d, baseline rows %d) -> CHECK NOT RUN"
               % (len(common), MIN_PAIRS, len(cmap), len(bmap)))
        _log_ar(name, msg)
        return 2, "AUTOREVERT ALERT %s: %s" % (name, msg)
    a_base = acc([{"verdict": bmap[i]} for i in common])
    a_cur = acc([{"verdict": cmap[i]} for i in common])
    ok = lambda v: v == "correct"
    b = sum(1 for i in common if ok(cmap[i]) and not ok(bmap[i]))
    c = sum(1 for i in common if not ok(cmap[i]) and ok(bmap[i]))
    p = mcnemar(b, c)
    drop = a_base - a_cur
    fire = drop >= REVERT_DROP and p < 0.10
    _log_ar(name, "n=%d base=%.3f cur=%.3f drop=%+.1fpts p=%.3f -> %s"
            % (len(common), a_base, a_cur, drop * 100, p, "REVERT" if fire else "keep"))
    if not fire:
        return 0, ""
    done = revert_fn(a_cur, a_base, drop, p, b, c)
    return 0, done or ""


def _revert_rerank(a_cur, a_base, drop, p, b, c):
    cfg_path = str(_paths.config_path())
    cfg = json.load(open(cfg_path))
    if cfg.get("rerank_skip_after_ms") != 0:
        return ""
    cfg["rerank_skip_after_ms"] = 4500
    _write_cfg(cfg_path, cfg)
    return ("RERANK AUTO-REVERTED to 4500: HOLDOUT %.3f vs baseline %.3f (-%.1f pts), McNemar p=%.3f"
            % (a_cur, a_base, drop * 100, p))


def _revert_history(a_cur, a_base, drop, p, b, c):
    cfg_path = str(_paths.config_path())
    cfg = json.load(open(cfg_path))
    if "history_mode" not in cfg:
        return ""
    cfg.pop("history_mode", None)
    _write_cfg(cfg_path, cfg)
    row = ("| %s | history_mode auto-revert | `recall.config.json` -> key removed | "
           "HOLDOUT %.1f%% vs baseline %.1f%% (-%.1f pts), McNemar p=%.3f | nightly |\n"
           % (time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()), a_cur * 100, a_base * 100, drop * 100, p))
    with open(os.path.join(_paths.health_dir(), "PAUSED.md"), "a") as fh:
        fh.write(row)
    return ("HISTORY AUTO-REVERTED: HOLDOUT %.3f vs baseline %.3f (-%.1f pts), McNemar p=%.3f (b=%d c=%d)"
            % (a_cur, a_base, drop * 100, p, b, c))


def _write_cfg(cfg_path, cfg):
    tmp = cfg_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(cfg, fh, indent=1)
    os.replace(tmp, cfg_path)


def autorevert(cur_h):
    """Run both checks; print every alert/revert. Returns 0, or 2 if either check
    could not pair >= MIN_PAIRS questions (the cron then exits non-zero and alerts)."""
    rc = 0
    for name, base_file, fn in (("reranker", "rerank_baseline.json", _revert_rerank),
                                ("history", "history_baseline.json", _revert_history)):
        try:
            r, msg = _check(name, base_file, cur_h, fn)
        except Exception as exc:
            _log_ar(name, "CHECK ERROR %r" % (exc,))
            r, msg = 2, "AUTOREVERT ALERT %s: CHECK ERROR %r" % (name, exc)
        if msg:
            print(msg)
        rc = max(rc, r)
    return rc


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="mpmf-eval nightly")
    ap.add_argument("--split", default="both", choices=["tune", "holdout", "both"])
    ap.add_argument("--budget-usd", type=float, default=BUDGET)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    args = _scorecard_args(a.budget_usd, a.split)

    if a.dry_run:
        # Validate the exact command the wrapper would run, argv parsing included,
        # but do not call any model and do not touch the store.
        from . import scorecard
        parsed = scorecard.build_parser().parse_args(args[3:])
        print("DRY-RUN nightly: cmd=%s" % " ".join(args))
        print("DRY-RUN nightly: parsed split=%s budget_usd=%.3f" % (parsed.split, parsed.budget_usd))
        for name, (n, acc_, ts) in _check_baselines().items():
            print("DRY-RUN nightly: %s n=%s acc=%s ts=%s" % (name, n, acc_, ts))
        rc = 0
        if os.path.isfile(SC):
            cur_h = [x for x in (json.loads(l) for l in open(SC)) if x.get("split") == "holdout"]
            for name, f in (("reranker", "rerank_baseline.json"), ("history", "history_baseline.json")):
                try:
                    rows = json.load(open(os.path.join(BASE, f))).get("rows")
                except Exception:
                    rows = {}
                _, bm, common = _pair(rows, cur_h)
                print("DRY-RUN nightly: pairing %s: %d of %d holdout rows paired (baseline rows %d)"
                      % (name, len(common), len(cur_h), len(bm)))
                if len(common) < MIN_PAIRS:
                    print("DRY-RUN nightly FAILED: %s pairs %d < %d" % (name, len(common), MIN_PAIRS))
                    rc = 2
        else:
            print("DRY-RUN nightly: pairing not checked (no scorecard.jsonl in %s)" % BASE)
        if rc == 0:
            print("DRY-RUN nightly OK")
        return rc

    os.makedirs(RUNS, exist_ok=True)
    prev_files = sorted(f for f in os.listdir(RUNS) if f.endswith(".jsonl"))[-1:]
    prev = None
    if prev_files:
        prev = [json.loads(l) for l in open(os.path.join(RUNS, prev_files[-1]))]

    try:
        os.remove(AR_LOG)
    except OSError:
        pass

    r = subprocess.run([sys.executable] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print("SCORECARD FAILED rc=%d" % r.returncode)
        print((r.stderr or "")[-500:])
        return 1

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

    rc = autorevert(cur_h)

    if not prev:
        return rc                   # first run: nothing to compare, stay silent
    prev_h = {x["id"]: x["verdict"] for x in prev if x.get("split") == "holdout"}
    cur_map = {x["id"]: x["verdict"] for x in cur_h}
    common = [i for i in cur_map if i in prev_h]
    if not common:
        return rc
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
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
