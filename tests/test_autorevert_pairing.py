#!/usr/bin/env python3
"""r92 step 2: the nightly auto-revert must really pair, really revert, and never skip quietly.

  * int scorecard ids pair with the string keys of the JSON baselines (64 of 64);
  * a simulated 10-point HOLDOUT drop produces REVERT and rewrites the config;
  * a simulated pairing failure (< 32 pairs) exits non-zero with an alert.
Runs against a throwaway MPMF_HOME; touches nothing else.
"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

def run():
    tmp = tempfile.mkdtemp(prefix="mpmf-ar-")
    os.environ.update(MPMF_HOME=tmp, MPMF_EVAL_DIR=tmp, MPMF_HEALTH_DIR=tmp,
                      MPMF_RECALL_CONFIG=os.path.join(tmp, "recall.config.json"))
    from mpmf_plus.eval import nightly as N
    N.BASE = tmp; N.AR_LOG = os.path.join(tmp, "autorevert_last.txt")
    base = {str(i): ("correct" if i < 30 else "declined") for i in range(1, 65)}
    for f in ("rerank_baseline.json", "history_baseline.json"):
        json.dump({"acc": 30 / 64, "n": 64, "rows": base}, open(os.path.join(tmp, f), "w"))
    def cfg():
        json.dump({"rerank_skip_after_ms": 0, "history_mode": True}, open(os.environ["MPMF_RECALL_CONFIG"], "w"))
    def cur(drop_n=0, ids=range(1, 65)):
        return [{"id": i, "split": "holdout",
                 "verdict": "wrong" if (i <= drop_n) else base.get(str(i), "correct")} for i in ids]  # int ids, like scorecard.jsonl
    fails = []
    # 1. pairing: ints vs str keys -> 64/64, no drop -> keep, rc 0
    cfg(); open(N.AR_LOG, "w").close()
    rc = N.autorevert(cur(0)); log = open(N.AR_LOG).read()
    print("[pairing]", log.strip().replace("\n", " | "), "rc=%d" % rc)
    if rc != 0 or log.count("n=64 ") != 2 or "keep" not in log: fails.append("pairing 64/64 keep")
    # 2. simulated drop of 7 correct -> -10.9 pts: REVERT, config rewritten, PAUSED row
    cfg(); open(N.AR_LOG, "w").close()
    rc = N.autorevert(cur(7)); log = open(N.AR_LOG).read(); c = json.load(open(os.environ["MPMF_RECALL_CONFIG"]))
    print("[drop 10pts]", log.strip().replace("\n", " | "), "rc=%d cfg=%s" % (rc, c))
    if log.count("REVERT") != 2 or c.get("rerank_skip_after_ms") != 4500 or "history_mode" in c: fails.append("simulated drop must REVERT")
    if "history_mode auto-revert" not in open(os.path.join(tmp, "PAUSED.md")).read(): fails.append("PAUSED row missing")
    # 3. simulated pairing failure: ids that match nothing -> rc 2 + alert, not SKIPPED-quiet
    cfg(); open(N.AR_LOG, "w").close()
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = N.autorevert(cur(0, ids=range(1001, 1065)))
    log = open(N.AR_LOG).read()
    print("[pairing failure]", buf.getvalue().strip().replace("\n", " | "), "rc=%d" % rc)
    if rc != 2 or "AUTOREVERT ALERT" not in buf.getvalue() or "SKIPPED" in log: fails.append("pairing failure must exit non-zero")
    # 4. missing baseline -> rc 2 too
    os.remove(os.path.join(tmp, "history_baseline.json"))
    with contextlib.redirect_stdout(io.StringIO()):
        rc = N.autorevert(cur(0))
    if rc != 2: fails.append("missing baseline must exit non-zero")
    print("FAIL: %s" % fails if fails else "OK test_autorevert_pairing")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(run())
