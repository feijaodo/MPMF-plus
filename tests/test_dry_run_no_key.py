#!/usr/bin/env python3
"""r92: `nightly --dry-run` must work with NO API key and a foreign HOME (clean-install case)."""
import os, subprocess, sys, tempfile, json
src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
d = tempfile.mkdtemp(prefix="mpmf-nokey-"); os.makedirs(d + "/eval"); os.makedirs(d + "/home")
rows = {str(i): "correct" for i in range(1, 65)}
for f in ("rerank_baseline.json", "history_baseline.json"):
    json.dump({"acc": 1.0, "n": 64, "rows": rows}, open(f"{d}/eval/{f}", "w"))
open(d + "/eval/scorecard.jsonl", "w").write("".join(json.dumps({"id": i, "split": "holdout", "verdict": "correct"}) + "\n" for i in range(1, 65)))
env = {"PATH": os.environ["PATH"], "HOME": d + "/home", "MPMF_HOME": d, "MPMF_EVAL_DIR": d + "/eval", "PYTHONPATH": src}
r = subprocess.run([sys.executable, "-m", "mpmf_plus.eval", "nightly", "--dry-run"], capture_output=True, text=True, env=env)
print(r.stdout.strip().splitlines()[-3:], "rc=%d" % r.returncode)
ok = r.returncode == 0 and "64 of 64" in r.stdout and "no OPENROUTER_API_KEY" not in r.stderr
print("OK test_dry_run_no_key" if ok else "FAIL: " + r.stderr[-300:])
sys.exit(0 if ok else 1)
