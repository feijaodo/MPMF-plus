#!/usr/bin/env python3
"""P8 analysis: paired off-vs-on comparison. Scoring is the scorecard's (correct=1, partly=.5).
McNemar is on strict 'correct' flips, exact two-sided, from mpmf_plus.eval.mcnemar (as in earlier steps);
a second McNemar on 'correct or partly' is reported next to it.

usage: p8_analyze.py off.json on.json [control_prefetch_only.json]
"""
import json, sys
from collections import defaultdict
sys.path.insert(0, "/root/mpmf-release/src")
from mpmf_plus.eval import mcnemar   # noqa: E402

off = {r["uid"]: r for r in json.load(open(sys.argv[1]))}
on = {r["uid"]: r for r in json.load(open(sys.argv[2]))}
ids = [u for u in off if u in on]
score = lambda v: 1.0 if v == "correct" else 0.5 if v == "partly" else 0.0
acc = lambda d, us: sum(score(d[u]["verdict"]) for u in us) / max(1, len(us))


def pct(xs, q):
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1))] if xs else None


def block(name, us):
    if not us:
        return None
    b = sum(on[u]["verdict"] == "correct" and off[u]["verdict"] != "correct" for u in us)
    c = sum(off[u]["verdict"] == "correct" and on[u]["verdict"] != "correct" for u in us)
    ok = lambda r: r["verdict"] in ("correct", "partly")
    b2 = sum(ok(on[u]) and not ok(off[u]) for u in us)
    c2 = sum(ok(off[u]) and not ok(on[u]) for u in us)
    d = {"set": name, "n": len(us), "acc_off": round(acc(off, us), 4), "acc_on": round(acc(on, us), 4),
         "delta_pts": round(100 * (acc(on, us) - acc(off, us)), 1),
         "strict_off": sum(off[u]["verdict"] == "correct" for u in us),
         "strict_on": sum(on[u]["verdict"] == "correct" for u in us),
         "gain_b": b, "loss_c": c, "mcnemar_p_strict": round(mcnemar(b, c), 4),
         "gain_b_cp": b2, "loss_c_cp": c2, "mcnemar_p_correct_or_partly": round(mcnemar(b2, c2), 4),
         "declined_off": sum(off[u]["verdict"] == "declined" for u in us),
         "declined_on": sum(on[u]["verdict"] == "declined" for u in us),
         "lat_off_p50": pct([off[u]["prefetch_ms"] for u in us], .5), "lat_on_p50": pct([on[u]["prefetch_ms"] for u in us], .5),
         "lat_off_p90": pct([off[u]["prefetch_ms"] for u in us], .9), "lat_on_p90": pct([on[u]["prefetch_ms"] for u in us], .9),
         "lat_off_max": max(off[u]["prefetch_ms"] for u in us), "lat_on_max": max(on[u]["prefetch_ms"] for u in us),
         "src_in_block_off": sum(off[u]["src_in_block"] for u in us), "src_in_block_on": sum(on[u]["src_in_block"] for u in us),
         "raw_rerank_timeouts": sum(on[u].get("raw_rerank") == "timeout" for u in us)}
    return d


groups = defaultdict(list)
for u in ids:
    s = u.split(":")[0]
    groups["ALL"].append(u)
    groups[s].append(u)
    groups["type=" + str(off[u]["typ"])].append(u)
out = [block(k, groups[k]) for k in ["ALL", "r85b", "p6tune"] + sorted(k for k in groups if k.startswith("type="))]
for d in out:
    if d:
        print(json.dumps(d))
extra = {"n_paired": len(ids), "cost_off": off[ids[-1]]["cost"], "cost_on": on[ids[-1]]["cost"],
         "errors_off": sum(bool(off[u]["err"]) for u in ids), "errors_on": sum(bool(on[u]["err"]) for u in ids),
         "on_block_has_raw_section": sum(on[u]["raw_has_header"] for u in ids),
         "block_chars_off_mean": round(sum(off[u]["block_chars"] for u in ids) / len(ids)),
         "block_chars_on_mean": round(sum(on[u]["block_chars"] for u in ids) / len(ids))}
print(json.dumps(extra))
if len(sys.argv) > 3:
    ctl = {r["uid"]: r for r in json.load(open(sys.argv[3]))}
    same = sum(ctl[u]["block_sha"] == off[u]["block_sha"] for u in ids if u in ctl)
    print(json.dumps({"control": "aug store + flag off vs base store + flag off",
                      "blocks_compared": sum(u in ctl for u in ids), "byte_identical": same}))
# wins / losses for review
wl = {"gained": [u for u in ids if on[u]["verdict"] == "correct" and off[u]["verdict"] != "correct"],
      "lost": [u for u in ids if off[u]["verdict"] == "correct" and on[u]["verdict"] != "correct"]}
print(json.dumps(wl))
