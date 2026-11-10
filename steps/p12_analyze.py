#!/usr/bin/env python3
"""P12 analysis: paired off-vs-on. Scoring is the scorecard's (correct=1, partly=.5). McNemar exact two-sided
(mpmf_plus.eval.mcnemar) on strict 'correct' flips, a second on 'correct or partly'.
Also: how often both source facts are in the block, walk activity, latency (first prefetch + repeats).

usage: p12_analyze.py off.json on.json
"""
import json, sys
sys.path.insert(0, "/root/mpmf-release/src")
from mpmf_plus.eval import mcnemar   # noqa: E402

off = {r["uid"]: r for r in json.load(open(sys.argv[1]))}
on = {r["uid"]: r for r in json.load(open(sys.argv[2]))}
ids = [u for u in off if u in on]
score = lambda v: 1.0 if v == "correct" else 0.5 if v == "partly" else 0.0
acc = lambda d, us: sum(score(d[u]["verdict"]) for u in us) / max(1, len(us))


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[int(q * (len(xs) - 1))], 1) if xs else None


def block(name, us):
    if not us:
        return None
    ok = lambda r: r["verdict"] in ("correct", "partly")
    b = sum(on[u]["verdict"] == "correct" and off[u]["verdict"] != "correct" for u in us)
    c = sum(off[u]["verdict"] == "correct" and on[u]["verdict"] != "correct" for u in us)
    b2 = sum(ok(on[u]) and not ok(off[u]) for u in us)
    c2 = sum(ok(off[u]) and not ok(on[u]) for u in us)
    rep = lambda d: [x for u in us for x in d[u]["repeat_ms"]]
    return {"set": name, "n": len(us), "acc_off": round(acc(off, us), 4), "acc_on": round(acc(on, us), 4),
            "delta_pts": round(100 * (acc(on, us) - acc(off, us)), 1),
            "strict_off": sum(off[u]["verdict"] == "correct" for u in us),
            "strict_on": sum(on[u]["verdict"] == "correct" for u in us),
            "gain_b": b, "loss_c": c, "mcnemar_p_strict": round(mcnemar(b, c), 4),
            "gain_b_cp": b2, "loss_c_cp": c2, "mcnemar_p_correct_or_partly": round(mcnemar(b2, c2), 4),
            "declined_off": sum(off[u]["verdict"] == "declined" for u in us),
            "declined_on": sum(on[u]["verdict"] == "declined" for u in us),
            "both_in_block_off": sum(off[u]["both_in_block"] for u in us),
            "both_in_block_on": sum(on[u]["both_in_block"] for u in us),
            "block_changed": sum(off[u]["block_sha"] != on[u]["block_sha"] for u in us),
            "walk_added_total": sum((on[u].get("walk_added") or 0) for u in us),
            "walk_ms_mean": round(sum((on[u].get("walk_ms") or 0) for u in us) / len(us), 2),
            "lat_first_p50": [pct([off[u]["prefetch_ms"] for u in us], .5), pct([on[u]["prefetch_ms"] for u in us], .5)],
            "lat_first_p90": [pct([off[u]["prefetch_ms"] for u in us], .9), pct([on[u]["prefetch_ms"] for u in us], .9)],
            "lat_warm_p50": [pct(rep(off), .5), pct(rep(on), .5)],
            "lat_warm_p90": [pct(rep(off), .9), pct(rep(on), .9)]}


groups = {"ALL": ids}
for u in ids:
    groups.setdefault("set=" + off[u]["set"], []).append(u)
    groups.setdefault("link=" + off[u]["typ"], []).append(u)
    groups.setdefault("set=%s link=%s" % (off[u]["set"], off[u]["typ"]), []).append(u)
for k in sorted(groups, key=lambda s: (s != "ALL", s)):
    print(json.dumps(block(k, groups[k])))
labs = {}
for u in ids:
    for l in on[u].get("walk_labels") or []:
        labs[l] = labs.get(l, 0) + 1
print(json.dumps({"n_paired": len(ids), "walk_labels_added": labs,
                  "errors_off": sum(bool(off[u]["err"]) for u in ids), "errors_on": sum(bool(on[u]["err"]) for u in ids),
                  "cost_off": off[ids[-1]]["cost"], "cost_on": on[ids[-1]]["cost"]}))
