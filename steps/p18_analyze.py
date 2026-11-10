#!/usr/bin/env python3
"""P18 analysis: paired arm comparisons from p18_eval.py output. Scoring is the scorecard's
(correct=1, partly=.5); McNemar exact two-sided (mpmf_plus.eval.mcnemar) on strict 'correct' flips and on
'correct or partly'.   usage: p18_analyze.py run.json"""
import json, sys
sys.path.insert(0, "/root/mpmf-release/src")
from mpmf_plus.eval import mcnemar   # noqa: E402

rows = json.load(open(sys.argv[1]))
by = {}
for r in rows:
    by.setdefault(r["arm"], {})[r["id"]] = r
score = lambda v: 1.0 if v == "correct" else 0.5 if v == "partly" else 0.0
ok = lambda r: r["verdict"] in ("correct", "partly")


def cmp(a, b, ids):
    A, B = by[a], by[b]
    g = sum(B[i]["verdict"] == "correct" and A[i]["verdict"] != "correct" for i in ids)
    l = sum(A[i]["verdict"] == "correct" and B[i]["verdict"] != "correct" for i in ids)
    g2 = sum(ok(B[i]) and not ok(A[i]) for i in ids)
    l2 = sum(ok(A[i]) and not ok(B[i]) for i in ids)
    acc = lambda D: sum(score(D[i]["verdict"]) for i in ids) / max(1, len(ids))
    return {"cmp": "%s -> %s" % (a, b), "n": len(ids), "acc_a": round(acc(A), 4), "acc_b": round(acc(B), 4),
            "delta_pts": round(100 * (acc(B) - acc(A)), 1),
            "strict_a": sum(A[i]["verdict"] == "correct" for i in ids),
            "strict_b": sum(B[i]["verdict"] == "correct" for i in ids),
            "gain": g, "loss": l, "p_strict": round(mcnemar(g, l), 5),
            "gain_cp": g2, "loss_cp": l2, "p_cp": round(mcnemar(g2, l2), 5),
            "declined_a": sum(A[i]["verdict"] == "declined" for i in ids),
            "declined_b": sum(B[i]["verdict"] == "declined" for i in ids)}


arms = [a for a in ("off", "oldtool", "on") if a in by]
ids_all = sorted(by[arms[0]])
groups = {"ALL": ids_all}
for i in ids_all:
    groups.setdefault("type=" + by[arms[0]][i]["type"], []).append(i)
for g, ids in groups.items():
    for a, b in (("off", "on"), ("off", "oldtool"), ("oldtool", "on")):
        if a in by and b in by:
            print(json.dumps(dict(set=g, **cmp(a, b, ids))))
for a in arms:
    R = list(by[a].values())
    n_tool = sum(r["tool_calls"] > 0 for r in R)
    print(json.dumps({"arm": a, "questions_with_tool_call": n_tool, "tool_calls": sum(r["tool_calls"] for r in R),
                      "src_in_block": sum(r["src_in_block"] for r in R),
                      "src_found_only_via_tool": sum((not r["src_in_block"]) and r["src_in_tool"] for r in R),
                      "tool_ms_mean": round(sum(l["ms"] for r in R for l in r["tool_log"]) /
                                            max(1, sum(r["tool_calls"] for r in R)), 1),
                      "tool_chars_mean": round(sum(l["chars"] for r in R for l in r["tool_log"]) /
                                               max(1, sum(r["tool_calls"] for r in R))),
                      "cost_end": R[-1]["cost"]}))
# where the tool was used: accuracy of questions in which it was called, per arm
used = [i for i in ids_all if by["on"][i]["tool_calls"] or by.get("oldtool", {}).get(i, {}).get("tool_calls")] if "on" in by else []
if used:
    for a, b in (("off", "on"), ("oldtool", "on")):
        if a in by:
            print(json.dumps(dict(set="tool-used-in-on-or-oldtool", **cmp(a, b, used))))
