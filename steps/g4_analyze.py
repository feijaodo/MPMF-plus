#!/usr/bin/env python3
"""G4 analysis with the pre-registered rule (see steps/g4.notes.md). usage: g4_analyze.py WORKDIR
Decision: ON vs OFF-live (arm 'oldtool'). WIN = (1) strict ON > OFF-live, (2) exact two-sided McNemar on strict
p<0.05 and b>c, (3) prefetch p90 (flag ON, 74 q) < prefetch_budget_ms. Anything else = no_effect."""
import json, math, sys
sys.path.insert(0, "/root/mpmf-release/src")
from mpmf_plus.eval import mcnemar   # noqa: E402

W = sys.argv[1]
BUDGET = 6500
rows = [json.loads(l) for l in open(W + "/rows.jsonl")]
pf = json.load(open(W + "/prefetch.json"))
by = {}
for r in rows:
    by.setdefault(r["arm"], {})[r["id"]] = r
ids = sorted(by["oldtool"])
for a in by:
    assert sorted(by[a]) == ids, "arm %s incomplete: %d of %d" % (a, len(by[a]), len(ids))
n = len(ids)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 2), round(c + h, 2)


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


strict = lambda r: r["verdict"] == "correct"
cp = lambda r: r["verdict"] in ("correct", "partly")
out = {}
for a in ("oldtool", "on", "off"):
    if a not in by:
        continue
    R = [by[a][i] for i in ids]
    s, c = sum(map(strict, R)), sum(map(cp, R))
    out[a] = {"strict": s, "strict_ci95": wilson(s, n), "correct_or_partly": c, "cp_ci95": wilson(c, n),
              "declined": sum(r["verdict"] == "declined" for r in R),
              "questions_with_tool_call": sum(r["tool_calls"] > 0 for r in R),
              "tool_calls": sum(r["tool_calls"] for r in R),
              "answer_ms_mean": round(sum(r["answer_ms"] for r in R) / n, 1),
              "answer_ms_p90": round(pct([r["answer_ms"] for r in R], .9), 1),
              "empty_answers": sum(r["empty_answer"] for r in R),
              "tool_errors": sum(1 for r in R for l in r["tool_log"] if l.get("err")),
              "unjudged": sum(r["verdict"] == "unjudged" for r in R)}


def flips(a, b, f):
    g = sum(f(by[b][i]) and not f(by[a][i]) for i in ids)
    l = sum(f(by[a][i]) and not f(by[b][i]) for i in ids)
    return g, l, round(mcnemar(g, l), 5)


b, c, p = flips("oldtool", "on", strict)
bc, cc, pc = flips("oldtool", "on", cp)
res = {"n": n, "arms": out, "ON_vs_OFFlive_strict": {"b": b, "c": c, "p": p},
       "ON_vs_OFFlive_correct_or_partly": {"b": bc, "c": cc, "p": pc}}
if "off" in by:
    b0, c0, p0 = flips("off", "on", strict)
    b1, c1, p1 = flips("off", "oldtool", strict)
    res["secondary_ON_vs_notool_strict"] = {"b": b0, "c": c0, "p": p0}
    res["secondary_OFFlive_vs_notool_strict"] = {"b": b1, "c": c1, "p": p1}
on_ms = pf["on_ms"]
res["prefetch_flag_on_ms"] = {"n": len(on_ms), "mean": round(sum(on_ms) / len(on_ms), 1), "p50": round(pct(on_ms, .5), 1),
                              "p90": round(pct(on_ms, .9), 1), "max": max(on_ms), "budget": BUDGET,
                              "errors": sum(bool(e) for e in pf["on_err"])}
off_ms = pf["off_ms"]
res["prefetch_flag_off_ms"] = {"mean": round(sum(off_ms) / len(off_ms), 1), "p90": round(pct(off_ms, .9), 1),
                               "errors": sum(bool(e) for e in pf["off_err"]),
                               "block_identical_on_vs_off": sum(pf["block_equal"])}
tm = [by["on"][i]["tool_ms_total"] for i in ids]
used = [x for x in tm if x > 0]
res["tool_path_extra_ms_per_question_ON"] = {"all74_mean": round(sum(tm) / n, 1), "all74_p90": round(pct(tm, .9), 1),
                                              "used_only_n": len(used),
                                              "used_only_mean": round(sum(used) / max(1, len(used)), 1),
                                              "used_only_p90": round(pct(used, .9), 1) if used else 0}
tm0 = [by["oldtool"][i]["tool_ms_total"] for i in ids]
res["tool_path_extra_ms_per_question_OFFlive"] = {"all74_mean": round(sum(tm0) / n, 1), "all74_p90": round(pct(tm0, .9), 1)}
w1 = out["on"]["strict"] > out["oldtool"]["strict"]
w2 = p < 0.05 and b > c
w3 = res["prefetch_flag_on_ms"]["p90"] < BUDGET
res["WIN_rule"] = {"1_strict_ON_gt_OFFlive": w1, "2_mcnemar_p_lt_0.05_and_b_gt_c": w2, "3_prefetch_p90_under_budget": w3,
                   "WIN": bool(w1 and w2 and w3), "EFFECT": "helped" if (w1 and w2 and w3) else "no_effect"}
res["cost_last_row_per_arm"] = {a: max(r["cost"] for r in by[a].values()) for a in by}
print(json.dumps(res, indent=1))
