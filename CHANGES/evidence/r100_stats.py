#!/usr/bin/env python3
"""r100 statistics: strict/lenient + exact McNemar, dev and held-out v2.

lenient score = (correct + 0.5*partly)/n   (the README's "score")
strict        = correct only
McNemar exact two-sided binomial on discordant pairs (b = off fail & on pass).
"""
import json, math, sys
from collections import Counter

def wilson(k, n, z=1.96):
    if n == 0: return [0.0, 1.0]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 4), round(min(1, c + h), 4)]

def binom_two_sided(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    # P(X<=k) * 2, X ~ Bin(n, .5), capped at 1
    s = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * s)

def mcnemar(off, on, ok=lambda v: v == "correct"):
    b = sum((not ok(o["verdict"])) and ok(n["verdict"]) for o, n in zip(off, on))
    c = sum(ok(o["verdict"]) and (not ok(n["verdict"])) for o, n in zip(off, on))
    return {"b_off_fail_on_pass": b, "c_off_pass_on_fail": c, "p_exact_two_sided": "%.2e" % binom_two_sided(b, c)}

def arm(rows):
    n = len(rows)
    cnt = Counter(r["verdict"] for r in rows)
    k = cnt["correct"]; part = cnt["partly"]
    return {"n": n, "strict": k, "strict_pct": round(100 * k / n, 1), "strict_ci95": wilson(k, n),
            "lenient_score": round((k + .5 * part) / n, 4), "lenient_pct": round(100 * (k + .5 * part) / n, 1),
            "correct": k, "partly": part, "declined": cnt["declined"], "wrong": cnt["wrong"],
            "unjudged": cnt["unjudged"], "seq_ok": sum(bool(r.get("seq_ok")) for r in rows),
            "hist_fired": sum(bool(r.get("hist")) for r in rows)}

out = {"method": "strict=verdict correct; lenient=(correct+0.5*partly)/n; McNemar exact two-sided binomial",
       "sets": {}}
for name in ("dev", "heldout"):
    off = json.load(open(f"/root/ops-mpmf/r100/{name}_off.json"))
    on = json.load(open(f"/root/ops-mpmf/r100/{name}_on.json"))
    assert [r["id"] for r in off] == [r["id"] for r in on], "row order differ"
    s = {"off": arm(off), "on": arm(on),
         "mcnemar_strict": mcnemar(off, on),
         "mcnemar_lenient": mcnemar(off, on, ok=lambda v: v in ("correct", "partly"))}
    out["sets"][name] = s
out["spend_usd"] = {n: {"off": round(json.load(open(f"/root/ops-mpmf/r100/{n}_off.json"))[-1]["cost"], 4),
                        "on": round(json.load(open(f"/root/ops-mpmf/r100/{n}_on.json"))[-1]["cost"], 4)}
                    for n in ("dev", "heldout")}
out["spend_total_usd"] = round(sum(v["on"] + v["off"] for v in out["spend_usd"].values()), 4)
print(json.dumps(out, indent=1))
json.dump(out, open("/root/ops-mpmf/r100/stats.json", "w"), indent=1)
