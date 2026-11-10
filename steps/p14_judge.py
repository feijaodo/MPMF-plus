#!/usr/bin/env python3
"""P14: judge precision of extracted USED_FOR/PART_OF edges on a random sample of 40.

usage: p14_judge.py <outdir>      (reads edges.jsonl written by p14_extract.py; sample seed 1415)
Judges (both see only the two facts + the user messages they came from + the conversation title,
NOT the extractor's `basis`): anthropic/claude-sonnet-5.5 (primary) and openai/gpt-4.1-mini (agreement check).
Verdicts: correct / reversed / wrong_relation / not_related. Precision = share of the primary judge's `correct`.
Also reported: per-relation precision, reversed-direction count, explicit-vs-implicit split, Wilson 95% CI, judge agreement.
"""
import json, math, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor

out = sys.argv[1]
sys.path.insert(0, os.environ.get("MPMF_SRC", "/root/mpmf-release/src"))
from mpmf_plus.eval import scorecard as sc   # noqa: E402

edges = [json.loads(l) for l in open(os.path.join(out, "edges.jsonl"))]
rng = random.Random(1415)
sample = rng.sample(edges, min(40, len(edges)))
J1, J2 = "anthropic/claude-sonnet-5.5", "openai/gpt-4.1-mini"

P = """A system extracted a functional relation between two facts from a personal conversation. Judge it strictly.
Relation claimed: {rel}
  USED_FOR: fact A is the means (tool/method/resource) the user uses or intends to use for the purpose/task in fact B.
  PART_OF:  fact A is a part/component/member/step of the whole described in fact B.

Conversation title: {t}
FACT A: {a}
  user message A: {ea}
FACT B: {b}
  user message B: {eb}

Verdicts:
- "correct": the claimed relation really holds, direction right, supported by the user's words.
- "reversed": the relation holds but with A and B swapped.
- "wrong_relation": a real relation exists but it is the other kind (e.g. USED_FOR vs PART_OF) or a causal one.
- "not_related": co-occurrence, shared topic, sequence, restatement, request/follow-up, or a guess the text does not support.

Reply with JSON only: {{"verdict": "correct|reversed|wrong_relation|not_related", "why": "<15 words"}}"""


def judge(model, e):
    raw = sc.llm(model, P.format(rel=e["rel"], t=e["conv_title"], a=e["cause_text"], ea=e["cause_ev"][:300],
                                 b=e["effect_text"], eb=e["effect_ev"][:300]), 300) or ""
    v = "unjudged"
    for tok in ("wrong_relation", "not_related", "reversed", "correct"):
        if '"%s"' % tok in raw:
            v = tok
            break
    m = re.search(r'"why"\s*:\s*"([^"]*)"', raw)
    return v, (m.group(1) if m else "")


def run(e):
    v1, w1 = judge(J1, e)
    v2, w2 = judge(J2, e)
    return dict(e, j1=v1, why1=w1, j2=v2, why2=w2)


with ThreadPoolExecutor(5) as ex:
    rows = list(ex.map(run, sample))


def wilson(k, n, z=1.96):
    if not n:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


n = len(rows)
k1 = sum(r["j1"] == "correct" for r in rows)
k2 = sum(r["j2"] == "correct" for r in rows)
both = sum(r["j1"] == "correct" and r["j2"] == "correct" for r in rows)
agree = sum(r["j1"] == r["j2"] for r in rows)
ex_ = [r for r in rows if r["explicit"]]
im_ = [r for r in rows if not r["explicit"]]
summ = {"n_edges_total": len(edges), "n_sample": n,
        "precision_primary(sonnet-5.5)": round(k1 / n, 3), "ci95": wilson(k1, n), "correct": k1,
        "reversed": sum(r["j1"] == "reversed" for r in rows), "not_related": sum(r["j1"] == "not_related" for r in rows),
        "unjudged": sum(r["j1"] == "unjudged" for r in rows),
        "precision_second(gpt-4.1-mini)": round(k2 / n, 3), "precision_both_agree_correct": round(both / n, 3),
        "judge_agreement": round(agree / n, 3),
        "wrong_relation": sum(r["j1"] == "wrong_relation" for r in rows),
        "by_rel": {k: {"n": sum(r["rel"] == k for r in rows), "correct": sum(r["rel"] == k and r["j1"] == "correct" for r in rows)} for k in ("USED_FOR", "PART_OF")},
        "explicit_n": len(ex_), "explicit_precision": round(sum(r["j1"] == "correct" for r in ex_) / max(1, len(ex_)), 3),
        "implicit_n": len(im_), "implicit_precision": round(sum(r["j1"] == "correct" for r in im_) / max(1, len(im_)), 3),
        "judge_cost_usd_(price table underestimates sonnet)": round(sc.COST["usd"], 4)}
json.dump({"summary": summ, "rows": rows}, open(os.path.join(out, "judged.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
