#!/usr/bin/env python3
"""P13: judge precision of extracted CAUSES edges on a random sample of 30.

usage: p13_judge.py <outdir>      (reads edges.jsonl written by p13_extract.py; sample seed 1314)
Judges (both see only the two facts + the user messages they came from + the conversation title,
NOT the extractor's `basis`): anthropic/claude-sonnet-5.5 (primary) and openai/gpt-4.1-mini (agreement check).
Verdicts: causal (A caused/is the reason for B, supported by the text) / not_causal (co-occurrence, sequence, topic,
restatement, reversed direction, or unsupported guess). Precision = share of the primary judge's `causal`.
Also reported: reversed-direction count, explicit-vs-implicit split, Wilson 95% CI, judge agreement.
"""
import json, math, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor

out = sys.argv[1]
sys.path.insert(0, os.environ.get("MPMF_SRC", "/root/mpmf-release/src"))
from mpmf_plus.eval import scorecard as sc   # noqa: E402

edges = [json.loads(l) for l in open(os.path.join(out, "edges.jsonl"))]
rng = random.Random(1314)
sample = rng.sample(edges, min(30, len(edges)))
J1, J2 = "anthropic/claude-sonnet-5.5", "openai/gpt-4.1-mini"

P = """A system extracted a CAUSES relation from a personal conversation: fact A is claimed to be the CAUSE (or reason)
of fact B. Judge it strictly.

Conversation title: {t}
FACT A (claimed cause): {a}
  user message A: {ea}
FACT B (claimed effect): {b}
  user message B: {eb}

Verdicts:
- "causal": reading the user's words, A really brought B about / is the stated reason for B (the user says so or the
  mechanism is unmistakable). Direction is right.
- "reversed": a causal link exists but B is the cause and A the effect.
- "not_causal": co-occurrence, sequence in time, shared topic, request/follow-up, restatement, or a guess the text does not support.

Reply with JSON only: {{"verdict": "causal|reversed|not_causal", "why": "<15 words"}}"""


def judge(model, e):
    raw = sc.llm(model, P.format(t=e["conv_title"], a=e["cause_text"], ea=e["cause_ev"][:300],
                                 b=e["effect_text"], eb=e["effect_ev"][:300]), 300) or ""
    v = "unjudged"
    for tok in ("not_causal", "reversed", "causal"):
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
k1 = sum(r["j1"] == "causal" for r in rows)
k2 = sum(r["j2"] == "causal" for r in rows)
both = sum(r["j1"] == "causal" and r["j2"] == "causal" for r in rows)
agree = sum(r["j1"] == r["j2"] for r in rows)
ex_ = [r for r in rows if r["explicit"]]
im_ = [r for r in rows if not r["explicit"]]
summ = {"n_edges_total": len(edges), "n_sample": n,
        "precision_primary(sonnet-5.5)": round(k1 / n, 3), "ci95": wilson(k1, n), "causal": k1,
        "reversed": sum(r["j1"] == "reversed" for r in rows), "not_causal": sum(r["j1"] == "not_causal" for r in rows),
        "unjudged": sum(r["j1"] == "unjudged" for r in rows),
        "precision_second(gpt-4.1-mini)": round(k2 / n, 3), "precision_both_agree_causal": round(both / n, 3),
        "judge_agreement": round(agree / n, 3),
        "explicit_n": len(ex_), "explicit_precision": round(sum(r["j1"] == "causal" for r in ex_) / max(1, len(ex_)), 3),
        "implicit_n": len(im_), "implicit_precision": round(sum(r["j1"] == "causal" for r in im_) / max(1, len(im_)), 3),
        "judge_cost_usd_(price table underestimates sonnet)": round(sc.COST["usd"], 4)}
json.dump({"summary": summ, "rows": rows}, open(os.path.join(out, "judged.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
