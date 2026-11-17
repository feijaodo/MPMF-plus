"""R90e: all 64 order questions, history ON. Prefetch is run single-flight
(the host shape), then answer+judge calls run 4-wide."""
import json, os, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/root/mpmf-release/src")
sys.path.insert(0, "/root/.mpmf")
from mpmf_plus.eval import scorecard as sc
from mpmf_plus import provider as P

sc.SNAPSHOT = "/root/.mpmf/athena.mpmf"
provider, prov = sc.load_provider()
P._RECALL_CONFIG_PATH = os.environ.get("HIST_CFG", "/root/mpmf-release/.r90e_history_on.json")
qs = json.load(open("/root/r88/questions_order.json"))
SEQ = """Question: {q}
Answer: {a}

Does the answer give the events in the CORRECT chronological order? Reply JSON only: {{"sequence_right": "yes|no"}}"""

# Phase 1: prefetch, one at a time (host calls prefetch once per turn).
ctxs = []
for x in qs:
    t = time.perf_counter()
    ctx = prov.prefetch(x["q"]) or ""
    ctxs.append((ctx, (time.perf_counter() - t) * 1000))

def work(x):
    x, (ctx, pre) = x
    ans = sc.llm(sc.ANSWER_MODEL, sc.ANS_P.format(ctx=ctx[:8000], q=x["q"]))
    with ThreadPoolExecutor(3) as ex:
        vs = list(ex.map(lambda m: sc.judge_one(m, x["q"], x["key"], ans), sc.JUDGES))
    c = Counter(v for v in vs if v != "unjudged")
    v = c.most_common(1)[0][0] if c else "unjudged"
    if c and len(c) > 1 and c.most_common(2)[0][1] == c.most_common(2)[1][1]:
        v = "partly"
    raw = sc.llm("deepseek/deepseek-v4.1-flash", SEQ.format(q=x["q"], a=ans[:1500]), 60)
    return {"id": x["id"], "q": x["q"], "verdict": v, "seq_ok": '"yes"' in raw,
            "prefetch_ms": round(pre, 1), "hist": "(history ·" in ctx, "block": len(ctx),
            "answer": ans[:300], "cost": sc.COST["usd"]}

rows = []
with ThreadPoolExecutor(4) as ex:
    for i, r in enumerate(ex.map(work, zip(qs, ctxs)), 1):
        rows.append(r)
        if i % 16 == 0:
            print(i, "/", len(rows), "$%.3f" % sc.COST["usd"], flush=True)
json.dump(rows, open(os.environ.get("OUT", "/root/mpmf-release/order_on64_r90e.json"), "w"), indent=1)
n = len(rows)
a = (sum(r["verdict"] == "correct" for r in rows) + 0.5 * sum(r["verdict"] == "partly" for r in rows)) / n
pre = sorted(r["prefetch_ms"] for r in rows)
print(json.dumps({"n": n, "acc": round(a, 4), "correct_equiv": round(a * n, 2),
                  "seq": round(sum(r["seq_ok"] for r in rows) / n, 4),
                  "hist": sum(r["hist"] for r in rows),
                  "p50": pre[n // 2], "p90": pre[int(.9 * (n - 1))], "max": pre[-1],
                  "cost": round(sc.COST["usd"], 4)}, indent=1))
