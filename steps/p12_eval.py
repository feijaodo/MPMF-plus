#!/usr/bin/env python3
"""P12 eval: one arm = one process; same store copy, only the `typed_walk` flag differs.
Prefetch is single-flight (host shape), then answer + 3 judges 4-wide with the scorecard's own
prompts / judges / majority rule (nothing in scoring is changed here).

usage: p12_eval.py ARM STORE_COPY TYPED_WALK(0|1) OUT.json
env:   P12_SETS=main,control   P12_REPEAT=n (latency: extra timed prefetches per question; default 0)
"""
import hashlib, json, os, re, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

arm, store, tw_on, out = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4]
assert os.path.realpath(store) != os.path.realpath("/root/.mpmf/athena.mpmf"), "refusing the live store"
os.environ["HOME"] = os.environ.get("HOME") or "/root"
os.makedirs("/root/p12/health", exist_ok=True)
os.environ["MPMF_HEALTH_DIR"] = "/root/p12/health"
cfg = json.load(open("/root/.mpmf/recall.config.json"))
cfg["typed_walk"] = tw_on
cfg["store_path"] = store
cfg_path = "/root/p12/cfg_%s.json" % arm
json.dump(cfg, open(cfg_path, "w"), indent=1)
os.environ["MPMF_RECALL_CONFIG"] = cfg_path
sys.path.insert(0, "/root/mpmf-release/src")
sys.path.insert(0, "/root/.mpmf")
from mpmf_plus.eval import scorecard as sc          # noqa: E402
from mpmf_plus import provider as P                 # noqa: E402
from mpmf_plus import typed_walk as TW              # noqa: E402

sc.SNAPSHOT = store
provider, prov = sc.load_provider()
P._RECALL_CONFIG_PATH = cfg_path
P.configure(COGNITION_INJECT_LOG=type(P.COGNITION_INJECT_LOG)("/root/p12/health/cognition_inject.log"))

qs = []
for name in os.environ.get("P12_SETS", "main,control").split(","):
    for x in json.load(open("/root/p12/sets/questions_p12_%s.json" % name)):
        x = dict(x)
        x["uid"] = "%s:%s" % (name, x["id"])
        x["set"] = name
        qs.append(x)
norm = lambda s: re.sub(r"\W+", " ", (s or "").lower()).strip()
print("arm", arm, "typed_walk", tw_on, "n", len(qs), flush=True)

pre = []
for i, x in enumerate(qs):
    TW.LAST.clear()
    t = time.perf_counter()
    try:
        ctx, err = prov.prefetch(x["q"]) or "", ""
    except Exception as e:
        ctx, err = "", repr(e)[:200]
    ms = (time.perf_counter() - t) * 1000
    info = dict(TW.LAST) if tw_on else {}
    reps = []
    for _ in range(int(os.environ.get("P12_REPEAT", "0"))):
        t = time.perf_counter()
        prov.prefetch(x["q"])
        reps.append((time.perf_counter() - t) * 1000)
    pre.append((ctx, ms, err, info, reps))
    if (i + 1) % 20 == 0:
        print("prefetch %d/%d" % (i + 1, len(qs)), flush=True)


def work(a):
    x, (ctx, ms, err, info, reps) = a
    ans = sc.llm(sc.ANSWER_MODEL, sc.ANS_P.format(ctx=ctx[:6000], q=x["q"]))
    with ThreadPoolExecutor(3) as ex:
        vs = list(ex.map(lambda m: sc.judge_one(m, x["q"], x["key"], ans), sc.JUDGES))
    c = Counter(v for v in vs if v != "unjudged")
    v = c.most_common(1)[0][0] if c else "unjudged"
    if c and len(c) > 1 and c.most_common(2)[0][1] == c.most_common(2)[1][1]:
        v = "partly"
    n = norm(ctx)
    a_in = norm(x["text_a"])[:60] in n
    b_in = norm(x["text_b"])[:60] in n
    return {"uid": x["uid"], "set": x["set"], "typ": x["link"], "verdict": v, "verdicts": vs,
            "prefetch_ms": round(ms, 1), "repeat_ms": [round(r, 1) for r in reps], "block_chars": len(ctx), "err": err,
            "block_sha": hashlib.sha1(ctx.encode()).hexdigest()[:12],
            "a_in_block": a_in, "b_in_block": b_in, "both_in_block": a_in and b_in,
            "walk_added": info.get("added"), "walk_labels": info.get("labels"),
            "walk_ms": round(info.get("ms", 0), 2) if info else None,
            "answer": ans[:300], "cost": round(sc.COST["usd"], 4)}


rows = []
with ThreadPoolExecutor(4) as ex:
    for i, r in enumerate(ex.map(work, zip(qs, pre)), 1):
        rows.append(r)
        if sc.COST["usd"] > 1.0:
            print("COST CAP hit; stopping", flush=True)
            break
json.dump(rows, open(out, "w"), indent=1)
acc = lambda rs: (sum(r["verdict"] == "correct" for r in rs) + 0.5 * sum(r["verdict"] == "partly" for r in rs)) / max(1, len(rs))
p = sorted(r["prefetch_ms"] for r in rows)
print(json.dumps({"arm": arm, "n": len(rows), "acc": round(acc(rows), 4), "cost": round(sc.COST["usd"], 4),
                  "p50": p[len(p) // 2], "p90": p[int(.9 * (len(p) - 1))], "max": p[-1],
                  "errors": sum(bool(r["err"]) for r in rows)}))
sys.stdout.flush()
os._exit(0)
