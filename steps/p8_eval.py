#!/usr/bin/env python3
"""P8 eval: one arm = one process. Prefetch is single-flight (the host shape), then
answer + 3-judge calls run 4-wide. Scoring is scorecard's own (ANS_P, judge_one,
majority vote, tie -> partly); nothing in it is changed here.

usage: p8_eval.py <arm> <store.mpmf> <raw_layer 0|1> <out.json>
env:   P8_SETS="r85b,p6tune"   P8_LIMIT=n (smoke only)
Question files are read, never written. LOCKED is not touched.
"""
import hashlib, json, os, re, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

arm, store, raw_on, out = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4]
os.environ["HOME"] = os.environ.get("HOME") or "/root"
os.makedirs("/root/p8/health", exist_ok=True)
os.environ["MPMF_HEALTH_DIR"] = "/root/p8/health"       # keep prefetch.log etc. out of the live health dir

# per-arm recall config = the LIVE config with the raw flag set explicitly
cfg = json.load(open("/root/.mpmf/recall.config.json"))
cfg["raw_layer"] = raw_on
cfg["store_path"] = store
cfg_path = "/root/p8/cfg_%s.json" % arm
json.dump(cfg, open(cfg_path, "w"), indent=1)
os.environ["MPMF_RECALL_CONFIG"] = cfg_path

sys.path.insert(0, "/root/mpmf-release/src")
sys.path.insert(0, "/root/.mpmf")
from mpmf_plus.eval import scorecard as sc          # noqa: E402
from mpmf_plus import provider as P                 # noqa: E402
from mpmf_plus import raw_layer as R                # noqa: E402

sc.SNAPSHOT = store
provider, prov = sc.load_provider()
P._RECALL_CONFIG_PATH = cfg_path
P.configure(COGNITION_INJECT_LOG=type(P.COGNITION_INJECT_LOG)("/root/p8/health/cognition_inject.log"))

SETS = {"r85b": "/root/r85b/questions.json", "p6tune": "/root/p6_sets/questions_p6_tune.json"}
qs = []
for name in os.environ.get("P8_SETS", "r85b,p6tune").split(","):
    for x in json.load(open(SETS[name])):
        x = dict(x)
        x["uid"] = "%s:%s" % (name, x["id"])
        x["typ"] = x.get("type") or x.get("set")
        x["src_msg"] = x.get("src_text") or x.get("text") or x.get("msg") or ""
        qs.append(x)
if os.environ.get("P8_LIMIT"):
    qs = qs[: int(os.environ["P8_LIMIT"])]

norm = lambda s: re.sub(r"\W+", " ", (s or "").lower()).strip()

print("arm", arm, "store", store, "raw", raw_on, "n", len(qs), flush=True)
pre = []
for i, x in enumerate(qs):
    t = time.perf_counter()
    try:
        ctx = prov.prefetch(x["q"]) or ""
        err = ""
    except Exception as e:                              # a failed prefetch is data, not a crash
        ctx, err = "", repr(e)[:200]
    ms = (time.perf_counter() - t) * 1000
    info = dict(R.LAST) if raw_on else {}
    pre.append((ctx, ms, err, info))
    if (i + 1) % 50 == 0:
        print("prefetch %d/%d" % (i + 1, len(qs)), flush=True)

SRC_N = 60
if os.environ.get("P8_PREFETCH_ONLY") == "1":       # control: block bytes + latency only, no LLM calls
    rows = [{"uid": x["uid"], "typ": x["typ"], "prefetch_ms": round(ms, 1), "block_chars": len(ctx),
             "block_sha": hashlib.sha1(ctx.encode()).hexdigest()[:12], "err": err}
            for x, (ctx, ms, err, info) in zip(qs, pre)]
    json.dump(rows, open(out, "w"), indent=1)
    print("prefetch-only rows", len(rows), flush=True)
    os._exit(0)


def work(a):
    x, (ctx, ms, err, info) = a
    ans = sc.llm(sc.ANSWER_MODEL, sc.ANS_P.format(ctx=ctx[:6000], q=x["q"]))
    with ThreadPoolExecutor(3) as ex:
        vs = list(ex.map(lambda m: sc.judge_one(m, x["q"], x["key"], ans), sc.JUDGES))
    c = Counter(v for v in vs if v != "unjudged")
    v = c.most_common(1)[0][0] if c else "unjudged"
    if c and len(c) > 1 and c.most_common(2)[0][1] == c.most_common(2)[1][1]:
        v = "partly"
    src = norm(x["src_msg"])[:SRC_N]
    return {"uid": x["uid"], "typ": x["typ"], "verdict": v, "verdicts": vs,
            "prefetch_ms": round(ms, 1), "block_chars": len(ctx), "err": err,
            "block_sha": hashlib.sha1(ctx.encode()).hexdigest()[:12],
            "src_in_block": bool(src) and src in norm(ctx),
            "raw_hits": info.get("hits"), "raw_cand": info.get("n_cand"),
            "raw_rerank": info.get("rerank"), "raw_ms": round(info.get("ms", 0), 1) if info else None,
            "raw_has_header": "# Raw messages you wrote" in ctx,
            "answer": ans[:300], "cost": round(sc.COST["usd"], 4)}


rows = []
with ThreadPoolExecutor(4) as ex:
    for i, r in enumerate(ex.map(work, zip(qs, pre)), 1):
        rows.append(r)
        if i % 50 == 0:
            print("judged %d/%d $%.3f" % (i, len(qs), sc.COST["usd"]), flush=True)
        if sc.COST["usd"] > 6.0:
            print("COST CAP hit; stopping", flush=True)
            break
json.dump(rows, open(out, "w"), indent=1)
n = len(rows)
acc = lambda rs: (sum(r["verdict"] == "correct" for r in rs) + 0.5 * sum(r["verdict"] == "partly" for r in rs)) / max(1, len(rs))
p = sorted(r["prefetch_ms"] for r in rows)
print(json.dumps({"arm": arm, "n": n, "acc": round(acc(rows), 4), "cost": round(sc.COST["usd"], 4),
                  "p50": p[n // 2], "p90": p[int(.9 * (n - 1))], "max": p[-1],
                  "errors": sum(bool(r["err"]) for r in rows)}))
sys.stdout.flush()
os._exit(0)
