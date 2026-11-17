#!/usr/bin/env python3
"""G5: raw-layer clean test. ONE store file for both arms; the ONLY difference is the raw_layer flag.

One arm = one process:   g5_eval.py ARM STORE_COPY WORKDIR      ARM in {on, off}
  - the per-arm flag is set with the env var MPMF_RAW_LAYER (true/false); the recall config file is the SAME
    file for both arms (WORKDIR/../cfg.json, md5 recorded). Before the first question the harness reads the flag
    back the way the code under test reads it (paths.load_config, hermes_adapter._raw_flag, raw_layer.enabled)
    and REFUSES to run if any of them disagrees with the arm.
  - store md5 is asserted equal to the pre-registered md5 before and after the run (read-only use).
  - prefetch is sequential, single-flight, one provider per question (g4 shape), timed; then answer + 3 judges
    (scorecard ANS_P / ANSWER_MODEL / JUDGES / majority, tie -> partly) 4-wide; same answer model both arms.
Mode G5_MODE=prefetch (no LLM): block bytes only, optional G5_LIMIT=n (used for the base-store control).
Private: questions, blocks and answers go to WORKDIR (under /root/g5), never into the repo.
"""
import hashlib, json, os, re, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

arm, store, work_dir = sys.argv[1], sys.argv[2], sys.argv[3]
assert arm in ("on", "off")
QFILE = "/root/g5/questions_g5.json"
QMD5 = os.environ["G5_QMD5"]
STORE_MD5 = os.environ["G5_STORE_MD5"]
CTRL = os.environ.get("G5_MODE") == "prefetch"          # prefetch-only control
assert os.path.realpath(store) != os.path.realpath("/root/.mpmf/athena.mpmf"), "refusing the live store"
assert os.path.realpath(store).startswith("/root/g_work/g5/copies/"), "store must be a copy under g_work/g5/copies"
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
if not CTRL:
    assert md5(store) == STORE_MD5, "store md5 changed"
assert md5(QFILE) == QMD5, "question file md5 changed"
SPEND_CEIL = 1.5

os.makedirs(work_dir + "/health", exist_ok=True)
os.environ["MPMF_HEALTH_DIR"] = work_dir + "/health"
os.environ["MPMF_THINKING_STORE"] = "/root/g_work/g5/copies/athena-thinking.mpmf"      # a copy, never the live one
assert os.path.exists(os.environ["MPMF_THINKING_STORE"])
cfg_path = os.environ["G5_CFG"]                                     # same file for both arms
cfg = json.load(open(cfg_path))
assert cfg["store_path"] == store or CTRL
os.environ["MPMF_RECALL_CONFIG"] = cfg_path
os.environ["MPMF_RAW_LAYER"] = "true" if arm == "on" else "false"   # THE one difference
os.environ["MPMF_SECOND_PASS"] = "0"
sys.path.insert(0, "/root/mpmf-release/src")
sys.path.insert(0, "/root/.mpmf")
from mpmf_plus.eval import scorecard as sc          # noqa: E402
from mpmf_plus import provider as P                 # noqa: E402
from mpmf_plus import hermes_adapter as H           # noqa: E402
from mpmf_plus import raw_layer as R                # noqa: E402
from mpmf_plus import paths as _paths               # noqa: E402

sc.SNAPSHOT = store
provider, _ = sc.load_provider()
P._RECALL_CONFIG_PATH = cfg_path
P.configure(COGNITION_INJECT_LOG=type(P.COGNITION_INJECT_LOG)(work_dir + "/health/cognition_inject.log"))

# ---- flag read-back, as the harness sees it, BEFORE the first question; refuse on mismatch ----
want = arm == "on"
seen = {"paths.load_config": bool(_paths.load_config().get("raw_layer")),
        "hermes_adapter._raw_flag": bool(H._raw_flag()), "raw_layer.enabled": bool(R.enabled()),
        "raw_layer.settings.on": bool(R.settings()["on"])}
print("FLAG READ-BACK arm=%s expected=%s seen=%s cfg=%s env=%s" % (
    arm, want, seen, cfg_path, os.environ["MPMF_RAW_LAYER"]), flush=True)
if any(v != want for v in seen.values()):
    print("REFUSING: flag read-back does not match the arm", flush=True)
    os._exit(3)
json.dump({"arm": arm, "expected": want, "seen": seen, "cfg": cfg_path, "cfg_md5": md5(cfg_path),
           "store": store, "store_md5_start": md5(store) if not CTRL else None},
          open(work_dir + "/flag_readback.json", "w"))

qs = json.load(open(QFILE))
if os.environ.get("G5_LIMIT"):
    qs = qs[: int(os.environ["G5_LIMIT"])]
norm = lambda s: re.sub(r"\W+", " ", (s or "").lower()).strip()
print("arm", arm, "n", len(qs), flush=True)

# ---- prefetch pass (sequential, single-flight, timed) ----
pre = []
for i, x in enumerate(qs):
    p = H.MPMFPlusProvider()
    p.initialize("g5-%s-%d" % (arm, i))
    R.LAST.clear()
    t = time.perf_counter()
    try:
        ctx, err = p.prefetch(x["q"]) or "", ""
    except Exception as e:
        ctx, err = "", repr(e)[:200]
    ms = (time.perf_counter() - t) * 1000
    pre.append((ctx, ms, err, dict(R.LAST) if arm == "on" else {}))
    if (i + 1) % 25 == 0:
        print("prefetch %d/%d" % (i + 1, len(qs)), flush=True)
with open(work_dir + "/blocks.jsonl", "w") as f:
    for x, (ctx, ms, err, info) in zip(qs, pre):
        f.write(json.dumps({"id": x["id"], "ctx": ctx, "ms": round(ms, 1), "err": err,
                            "raw_rerank": info.get("rerank"), "raw_hits": info.get("hits"),
                            "raw_cand": info.get("n_cand")}) + "\n")
os.chmod(work_dir + "/blocks.jsonl", 0o600)
if CTRL:
    print("prefetch-only done", flush=True)
    os._exit(0)

SRC_N = 60
HDR = "# Raw messages you wrote"


def src_flags(x, ctx):
    srcs = [norm(s)[:SRC_N] for s in x["src_text"].split(" || ") if norm(s)]
    nc = norm(ctx)
    hit = [s in nc for s in srcs]
    return any(hit), all(hit)


def work(a):
    x, (ctx, ms, err, info) = a
    if sc.COST["usd"] > SPEND_CEIL:
        return None
    ans = sc.llm(sc.ANSWER_MODEL, sc.ANS_P.format(ctx=ctx[:6000], q=x["q"]))
    with ThreadPoolExecutor(3) as ex:
        vs = list(ex.map(lambda m: sc.judge_one(m, x["q"], x["key"], ans), sc.JUDGES))
    c = Counter(v for v in vs if v != "unjudged")
    v = c.most_common(1)[0][0] if c else "unjudged"
    if c and len(c) > 1 and c.most_common(2)[0][1] == c.most_common(2)[1][1]:
        v = "partly"
    any_in, all_in = src_flags(x, ctx)
    return {"id": x["id"], "type": x["type"], "arm": arm, "verdict": v, "verdicts": vs,
            "prefetch_ms": round(ms, 1), "block_chars": len(ctx), "truncated_at_6000": len(ctx) > 6000,
            "err": err, "block_sha": hashlib.sha1(ctx.encode()).hexdigest()[:12],
            "src_in_block_any": any_in, "src_in_block_all": all_in, "has_raw_header": HDR in ctx,
            "raw_rerank": info.get("rerank"), "raw_hits": info.get("hits"), "raw_cand": info.get("n_cand"),
            "empty_answer": not ans.strip(), "answer": ans[:300], "cost": round(sc.COST["usd"], 4)}


t0 = time.time()
rows = []
with ThreadPoolExecutor(4) as ex:
    for r in ex.map(work, zip(qs, pre)):
        if r:
            rows.append(r)
with open(work_dir + "/rows.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
os.chmod(work_dir + "/rows.jsonl", 0o600)
end_md5 = md5(store)
json.dump({"arm": arm, "ran": len(rows), "of": len(qs), "secs": round(time.time() - t0),
           "cost_this_process": round(sc.COST["usd"], 4), "judge_unjudged": sum(v == "unjudged" for r in rows for v in r["verdicts"]),
           "store_md5_end": end_md5, "store_unchanged": end_md5 == STORE_MD5},
          open(work_dir + "/summary.json", "w"))
print(json.dumps({"arm": arm, "ran": len(rows), "of": len(qs), "cost": round(sc.COST["usd"], 4),
                  "store_unchanged": end_md5 == STORE_MD5}), flush=True)
sys.stdout.flush()
os._exit(0)
