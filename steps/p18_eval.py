#!/usr/bin/env python3
"""P18 eval: second-pass retrieval (the model calls mpmf_recall while answering).

One process, one store COPY, three arms over the SAME prefetch block per question (prefetch is done
once, single-flight, host shape; the block is therefore identical across arms):
  off     scorecard prompt (ANS_P, ctx[:6000]), no tool                 = what the scorecard measures today
  oldtool same prompt + tool addendum, flag second_pass OFF (the shipped mpmf_recall: 300-char hits)
  on      same prompt + tool addendum, flag second_pass ON  (new: same rendering as prefetch, only
          memories not already in the block, <= second_pass_max_calls per turn)
Scoring is the scorecard's own (judge_one, 3 judges, majority, tie -> partly); nothing in it is changed.

usage: p18_eval.py STORE_COPY OUT.json
env:   P18_SET=tune|holdout (default tune)  P18_LIMIT=n (smoke only)  P18_ARMS=off,oldtool,on
"""
import json, os, re, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

store, out = sys.argv[1], sys.argv[2]
assert os.path.realpath(store) != os.path.realpath("/root/.mpmf/athena.mpmf"), "refusing the live store"
os.environ["HOME"] = os.environ.get("HOME") or "/root"
os.makedirs("/root/p18/health", exist_ok=True)
os.environ["MPMF_HEALTH_DIR"] = "/root/p18/health"
cfg = json.load(open("/root/.mpmf/recall.config.json"))
cfg["store_path"] = store
cfg["second_pass"] = False
cfg_path = "/root/p18/cfg.json"
json.dump(cfg, open(cfg_path, "w"), indent=1)
os.environ["MPMF_RECALL_CONFIG"] = cfg_path
sys.path.insert(0, "/root/mpmf-release/src")
sys.path.insert(0, "/root/.mpmf")
from mpmf_plus.eval import scorecard as sc          # noqa: E402
from mpmf_plus import provider as P                 # noqa: E402
from mpmf_plus import second_pass as SP             # noqa: E402
from mpmf_plus import hermes_adapter as H           # noqa: E402

sc.SNAPSHOT = store
provider, prov0 = sc.load_provider()
P._RECALL_CONFIG_PATH = cfg_path
P.configure(COGNITION_INJECT_LOG=type(P.COGNITION_INJECT_LOG)("/root/p18/health/cognition_inject.log"))

SETDIR = "/root/p6_sets/questions_p6_%s.json" % os.environ.get("P18_SET", "tune")
qs = json.load(open(SETDIR))
if os.environ.get("P18_LIMIT"):
    qs = qs[:int(os.environ["P18_LIMIT"])]
ARMS = os.environ.get("P18_ARMS", "off,oldtool,on").split(",")
norm = lambda s: re.sub(r"\W+", " ", (s or "").lower()).strip()
print("set", SETDIR, "n", len(qs), "arms", ARMS, flush=True)

TOOL_ADDENDUM = ("\nYou also have a tool, mpmf_recall, that searches the same memory. If the context does not "
                 "contain what you need, you may call it (at most twice) before answering. If neither the context "
                 "nor the tool results contain the answer, reply exactly DECLINED.\n")
MAX_ROUNDS = 3

# ---- prefetch once per question, one provider instance per question (turn state lives on it) ----
os.environ["MPMF_SECOND_PASS"] = "1"       # so prefetch() records which nodes the block carries
pre = []
for i, x in enumerate(qs):
    p = H.MPMFPlusProvider()
    p.initialize("p18-%d" % i)
    t = time.perf_counter()
    try:
        ctx, err = p.prefetch(x["q"]) or "", ""
    except Exception as e:
        ctx, err = "", repr(e)[:200]
    pre.append((p, ctx, (time.perf_counter() - t) * 1000, err))
    if (i + 1) % 20 == 0:
        print("prefetch %d/%d" % (i + 1, len(qs)), flush=True)


def chat(messages, tools, max_tokens=500):
    for attempt in range(4):
        try:
            body = {"model": sc.ANSWER_MODEL, "temperature": 0.0, "max_tokens": max_tokens, "messages": messages}
            if tools:
                body["tools"] = tools
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              headers={"Authorization": "Bearer " + sc.key()}, json=body, timeout=90)
            j = r.json()
            if "choices" not in j:
                raise RuntimeError(str(j)[:200])
            u = j.get("usage", {})
            pt, ct = int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
            pin, pout = sc.PRICE.get(sc.ANSWER_MODEL, (0.5, 2.0))
            with sc._LOCK:
                sc.COST["calls"] += 1
                sc.COST["usd"] += (pt * pin + ct * pout) / 1e6
            return j["choices"][0]["message"]
        except Exception:
            time.sleep(1 + attempt)
    return {"content": ""}


def answer(arm, x, p, ctx):
    """returns (answer, tool_log, extra_text_seen)"""
    prompt = sc.ANS_P.format(ctx=ctx[:6000], q=x["q"])
    if arm == "off":
        return sc.llm(sc.ANSWER_MODEL, prompt), [], ""
    os.environ["MPMF_SECOND_PASS"] = "1" if arm == "on" else "0"
    schema = [{"type": "function", "function": {
        "name": s["name"], "description": s["description"], "parameters": s["parameters"]}}
        for s in p.get_tool_schemas() if s["name"] == "mpmf_recall"]
    # the schema the model sees differs by arm only through get_tool_schemas() (flag read at call time)
    msgs = [{"role": "user", "content": prompt.replace("\nCONTEXT:", TOOL_ADDENDUM + "\nCONTEXT:", 1)}]
    log, seen = [], ""
    for rnd in range(MAX_ROUNDS + 1):
        m = chat(msgs, schema if rnd < MAX_ROUNDS else None)
        tcs = m.get("tool_calls") or []
        if not tcs:
            return (m.get("content") or "").strip(), log, seen
        msgs.append({"role": "assistant", "content": m.get("content"), "tool_calls": tcs})
        for tc in tcs:
            try:
                a = json.loads(tc["function"]["arguments"] or "{}")
            except Exception:
                a = {}
            t = time.perf_counter()
            os.environ["MPMF_SECOND_PASS"] = "1" if arm == "on" else "0"
            res = p.handle_tool_call(tc["function"]["name"], a)
            ms = (time.perf_counter() - t) * 1000
            seen += "\n" + str(res)
            log.append({"q": a.get("query"), "ms": round(ms, 1), "chars": len(str(res)),
                        "info": dict(SP.LAST) if arm == "on" else {}})
            msgs.append({"role": "tool", "tool_call_id": tc.get("id"), "content": str(res)})
    return "", log, seen


def work(a):
    arm, x, (p, ctx, pms, err) = a
    ans, log, seen = answer(arm, x, p, ctx)
    with ThreadPoolExecutor(3) as ex:
        vs = list(ex.map(lambda m: sc.judge_one(m, x["q"], x["key"], ans), sc.JUDGES))
    c = Counter(v for v in vs if v != "unjudged")
    v = c.most_common(1)[0][0] if c else "unjudged"
    if c and len(c) > 1 and c.most_common(2)[0][1] == c.most_common(2)[1][1]:
        v = "partly"
    src = norm(x["src_text"])[:60]
    return {"id": x["id"], "type": x["type"], "arm": arm, "verdict": v, "verdicts": vs,
            "tool_calls": len(log), "tool_log": log, "block_chars": len(ctx), "prefetch_ms": round(pms, 1),
            "src_in_block": src in norm(ctx), "src_in_tool": src in norm(seen),
            "answer": ans[:300], "err": err, "cost": round(sc.COST["usd"], 4)}


rows = []
for arm in ARMS:
    t0 = time.time()
    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(work, [(arm, x, pr) for x, pr in zip(qs, pre)]))
    rows += res
    acc = (sum(r["verdict"] == "correct" for r in res) + .5 * sum(r["verdict"] == "partly" for r in res)) / len(res)
    print(json.dumps({"arm": arm, "n": len(res), "acc": round(acc, 4), "strict": sum(r["verdict"] == "correct" for r in res),
                      "declined": sum(r["verdict"] == "declined" for r in res),
                      "tool_calls": sum(r["tool_calls"] for r in res), "secs": round(time.time() - t0),
                      "cost": round(sc.COST["usd"], 4)}), flush=True)
    if sc.COST["usd"] > 2.0:
        print("COST CAP hit; stopping", flush=True)
        break
json.dump(rows, open(out, "w"), indent=1)
sys.stdout.flush()
os._exit(0)
