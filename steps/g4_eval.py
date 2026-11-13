#!/usr/bin/env python3
"""G4: second-pass retrieval, pre-registered confirm on HOLDOUT (74 q), one trial per question per arm.

Derived from steps/p18_eval.py (same prompt, tool addendum, answer model, 3 judges, majority, tie->partly);
differences: HOLDOUT only (md5 asserted), arms oldtool (= OFF-live), on, off (= no tool, secondary),
rows appended per (arm, id) to a jsonl so a crash resumes only what did not run, a prefetch pass with the
flag ON (timed, feeds all arms) and a second prefetch pass with the flag OFF (timed + block compared),
per-question answer wall time, error counters.

usage: g4_eval.py STORE_COPY WORKDIR      (WORKDIR holds cfg.json, health/, rows.jsonl, prefetch.json)
"""
import hashlib, json, os, re, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

store, work_dir = sys.argv[1], sys.argv[2]
assert os.path.realpath(store) != os.path.realpath("/root/.mpmf/athena.mpmf"), "refusing the live store"
assert os.path.realpath(store).startswith("/root/g_work/g4/copies/"), "store must be a copy under g_work/g4/copies"
SMOKE = os.environ.get("G4_SMOKE") == "1"      # plumbing check on TUNE (3 questions); never on HOLDOUT
HOLDOUT = "/root/p6_sets/questions_p6_%s.json" % ("tune" if SMOKE else "holdout")
if not SMOKE:
    assert hashlib.md5(open(HOLDOUT, "rb").read()).hexdigest() == "770fce0189dee468863c8e80a86f4a36", "holdout md5 changed"
SPEND_CEIL = 3.2            # 80% of the $4 cap, this process only

os.makedirs(work_dir + "/health", exist_ok=True)
os.environ["MPMF_HEALTH_DIR"] = work_dir + "/health"
os.environ["MPMF_THINKING_STORE"] = os.path.dirname(store) + "/athena-thinking.mpmf"   # a copy, never the live one
assert os.path.exists(os.environ["MPMF_THINKING_STORE"])
cfg = json.load(open("/root/.mpmf/recall.config.json"))
cfg["store_path"] = store
cfg["second_pass"] = False
cfg_path = work_dir + "/cfg.json"
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
P.configure(COGNITION_INJECT_LOG=type(P.COGNITION_INJECT_LOG)(work_dir + "/health/cognition_inject.log"))

qs = json.load(open(HOLDOUT))
ARMS = os.environ.get("G4_ARMS", "oldtool,on,off").split(",")
if SMOKE:
    qs = qs[:3]
norm = lambda s: re.sub(r"\W+", " ", (s or "").lower()).strip()
print("set", HOLDOUT, "n", len(qs), "arms", ARMS, flush=True)

TOOL_ADDENDUM = ("\nYou also have a tool, mpmf_recall, that searches the same memory. If the context does not "
                 "contain what you need, you may call it (at most twice) before answering. If neither the context "
                 "nor the tool results contain the answer, reply exactly DECLINED.\n")
MAX_ROUNDS = 3
ERR = Counter()

# ---- prefetch passes (sequential, single-flight, one provider instance per question) ----
PF = work_dir + "/prefetch.json"
if os.path.exists(PF):
    pf = json.load(open(PF))
else:
    pf = {}


def prefetch_pass(flag):
    os.environ["MPMF_SECOND_PASS"] = "1" if flag else "0"
    out = []
    provs = []
    for i, x in enumerate(qs):
        p = H.MPMFPlusProvider()
        p.initialize("g4-%s-%d" % ("on" if flag else "off", i))
        t = time.perf_counter()
        try:
            ctx, err = p.prefetch(x["q"]) or "", ""
        except Exception as e:
            ctx, err = "", repr(e)[:200]
        out.append({"ms": (time.perf_counter() - t) * 1000, "ctx": ctx, "err": err})
        provs.append(p)
        if (i + 1) % 20 == 0:
            print("prefetch(%s) %d/%d" % ("on" if flag else "off", i + 1, len(qs)), flush=True)
    return provs, out


# ON pass first: any cold-cache cost falls on the arm under test, not in its favour.
provs, on_pass = prefetch_pass(True)
_, off_pass = prefetch_pass(False)
pf = {"on_ms": [round(r["ms"], 1) for r in on_pass], "off_ms": [round(r["ms"], 1) for r in off_pass],
      "on_err": [r["err"] for r in on_pass], "off_err": [r["err"] for r in off_pass],
      "block_equal": [a["ctx"] == b["ctx"] for a, b in zip(on_pass, off_pass)],
      "block_chars": [len(r["ctx"]) for r in on_pass]}
json.dump(pf, open(PF, "w"))
pre = [(p, r["ctx"], r["ms"], r["err"]) for p, r in zip(provs, on_pass)]   # blocks shared by all arms


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
        except requests.exceptions.Timeout:
            ERR["answer_timeout"] += 1
            time.sleep(1 + attempt)
        except Exception:
            ERR["answer_error"] += 1
            time.sleep(1 + attempt)
    ERR["answer_gave_up"] += 1
    return {"content": ""}


def answer(arm, x, p, ctx):
    prompt = sc.ANS_P.format(ctx=ctx[:6000], q=x["q"])
    if arm == "off":
        return sc.llm(sc.ANSWER_MODEL, prompt), [], ""
    flag = "1" if arm == "on" else "0"
    os.environ["MPMF_SECOND_PASS"] = flag
    schema = [{"type": "function", "function": {
        "name": s["name"], "description": s["description"], "parameters": s["parameters"]}}
        for s in p.get_tool_schemas() if s["name"] == "mpmf_recall"]
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
            os.environ["MPMF_SECOND_PASS"] = flag
            terr = ""
            try:
                res = p.handle_tool_call(tc["function"]["name"], a)
            except Exception as e:
                res, terr = "tool error", repr(e)[:200]
                ERR["tool_exception"] += 1
            ms = (time.perf_counter() - t) * 1000
            seen += "\n" + str(res)
            log.append({"q": a.get("query"), "ms": round(ms, 1), "chars": len(str(res)), "err": terr,
                        "info": dict(SP.LAST) if arm == "on" else {}})
            msgs.append({"role": "tool", "tool_call_id": tc.get("id"), "content": str(res)})
    return "", log, seen


def work(a):
    arm, i, x, (p, ctx, pms, err) = a
    if sc.COST["usd"] > SPEND_CEIL:
        return None
    t0 = time.perf_counter()
    ans, log, seen = answer(arm, x, p, ctx)
    ans_ms = (time.perf_counter() - t0) * 1000
    with ThreadPoolExecutor(3) as ex:
        vs = list(ex.map(lambda m: sc.judge_one(m, x["q"], x["key"], ans), sc.JUDGES))
    ERR["judge_unjudged"] += sum(v == "unjudged" for v in vs)
    c = Counter(v for v in vs if v != "unjudged")
    v = c.most_common(1)[0][0] if c else "unjudged"
    if c and len(c) > 1 and c.most_common(2)[0][1] == c.most_common(2)[1][1]:
        v = "partly"
    src = norm(x["src_text"])[:60]
    row = {"id": x["id"], "type": x["type"], "arm": arm, "verdict": v, "verdicts": vs,
           "tool_calls": len(log), "tool_log": log, "tool_ms_total": round(sum(l["ms"] for l in log), 1),
           "block_chars": len(ctx), "prefetch_ms": round(pms, 1), "answer_ms": round(ans_ms, 1),
           "src_in_block": src in norm(ctx), "src_in_tool": src in norm(seen),
           "empty_answer": not ans.strip(), "err": err, "cost": round(sc.COST["usd"], 4)}
    with open(work_dir + "/rows.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


done = set()
if os.path.exists(work_dir + "/rows.jsonl"):
    for line in open(work_dir + "/rows.jsonl"):
        r = json.loads(line)
        done.add((r["arm"], r["id"]))
    print("RESUME: %d (arm,id) rows already done" % len(done), flush=True)

for arm in ARMS:
    t0 = time.time()
    todo = [(arm, i, x, pr) for i, (x, pr) in enumerate(zip(qs, pre)) if (arm, x["id"]) not in done]
    with ThreadPoolExecutor(3) as ex:
        res = [r for r in ex.map(work, todo) if r]
    print(json.dumps({"arm": arm, "ran": len(res), "of": len(todo), "secs": round(time.time() - t0),
                      "cost_this_process": round(sc.COST["usd"], 4), "errors": dict(ERR)}), flush=True)
    if sc.COST["usd"] > SPEND_CEIL:
        print("SPEND CEILING hit; stopping", flush=True)
        break
json.dump({"errors": dict(ERR), "cost_this_process": round(sc.COST["usd"], 4)}, open(work_dir + "/summary.json", "w"))
sys.stdout.flush()
os._exit(0)
