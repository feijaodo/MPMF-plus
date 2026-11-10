#!/usr/bin/env python3
"""P9: keep_details before/after on real turns (REAL extraction + 3 blind judges).

Turns: real user messages from the p8 corpus (outside the repo, private). The sample is
fixed by SEED before anything runs. For each turn the FIRST extraction is shared between
the arms (cached), so the only difference is the keep_details retry.
  off = ingest(keep_details=False)   on = ingest(keep_details=True)
Judges (scorecard.JUDGES, blind, arms judged separately): specificity 1-5 and the number
of facts unsupported by / contradicting the message. Also a programmatic count of
specifics dropped.   usage: p9_eval.py <outdir> <mode: random|dense> [n]
"""
import hashlib, json, os, random, re, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.environ.get("MPMF_SRC", "/root/mpmf-release/src"))
from mpmf.memory import MemoryManager
from mpmf_plus import ingest as ing, keep_details as K
from mpmf_plus.eval import scorecard as sc

OUT, MODE = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
SEED = int(os.environ.get("P9_SEED", "909"))
CORPUS = "/root/p8/corpus/raw_items.jsonl"
DIM = 1024
os.makedirs(OUT, exist_ok=True)


def fake_vec(text):
    h = hashlib.sha256(text.encode()).digest()
    v = [((h[i % 32] / 255.0) - 0.5) + 1e-3 for i in range(DIM)]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def sample():
    items = [json.loads(l) for l in open(CORPUS)]
    byconv = {}
    for it in items:
        byconv.setdefault(it["conv"], []).append(it)
    cands = []
    for conv, L in byconv.items():
        L.sort(key=lambda x: x["idx"])
        for k, it in enumerate(L):
            t = it["text"]
            if not (80 <= len(t) <= 600):
                continue
            sp = K.specifics(t)
            ns = len(sp["proper"]) + len(sp["numbers"])
            cands.append((it, L[max(0, k - 2):k], ns))
    rng = random.Random(SEED)
    rng.shuffle(cands)
    if MODE == "dense":
        cands = [c for c in cands if c[2] >= 4]
    seen, out = set(), []
    for it, ctx, ns in cands:
        if it["conv"] in seen:
            continue
        seen.add(it["conv"])
        out.append((it, ctx, ns))
        if len(out) == N:
            break
    return out


JP = """Judge a memory extractor. It read ONE user message and wrote standalone fact sentences.
You do not know how they were produced.

USER MESSAGE: {msg}

FACTS:
{facts}

Reply JSON only: {{"specificity": <1-5: 5 = every name, brand, number, date, place of the message that is worth remembering is kept in the facts; 1 = generic, specifics lost>,
"unsupported": <number of facts that state something the message does not say or that contradicts it>}}"""


def judge(model, msg, texts):
    facts = "\n".join("- " + t for t in texts) or "- (no facts)"
    raw = sc.llm(model, JP.format(msg=msg[:1500], facts=facts), 150)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        j = json.loads(m.group(0))
        return [float(j["specificity"]), float(j["unsupported"])]
    except Exception:
        return None


class M:
    def __init__(self, mem):
        self.mem = mem

    def write_txn(self, timeout=60):
        return self.mem.write_txn(timeout=timeout)

    def embed(self, t):
        return fake_vec(t)


def run_turn(args):
    n, (it, ctx, ns) = args
    first = {}

    def call(body):
        if "REVIEW:" not in body:
            if "x" not in first:
                first["x"] = ing._default_call(body)
            return first["x"]
        return ing._default_call(body)

    res, kd = {}, None
    for arm, flag in (("off", False), ("on", True)):
        with tempfile.TemporaryDirectory() as d:
            mem = MemoryManager(os.path.join(d, "s.mpmf"), vector_dimensions=DIM)
            mem.load()
            r = ing.ingest(it["text"], conv_id="p9_%03d" % n, store=M(mem), call_model=call,
                           embed_fn=fake_vec, log_path="", embed_dim=DIM, keep_details=flag,
                           context=[{"role": "user", "text": c["text"]} for c in ctx])
        res[arm] = [w["text"] for w in r["written"]]
        if arm == "on":
            kd = r["keep_details"]
    msg = it["text"]
    row = {"n": n, "id": "%s:%s" % (it["conv"], it["idx"]), "specifics_in_msg": ns, "kd": kd,
           "msg": msg, "facts": res, "changed": res["off"] != res["on"],
           "dropped": {a: sum(len(K.missing_specifics(msg, t)) for t in res[a]) for a in res}}
    row["judge"] = {}
    with ThreadPoolExecutor(3) as ex:
        row["judge"]["off"] = list(ex.map(lambda m: judge(m, msg, res["off"]), sc.JUDGES))
        # identical facts => identical verdicts (re-judging only adds judge noise)
        row["judge"]["on"] = row["judge"]["off"] if not row["changed"] else \
            list(ex.map(lambda m: judge(m, msg, res["on"]), sc.JUDGES))
    return row


def safe_turn(a):
    try:
        return run_turn(a)
    except Exception as exc:  # noqa: BLE001 — an unparseable first extraction fails both arms alike
        return {"n": a[0], "error": "%s: %s" % (type(exc).__name__, exc)}


def main():
    S = sample()
    print("sample", len(S), "mode", MODE, "specifics/turn", [s[2] for s in S], flush=True)
    rows = []
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(safe_turn, list(enumerate(S))):
            rows.append(r)
    errs = [r for r in rows if "error" in r]
    print("extraction failures (turn skipped in both arms):", len(errs), [e["error"][:60] for e in errs])
    rows = [r for r in rows if "error" not in r]
    json.dump(rows, open(os.path.join(OUT, "rows_%s.json" % MODE), "w"), indent=1)
    print("cost", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in sc.COST.items()}))


main()
