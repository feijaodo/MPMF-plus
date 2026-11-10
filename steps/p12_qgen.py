#!/usr/bin/env python3
"""P12: two-fact questions for the typed-link walk. Built from the STORE COPY's typed edges.

usage: p12_qgen.py <store_copy.mpmf> <outdir>
Pairs: DERIVED_FROM (older fact + the state built from it) and supersedes (memory<->memory).
IS_A links a state to a content-free `category` hub, so it yields no fact pair (reported).
The writer sees ONLY the two node texts. A question is kept when the writer says the two facts
combine into one question. Eligibility ("plain cosine does not retrieve both"): NOT both nodes in the
cosine top-8 of the question over the memory namespace (walk and rerank play no part in it).
Eligible questions (up to 40) form the MAIN set; the rest are the CONTROL set (cosine already gets both).
Nothing here looks at either arm's answers. Output is private chat content: chmod 700 dir, outside the repo.
"""
import json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor

store, outdir = sys.argv[1], sys.argv[2]
assert os.path.realpath(store) != os.path.realpath("/root/.mpmf/athena.mpmf"), "refusing the live store"
os.makedirs(outdir, exist_ok=True)
os.environ["HOME"] = os.environ.get("HOME") or "/root"
os.makedirs("/root/p12/health", exist_ok=True)
os.environ["MPMF_HEALTH_DIR"] = "/root/p12/health"
cfg = json.load(open("/root/.mpmf/recall.config.json"))
cfg["store_path"] = store
cfg["typed_walk"] = False
cfg_path = "/root/p12/cfg_gen.json"
json.dump(cfg, open(cfg_path, "w"), indent=1)
os.environ["MPMF_RECALL_CONFIG"] = cfg_path
sys.path.insert(0, os.environ.get("MPMF_SRC", "/root/mpmf-release/src"))
sys.path.insert(0, "/root/.mpmf")
import numpy as np                                  # noqa: E402
from mpmf_plus.eval import scorecard as sc          # noqa: E402
from mpmf_plus import provider as P                 # noqa: E402

P.configure(PATH=store)
P._invalidate_open_cache()
mem = P._open().mem
labels = mem._edge_labels
ns = lambda n: (mem.get_wiki(n) or {}).get("_ns")
txt = lambda n: " ".join((mem.get_text(n) or "").split())
rng = random.Random(1212)

df, sup, isa = [], [], 0
for e in mem._edges:
    L = labels.get(e.edge_id)
    a, b = e.source_node_id, e.target_node_id
    if L == "IS_A":
        isa += 1
    elif L == "DERIVED_FROM" and ns(a) == "memory" and ns(b) == "memory":
        # a = state, b = fact it was derived from. Skip the fact that IS the state's own source.
        w = mem.get_wiki(a) or {}
        if len(txt(b)) >= 40 and txt(b)[:60] not in txt(a):
            df.append((b, a, "DERIVED_FROM"))
    elif L == "supersedes" and ns(a) == "memory" and ns(b) == "memory":
        wa, wb = mem.get_wiki(a) or {}, mem.get_wiki(b) or {}
        if wa.get("_superseded_by") or wb.get("_superseded_by"):
            continue
        if len(txt(a)) >= 40 and len(txt(b)) >= 40:
            sup.append((a, b, "supersedes"))
rng.shuffle(df); rng.shuffle(sup)
cand = df[:90] + sup[:70]
print("pool: DERIVED_FROM %d, supersedes %d, IS_A edges %d (content-free hubs, no pairs); sampled %d" %
      (len(df), len(sup), isa, len(cand)), flush=True)

WRITER = "deepseek/deepseek-v4.1-flash"
PROMPT = """You write test questions for a personal-memory system. You see two stored facts about one user.

FACT 1: {a}
FACT 2: {b}

If the two facts do not concern the same subject, or cannot be combined into ONE sensible question,
reply exactly {{"skip": true}}.
Otherwise write ONE natural question the user might ask later whose answer REQUIRES BOTH facts (an answer
from either fact alone must be incomplete). Do not copy long phrases from the facts; do not mention
"fact 1/2" or "stored". Give the answer key as one or two sentences containing what each fact contributes.
Reply with JSON only: {{"q": "...", "key": "...", "a_part": "<what fact 1 contributes>", "b_part": "<what fact 2 contributes>"}}"""


def write(c):
    a, b, L = c
    raw = sc.llm(WRITER, PROMPT.format(a=txt(a)[:500], b=txt(b)[:500]), 400)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except Exception:
        j = {}
    if not j.get("q") or not j.get("key") or j.get("skip"):
        return None
    return {"q": j["q"].strip(), "key": j["key"].strip(), "a_part": j.get("a_part", ""), "b_part": j.get("b_part", ""),
            "node_a": int(a), "node_b": int(b), "link": L, "text_a": txt(a)[:500], "text_b": txt(b)[:500]}


with ThreadPoolExecutor(6) as ex:
    qs = [x for x in ex.map(write, cand) if x]
print("writer kept %d of %d  ($%.3f)" % (len(qs), len(cand), sc.COST["usd"]), flush=True)

# cosine top-8 over the memory namespace, superseded rows masked (exactly d20's candidate pool)
ids, M, norms, keep = P._d20_node_matrix(mem)
allowed = np.array([ns(i) == "memory" for i in ids]) & keep


def cos_top(qv, n=8):
    q = np.asarray(qv, dtype=np.float32)
    s = (M @ q) / (norms * (float(np.linalg.norm(q)) or 1.0) + 1e-9)
    s = np.where(allowed, s, -np.inf)
    o = np.argsort(-s)[:n]
    return [int(ids[i]) for i in o]


seen = set()
for x in qs:
    qv = P.embed_query(x["q"])
    top = cos_top(qv, 8)
    x["cos_top8"] = top
    x["a_in_top8"], x["b_in_top8"] = x["node_a"] in top, x["node_b"] in top
    x["eligible"] = not (x["a_in_top8"] and x["b_in_top8"])
qs = [x for x in qs if not (x["q"].lower() in seen or seen.add(x["q"].lower()))]
elig = [x for x in qs if x["eligible"]]
ctrl = [x for x in qs if not x["eligible"]]
# main = up to 40 eligible, DERIVED_FROM and supersedes interleaved so neither link type is crowded out
byL = {L: [x for x in elig if x["link"] == L] for L in ("DERIVED_FROM", "supersedes")}
main = []
while len(main) < 40 and (byL["DERIVED_FROM"] or byL["supersedes"]):
    for L in ("DERIVED_FROM", "supersedes"):
        if byL[L] and len(main) < 40:
            main.append(byL[L].pop(0))
for i, x in enumerate(main):
    x["id"] = "m%02d" % i
for i, x in enumerate(ctrl[:20]):
    x["id"] = "c%02d" % i
json.dump(main, open(os.path.join(outdir, "questions_p12_main.json"), "w"), indent=1)
json.dump(ctrl[:20], open(os.path.join(outdir, "questions_p12_control.json"), "w"), indent=1)
print(json.dumps({"written": len(qs), "eligible": len(elig), "main": len(main),
                  "main_by_link": {L: sum(x["link"] == L for x in main) for L in ("DERIVED_FROM", "supersedes")},
                  "control": len(ctrl[:20]), "one_of_two_in_top8_main": sum(x["a_in_top8"] != x["b_in_top8"] for x in main),
                  "neither_in_top8_main": sum(not x["a_in_top8"] and not x["b_in_top8"] for x in main),
                  "cost": round(sc.COST["usd"], 4)}), flush=True)
os._exit(0)
