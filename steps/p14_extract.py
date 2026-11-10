#!/usr/bin/env python3
"""P14 (RESEARCH): functional extraction pilot -> typed USED_FOR / PART_OF edges on a STORE COPY.

usage: p14_extract.py <store_copy.mpmf> <outdir>

* Refuses any path that is a live store (athena/espy/ayu/shared) or sits outside /root/p14.
* Picks 100 conversations at random (seed 1414) from the copy: memory-ns `fact` nodes carrying a
  `_conv_id`, conversations with 8..60 facts. Each fact is shown with the user message it came
  from (`_evidence`, clipped).
* The extractor (deepseek-v4.1-flash, temp 0) lists pairs (rel, fact a, fact b): USED_FOR = a is the means, b the purpose;
  PART_OF = a is the part, b the whole, where the user's own words state or clearly imply it.
* Each accepted pair becomes one edge in the copy: Plane.FUNCTIONAL, relation_label USED_FOR / PART_OF,
  source = means/part node, target = purpose/whole node. De-duplicated, no self loops, both ends in the same conversation.
* Output: <outdir>/edges.jsonl (private chat content: outside the repo, chmod 700), extract_summary.json.
Nothing is written to any live store (the live file's mtime/size are checked before and after).
"""
import json, os, random, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

store, outdir = sys.argv[1], sys.argv[2]
rp = os.path.realpath(store)
assert rp.startswith("/root/p14/"), "store copy must live under /root/p14: %s" % rp
LIVE = "/root/.mpmf/athena.mpmf"
assert rp != os.path.realpath(LIVE), "refusing the live store"
os.makedirs(outdir, exist_ok=True)
os.chmod(outdir, 0o700)
os.makedirs("/root/p14/health", exist_ok=True)
os.environ["MPMF_HEALTH_DIR"] = "/root/p14/health"
sys.path.insert(0, os.environ.get("MPMF_SRC", "/root/mpmf-release/src"))
from mpmf_plus.eval import scorecard as sc   # noqa: E402
from mpmf_plus import provider as P          # noqa: E402
from mpmf.format import Plane                # noqa: E402

live_before = (os.stat(LIVE).st_mtime_ns, os.stat(LIVE).st_size)
P.configure(PATH=store)
P._invalidate_open_cache()
mem = P._open().mem
txt = lambda n: " ".join((mem.get_text(n) or "").split())

EXTRACTOR = "deepseek/deepseek-v4.1-flash"
N_CONV, MIN_F, MAX_F, SEED = 100, 8, 60, 1414

conv = defaultdict(list)
for n in mem.all_node_ids():
    w = mem.get_wiki(n) or {}
    if w.get("_ns") == "memory" and w.get("kind") == "fact" and w.get("_conv_id"):
        conv[w["_conv_id"]].append(n)
pool = sorted(c for c, v in conv.items() if MIN_F <= len(v) <= MAX_F)
rng = random.Random(SEED)
chosen = rng.sample(pool, N_CONV)
print("conversations with facts: %d, eligible (%d..%d facts): %d, chosen %d" %
      (len(conv), MIN_F, MAX_F, len(pool), len(chosen)), flush=True)

PROMPT = """Below are facts extracted from ONE conversation between a user and an assistant. Each fact has an id,
the fact, and (after ">>") the user's own message it came from.

{facts}

Find FUNCTIONAL relations between two of these facts. Two kinds only:
- USED_FOR: fact A says the user uses / will use / intends to use some thing, tool, method or resource (A) for a
  purpose or task (B), and fact B states that purpose or task.  A = the means, B = the purpose.
- PART_OF: fact A describes a component / member / step / section of something (A), and fact B describes the
  whole it belongs to (B). A = the part, B = the whole.
The user's words must state or clearly imply the relation ("used for", "to", "for", "so I can", "part of", "belongs to",
"is a module of", "step 2 of"...). Do NOT output: two facts that merely share a topic, sequence, a fact and its own
restatement, a request and its follow-up, a cause/effect (that is a different relation), a fact about the same item
stated twice, or anything where you must guess.
Be strict: an empty list is a normal answer. At most 6 pairs.

Reply with JSON only:
{{"pairs": [{{"rel": "USED_FOR|PART_OF", "a": <id>, "b": <id>, "basis": "<short phrase of the user's words that supports it>", "explicit": true|false}}]}}"""


def extract(cid):
    ids = conv[cid][:MAX_F]
    lines = []
    for n in ids:
        w = mem.get_wiki(n) or {}
        ev = " ".join(str(w.get("_evidence") or "").split())[:240]
        lines.append("[%d] %s  >> %s" % (n, txt(n)[:240], ev))
    raw = sc.llm(EXTRACTOR, PROMPT.format(facts="\n".join(lines)), 900) or ""
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except Exception:
        j = {}
    ok = set(ids)
    out, seen = [], set()
    for p in (j.get("pairs") or [])[:6]:
        try:
            a, b = int(p["a"]), int(p["b"])
            rel = str(p["rel"]).strip().upper().replace(" ", "_")
        except Exception:
            continue
        if rel not in ("USED_FOR", "PART_OF") or a == b or a not in ok or b not in ok or (a, b, rel) in seen:
            continue
        seen.add((a, b, rel))
        out.append({"conv": cid, "rel": rel, "cause": a, "effect": b, "basis": str(p.get("basis", ""))[:200],
                    "explicit": bool(p.get("explicit")),
                    "cause_text": txt(a)[:300], "effect_text": txt(b)[:300],
                    "cause_ev": str((mem.get_wiki(a) or {}).get("_evidence") or "")[:300],
                    "effect_ev": str((mem.get_wiki(b) or {}).get("_evidence") or "")[:300],
                    "conv_title": str((mem.get_wiki(a) or {}).get("_conv_title") or "")[:80]})
    print("conv %s facts=%d raw_len=%d pairs=%d" % (cid, len(ids), len(raw), len(out)), flush=True)
    return {"conv": cid, "n_facts": len(ids), "raw_ok": bool(j), "pairs": out}


with ThreadPoolExecutor(6) as ex:
    res = list(ex.map(extract, chosen))
edges = [e for r in res for e in r["pairs"]]
print("extractor: parsed %d/%d, convs with >=1 edge %d, edges %d  ($%.3f)" %
      (sum(r["raw_ok"] for r in res), len(res), sum(1 for r in res if r["pairs"]), len(edges), sc.COST["usd"]), flush=True)

# write the typed edges into the COPY only
n0 = len(mem._edges)
with mem.write_txn():
    for k, e in enumerate(edges):
        ed = mem.create_edge(source_node_id=e["cause"], target_node_id=e["effect"], plane=Plane.FUNCTIONAL,
                             relation_label=e["rel"], weight=1.0)
        e["edge_id"] = ed.edge_id
        e["i"] = k
mem2 = P._open().mem
causes = sum(1 for eid, L in mem._edge_labels.items() if L in ("USED_FOR", "PART_OF"))
print("edges before %d, after %d, USED_FOR/PART_OF in copy %d" % (n0, len(mem._edges), causes), flush=True)

with open(os.path.join(outdir, "edges.jsonl"), "w") as f:
    for e in edges:
        f.write(json.dumps(e) + "\n")
live_after = (os.stat(LIVE).st_mtime_ns, os.stat(LIVE).st_size)
summ = {"conversations": N_CONV, "eligible_pool": len(pool), "parsed": sum(r["raw_ok"] for r in res),
        "convs_with_edge": sum(1 for r in res if r["pairs"]), "edges": len(edges),
        "explicit": sum(e["explicit"] for e in edges), "facts_in_chosen": sum(r["n_facts"] for r in res),
        "functional_edges_in_copy": causes, "used_for": sum(e["rel"]=="USED_FOR" for e in edges), "part_of": sum(e["rel"]=="PART_OF" for e in edges), "extractor": EXTRACTOR, "cost_usd": round(sc.COST["usd"], 4),
        "live_unchanged": live_before == live_after, "copy": rp}
json.dump(summ, open(os.path.join(outdir, "extract_summary.json"), "w"), indent=1)
print(json.dumps(summ))
# live_unchanged is informational: the running gateway appends to the live store on its own; this script only
# ever opens the copy under /root/p14 (asserted at the top), so a live change is not caused by it.
