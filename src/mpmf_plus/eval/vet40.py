#!/usr/bin/env python3
"""Round 40 / Item 3 — vetting screen. Judge LESSONS (P2 pool nodes).

r98: moved from ~/.mpmf/gate_eval/vet40.py into the package, so the judge prompt the
intake gate and cognition use is SHIPPED code (a deploy gate can test it). The gate_eval
copy is now a shim; `python3 -m mpmf_plus.eval.vet40 <phase>` runs from the package.

Output: gate_eval/vet40.json (+ vet40_candidates.json snapshot). NO store writes.
Never prints the API key. Per-model spend stops enforced in-script.
Phases: messages | snapshot | pilot | stage1 | stage2 | clusters | report
"""
import argparse, hashlib, json, os, random, re, sys, time, urllib.request
from pathlib import Path

from .. import paths as _paths

#: $MPMF_GATE_EVAL, else $MPMF_HOME/gate_eval — no literal home path (name gate).
GE = _paths.gate_eval_dir()
SYS = ("You review a lesson written by an AI agent's self-reflection process, to decide "
       "whether it should be injected into future prompts. Judge only the lesson text. "
       "Reply with ONLY JSON: {\"form\": \"principle\"|\"session_fragment\"|\"code_trivia\"|\"other\", "
       "\"trigger_clear\": true|false, \"sound\": true|false, \"overbroad\": true|false, "
       "\"reason\": \"<= 20 words\"}. form=principle means a reusable rule with an identifiable "
       "trigger. session_fragment means it merely records what happened in one session. "
       "code_trivia means it is only about one file, line or variable. trigger_clear: the "
       "situation it applies to can be identified from the text. sound: a careful senior "
       "engineer would endorse following it as written when its trigger applies. overbroad: "
       "following it outside its trigger would make replies worse (e.g. bans a language or "
       "tool, forces needless tool calls, adds ritual).")
MODELS = {"glm-5.3-flash": "z-ai/glm-5.3-flash",
          "deepseek-v4.1-flash-1017": "deepseek/deepseek-v4.1-flash-1017",
          "sonnet-5.5": "anthropic/claude-sonnet-5.5",
          "gpt-6.1-sol": "openai/gpt-6.1-sol",
          "gemini-4-argon": "google/gemini-4-argon"}
CAP = {"glm-5.3-flash": 0.15, "deepseek-v4.1-flash-1017": 0.40,
       "sonnet-5.5": 1.30, "gpt-6.1-sol": 1.10, "gemini-4-argon": 1.10}
TOTAL_CAP = 4.00
KEY = os.environ.get("OPENROUTER_API_KEY", "")
NS = "vet40"

# ---------- store (copy) ----------
def load_store():
    sys.path.insert(0, str(_paths.home()))
    import athena_memory as mod
    mod.PATH = str(GE / "athena_census40.mpmf"); mod._OPEN_CACHE.clear()
    return mod, mod._open().mem

def candidates(mod, mm):
    f = GE / "vet40_candidates.json"
    if f.exists():
        return json.loads(f.read_text())
    out = []
    for nid in mod._sit_pool_ids(mm):
        t = mm.get_text(nid) or ""
        out.append({"id": int(nid), "name": mm.get_name(nid), "sha1": hashlib.sha1(t.encode()).hexdigest(),
                    "text": t, "status": mod._lesson_status(mm, nid)})
    f.write_text(json.dumps(out, indent=1))
    return out

# ---------- cache ----------
def cache_path(alias): return GE / ("vet40_cache_%s.json" % alias.replace("/", "_").replace(".", "_"))
def load_cache(alias):
    p = cache_path(alias)
    return json.loads(p.read_text()) if p.exists() else {}
def save_cache(alias, c):
    p = cache_path(alias); tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(c, indent=1)); os.replace(tmp, p)   # merge-write, never clobber keys

# ---------- model call ----------
def call(alias, user_msg, max_tokens=2000):
    body = {"model": MODELS[alias], "temperature": 0, "max_tokens": max_tokens,
            "reasoning": {"effort": "minimal"}, "usage": {"include": True},
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": user_msg}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + KEY,
        "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.loads(r.read().decode())
    txt = (d.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    cost = float((d.get("usage") or {}).get("cost") or 0.0)
    return txt, cost

def parse_json(t):
    t = (t or "").strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

def is_keep(v):
    return bool(v and v.get("form") == "principle" and v.get("trigger_clear")
                and v.get("sound") and not v.get("overbroad"))

def judge(alias, text, spend):
    txt, cost = call(alias, "LESSON:\n" + text[:700] + "\n\nJSON now.")
    spend[alias] = spend.get(alias, 0.0) + cost
    v = parse_json(txt)
    if v is None:
        return {"parse_fail": True, "raw": txt[:200], "keep": None}
    v["keep"] = is_keep(v)
    return v

def check_spend(spend):
    return {a: round(v, 4) for a, v in spend.items() if v > CAP[a]}

# ---------- phases ----------
def phase_messages(mod, mm):
    cands = candidates(mod, mm)
    rnd = random.Random(40); picks = rnd.sample(cands, 2)
    print("=== item 3b: exact judge message for 2 random nodes ===")
    for c in picks:
        print("---- node id=%d name=%s sha1=%s ----" % (c["id"], c["name"], c["sha1"]))
        print("[SYSTEM]\n%s\n" % SYS)
        print("[USER]\nLESSON:\n%s\n\nJSON now.\n" % c["text"][:700])
    print("=== item 3f: layout of 3 real nodes (field order + ' | ' separators) ===")
    for c in cands[:3]:
        print("---- id=%d name=%s ----" % (c["id"], c["name"]))
        print(repr(c["text"][:400]))
    print("candidate count:", len(cands))

def phase_snapshot(mod, mm):
    c = candidates(mod, mm)
    print("candidates snapshot: %d nodes -> %s" % (len(c), GE / "vet40_candidates.json"))
    print("status counts:", {s: sum(1 for x in c if x["status"] == s) for s in set(x["status"] for x in c)})

def phase_pilot(mod, mm):
    cands = candidates(mod, mm); rnd = random.Random(40); picks = rnd.sample(cands, 6)
    spend = {}
    for a in MODELS:
        print("== pilot %s ==" % a)
        for c in picks:
            v = judge(a, c["text"], spend)
            print("  id=%d keep=%r form=%r tc=%r snd=%r ob=%r reason=%r"
                  % (c["id"], v.get("keep"), v.get("form"), v.get("trigger_clear"),
                     v.get("sound"), v.get("overbroad"), v.get("reason")))
        print("  spend=%s" % round(spend.get(a, 0), 4))

def _run_models(mod, mm, aliases, node_ids=None):
    cands = {c["id"]: c for c in candidates(mod, mm)}
    ids = node_ids or list(cands)
    spend = {}
    for a in aliases:
        cache = load_cache(a); n = 0
        for nid in ids:
            c = cands[nid]
            key = str(nid)
            if key in cache and cache[key].get("sha1") == c["sha1"]:
                continue
            if spend.get(a, 0.0) > CAP[a]:
                print("[%s] STOPPED at cap $%.2f after %d calls" % (a, CAP[a], n)); break
            v = judge(a, c["text"], spend)
            v["sha1"] = c["sha1"]
            cache[key] = v; n += 1
            if n % 50 == 0:
                save_cache(a, cache)
                print("  [%s] %d done spend=$%.4f" % (a, n, spend.get(a, 0.0)))
        save_cache(a, cache)
        print("[%s] calls=%d cache=%d spend=$%.4f" % (a, n, len(cache), spend.get(a, 0.0)))
    return spend

def phase_stage1(mod, mm):
    print(_run_models(mod, mm, ["glm-5.3-flash", "deepseek-v4.1-flash-1017"]))

def phase_stage2(mod, mm):
    cands = {c["id"]: c for c in candidates(mod, mm)}
    c1 = load_cache("glm-5.3-flash"); c2 = load_cache("deepseek-v4.1-flash-1017")
    disagree, agree = [], []
    for nid in cands:
        k1 = c1.get(str(nid), {}).get("keep"); k2 = c2.get(str(nid), {}).get("keep")
        if k1 is None or k2 is None: continue
        (disagree if k1 != k2 else agree).append(nid)
    rnd = random.Random(40); audit = rnd.sample(agree, max(1, len(agree)//10))
    target = sorted(set(disagree) | set(audit))
    print("disagree=%d agree=%d audit=%d -> panel on %d nodes" % (len(disagree), len(agree), len(audit), len(target)))
    spend = _run_models(mod, mm, ["sonnet-5.5", "gpt-6.1-sol", "gemini-4-argon"], target)
    (GE / "vet40_panel_targets.json").write_text(json.dumps({"disagree": disagree, "audit": audit}))
    print(spend)

def phase_clusters(mod, mm):
    import numpy as np
    cands = candidates(mod, mm)
    ids = [c["id"] for c in cands]
    vecs = []
    for nid in ids:
        v = mm.get_vector(nid)
        vecs.append(v)
    X = np.asarray([np.asarray(v, dtype=np.float32) for v in vecs])
    n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = n @ n.T
    parent = list(range(len(ids)))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if S[i, j] >= 0.90:
                ri, rj = find(i), find(j)
                if ri != rj: parent[ri] = rj
    clusters = {}
    for i in range(len(ids)): clusters.setdefault(find(i), []).append(i)
    multi = {k: v for k, v in clusters.items() if len(v) > 1}
    sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    print("nodes=%d clusters=%d multi-member=%d" % (len(ids), len(clusters), len(multi)))
    print("size distribution (top 15):", sizes[:15])
    big = sorted(multi.values(), key=len, reverse=True)[:5]
    print("5 biggest clusters:")
    for v in big:
        rep = cands[v[0]]
        print("  size=%d rep_id=%d rep_text=%r" % (len(v), rep["id"], rep["text"][:200]))
    (GE / "vet40_clusters.json").write_text(json.dumps(
        {"clusters": len(clusters), "multi": len(multi), "sizes": sizes[:50],
         "biggest": [[cands[i]["id"] for i in v] for v in big]}, indent=1))

def phase_controls(mod, mm):
    cands = {c["id"]: c for c in candidates(mod, mm)}
    c1 = load_cache("glm-5.3-flash"); c2 = load_cache("deepseek-v4.1-flash-1017")
    def both(nid):
        return [c.get(str(nid), {}) for c in (c1, c2)]
    print("=== (i) POSITIVE control: P2 nodes with status 'verified' ===")
    ver = [c for c in cands.values() if c["status"] == "verified"]
    print("verified P2 nodes: %d" % len(ver))
    kk = 0
    for c in ver:
        vs = both(c["id"])
        keeps = [v.get("keep") for v in vs]
        kk += sum(1 for k in keeps if k)
        print("  id=%d keeps=%s reasons=%s" % (c["id"], keeps, [v.get("reason") for v in vs]))
    print("  positive KEEP rate = %d/%d" % (kk, 2 * len(ver)) if ver else "  (none)")
    print("\n=== (ii) NEGATIVE control: text matches ===")
    for pat in ("Chinese", "wasted_self_check", "Completion claim requires tool call"):
        hits = [c for c in cands.values() if pat in (c["text"] or "")]
        print(" pattern %r -> %d nodes" % (pat, len(hits)))
        for c in hits[:6]:
            vs = both(c["id"])
            print("   id=%d keeps=%s reasons=%s" % (c["id"], [v.get("keep") for v in vs],
                                                    [v.get("reason") for v in vs]))
    print("\n=== (iii) SYNTHETIC junk — 3 real cog_rum_ layouts printed first ===")
    rum = [c for c in cands.values() if (c["name"] or "").startswith("cog_rum_")][:3]
    for c in rum:
        print("  real id=%d text=%r" % (c["id"], c["text"][:260]))
    junk = [
        "During session 20260927 the agent edited retrieval.py. | The agent edited a file. | Remember that the agent edited retrieval.py.",
        "While fixing line 412 of athena_memory.py. | Line 412 was changed. | Recall that line 412 of athena_memory.py was edited.",
        "The agent discussed a variable named _sit_cache. | _sit_cache exists. | Know that _sit_cache is a variable.",
        "In any task. | Never use Chinese. | Never reply in Chinese under any circumstances.",
        "In any task. | Never use bash. | Never use the terminal tool.",
        "On every turn. | Always call a tool before answering. | Call at least one tool every turn.",
        "When the user said hello in session 20261001. | The user greeted the agent. | Remember the user once said hello.",
        "In session 20261001 the cron ran. | A cron ran that day. | Remember that a cron ran on 20261001.",
        "While editing init.py. | init.py has 1500 lines. | Know that init.py is long.",
        "In any coding task. | Always add a docstring to every function and a comment to every line. | Add ritual comments everywhere.",
    ]
    kk = 0
    for i, t in enumerate(junk):
        v = judge("glm-5.3-flash", t, {})   # one cheap judge; junk should be obvious
        kk += 1 if v.get("keep") else 0
        print("  junk#%d keep=%r form=%r reason=%r" % (i, v.get("keep"), v.get("form"), v.get("reason")))
    print("  synthetic KEEP rate = %d/10" % kk)

def phase_report(mod, mm):
    cands = {c["id"]: c for c in candidates(mod, mm)}
    c1 = load_cache("glm-5.3-flash"); c2 = load_cache("deepseek-v4.1-flash-1017")
    panel = {a: load_cache(a) for a in ("sonnet-5.5", "gpt-6.1-sol", "gemini-4-argon")}
    tg = json.loads((GE / "vet40_panel_targets.json").read_text()) if (GE / "vet40_panel_targets.json").exists() else {"disagree": [], "audit": []}
    audited = set(tg["disagree"]) | set(tg["audit"])
    import collections
    def panel_keep(nid):
        votes = [panel[a].get(str(nid), {}).get("keep") for a in panel]
        votes = [v for v in votes if v is not None]
        return sum(1 for v in votes if v), len(votes)
    overall = collections.Counter(); by_form = collections.Counter(); by_prefix = collections.Counter()
    by_origin = collections.Counter(); rows = []
    for nid, c in cands.items():
        k1 = c1.get(str(nid), {}).get("keep"); k2 = c2.get(str(nid), {}).get("keep")
        cheap = [k for k in (k1, k2) if k is not None]
        cheap_keep = (sum(1 for k in cheap if k) >= len(cheap) / 2) if cheap else None
        pk, pn = panel_keep(nid)
        if nid in audited and pn:
            final = pk > pn / 2
        else:
            final = bool(cheap_keep)
        tag = "KEEP" if final else "REJECT"
        overall[tag] += 1
        f = c1.get(str(nid), {}).get("form") or c2.get(str(nid), {}).get("form") or "?"
        by_form[(f, tag)] += 1
        pfx = ("cog_rum_" if (c["name"] or "").startswith("cog_rum_") else
               "cog_prov_" if (c["name"] or "").startswith("cog_prov_") else
               "cog_val_" if (c["name"] or "").startswith("cog_val_") else "(other)")
        by_prefix[(pfx, tag)] += 1
        for o in ("false_completion_claim", "compaction_loss", "wasted_self_check",
                  "Completion claim requires tool call"):
            if o in (c["text"] or ""): by_origin[(o, tag)] += 1
        rows.append({"id": nid, "name": c["name"], "text": c["text"], "status": c["status"],
                     "cheap": [k1, k2], "panel": [pk, pn], "final": tag,
                     "reasons": [c1.get(str(nid), {}).get("reason"), c2.get(str(nid), {}).get("reason")],
                     "audited": nid in audited})
    (GE / "vet40.json").write_text(json.dumps({"overall": dict(overall), "by_form": {str(k): v for k, v in by_form.items()},
        "by_prefix": {str(k): v for k, v in by_prefix.items()}, "by_origin": {str(k): v for k, v in by_origin.items()},
        "rows": rows}, indent=1))
    print("OVERALL:", dict(overall))
    print("by form:", {str(k): v for k, v in by_form.items()})
    print("by prefix:", {str(k): v for k, v in by_prefix.items()})
    print("by origin (text match):", {str(k): v for k, v in by_origin.items()})
    ag = [nid for nid in tg["audit"] if nid in cands]
    agree_n = sum(1 for nid in ag if (c1.get(str(nid), {}).get("keep") == panel_keep(nid)[0] > panel_keep(nid)[1] / 2))
    print("cheap-vs-panel agreement on the audited agree-set: %d/%d" % (agree_n, len(ag)))
    rnd = random.Random(40)
    def show(title, sel):
        print("\n=== %s (%d) ===" % (title, len(sel)))
        for r in sel:
            print(" id=%d %s status=%s cheap=%s panel=%s final=%s" % (r["id"], r["name"], r["status"], r["cheap"], r["panel"], r["final"]))
            print("   text: %r" % (r["text"][:400]))
            print("   reasons: %s" % r["reasons"])
    kp = [r for r in rows if r["final"] == "KEEP"]; rj = [r for r in rows if r["final"] == "REJECT"]
    show("30 random KEEPs", rnd.sample(kp, min(30, len(kp))))
    show("30 random REJECTs", rnd.sample(rj, min(30, len(rj))))
    disp = [r for r in rows if r["audited"] and r["cheap"][0] is not None and r["cheap"][1] is not None and r["cheap"][0] != r["cheap"][1]]
    disp.sort(key=lambda r: abs(r["panel"][0] - r["panel"][1] / 2) if r["panel"][1] else 9)
    show("20 lowest-confidence disputed", disp[:20])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("phase"); a = ap.parse_args()
    mod, mm = load_store()
    {"messages": phase_messages, "snapshot": phase_snapshot, "pilot": phase_pilot,
     "stage1": phase_stage1, "stage2": phase_stage2, "clusters": phase_clusters,
     "controls": phase_controls, "report": phase_report}[a.phase](mod, mm)
