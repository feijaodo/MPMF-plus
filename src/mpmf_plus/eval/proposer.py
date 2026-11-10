"""R85 proposer — propose-only. ONE registry change, tested on TUNE, confirmed on HOLDOUT.

    python3 proposer.py            # reads scorecard.jsonl, writes proposals.jsonl
    python3 proposer.py --dry-run  # pick the setting, print the plan, no LLM spend

Never writes code. A proposal is a JSON line with the paired evidence; a
winning-both proposal also prints one line for the user. p7: if auto_apply is on in
recall.config.json AND the setting is a recall_config registry key, the winner is applied
by eval/autoapply.py (PAUSED.md row, message for the user, nightly auto-revert). Default off.
"""
import argparse
import json
import os
from .. import paths as _paths
_BASE = os.environ.get('MPMF_EVAL_DIR') or str(_paths.home() / "eval")
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

BASE = _BASE
REGISTRY = os.path.join(BASE, "registry.json")
SCORECARD = os.path.join(BASE, "scorecard.jsonl")
PROPOSALS = os.path.join(BASE, "proposals.jsonl")
STATE = os.path.join(BASE, "proposer_state.json")

sys.path.insert(0, BASE)
from . import scorecard as sc  # noqa: E402

# failure mode -> the setting most likely to move it, in priority order
# top_n is deliberately absent: the plugin passes top_k=8 explicitly, so
# provider.DEFAULT_K cannot move the injected block (a no-op change would burn
# a full TUNE run and always lose). It stays in the registry as documentation.
HEURISTIC = {
    "declined": ["render_cap_memory", "d20_k", "rrf_k"],
    "wrong": ["d20_k", "rrf_k", "rerank_budget_ms"],
    "partly": ["rerank_budget_ms", "rrf_k", "render_cap_memory"],
}


def mcnemar(b, c):
    """Exact two-sided binomial p for discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def run_items(prov, items, budget_usd):
    rows = []

    def work(x):
        if sc.COST["usd"] > budget_usd:
            return None
        t0 = time.time()
        try:
            ctx = prov.prefetch(x["q"]) or ""
        except Exception:
            ctx = ""
        pre = (time.time() - t0) * 1000
        ans = sc.llm(sc.ANSWER_MODEL, sc.ANS_P.format(ctx=ctx[:6000], q=x["q"]))
        with ThreadPoolExecutor(3) as ex:
            vs = list(ex.map(lambda m: sc.judge_one(m, x["q"], x["key"], ans), sc.JUDGES))
        c = Counter(v for v in vs if v != "unjudged")
        verdict = c.most_common(1)[0][0] if c else "unjudged"
        if c and len(c) > 1 and c.most_common(2)[0][1] == c.most_common(2)[1][1]:
            verdict = "partly"
        return {"id": x["id"], "verdict": verdict, "prefetch_ms": round(pre, 1)}

    with ThreadPoolExecutor(sc.WORKERS) as ex:
        for r in ex.map(work, items):
            if r:
                rows.append(r)
    return rows


def score(rows):
    return sum(1 for r in rows if r["verdict"] == "correct") + 0.5 * sum(
        1 for r in rows if r["verdict"] == "partly")


def pick_setting(failures, registry, avoid):
    modes = Counter(r["verdict"] for r in failures)
    for mode in [m for m, _ in modes.most_common()]:
        for sid in HEURISTIC.get(mode, []):
            if sid in avoid:
                continue
            for s in registry["settings"]:
                if s["id"] == sid:
                    return s, mode
    return None, None


def next_value(s):
    lo, hi = s["allowed"]
    cur = s["current"]
    if isinstance(cur, bool):
        return not cur
    step = s["step"] or 1
    v = cur + step
    return lo if v > hi else round(v, 4)


def apply_setting(provider, setting, value, plugin_cfg):
    if setting["apply"] == "configure":
        attr = setting["target"].split(".")[-1]
        old = getattr(provider, attr, None)
        setattr(provider, attr, value)
        return old
    plugin_cfg[setting["target"].split(".")[-1]] = value
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mpmf-eval proposer")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--budget-usd", type=float, default=0.10)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(SCORECARD)]
    tune_base = [r for r in rows if r["split"] == "tune"]
    hold_base = [r for r in rows if r["split"] == "holdout"]
    if not tune_base:
        print("no TUNE rows in scorecard — run scorecard.py first")
        return
    failures = [r for r in tune_base if r["verdict"] != "correct"]
    registry = json.load(open(REGISTRY))
    state = json.load(open(STATE)) if os.path.exists(STATE) else {"tried": []}

    setting, mode = pick_setting(failures, registry, set(state["tried"][-3:]))
    if not setting:
        print("no untried setting for the current failure modes")
        return
    value = next_value(setting)
    plan = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "setting": setting["id"], "target": setting["target"],
            "from": setting["current"], "to": value,
            "dominant_failure": mode,
            "tune_failures": len(failures), "tune_n": len(tune_base)}
    print("PLAN", json.dumps(plan))
    if a.dry_run:
        return

    provider, prov = sc.load_provider()
    plugin_cfg = {}
    apply_setting(provider, setting, value, plugin_cfg)
    if setting["apply"] == "plugin_config":
        # plugin render caps are read at construction: rebuild the provider
        plugin = sys.modules["mpmf_plus_user"]
        cfg = dict(plugin._load_plugin_config() or {})
        cfg[setting["target"].split(".")[-1]] = value
        pass
        prov = plugin.MPMFPlusProvider(cfg)
        try:
            prov.initialize("r85-proposer")
        except Exception:
            pass

    qs = {q["id"]: q for q in json.load(open(sc.QUESTIONS))}
    tune_items = [qs[r["id"]] for r in tune_base]
    cand = run_items(prov, tune_items, a.budget_usd)
    base_by_id = {r["id"]: r["verdict"] for r in tune_base}
    ok = lambda v: v == "correct"
    b = sum(1 for r in cand if ok(r["verdict"]) and not ok(base_by_id[r["id"]]))
    c = sum(1 for r in cand if not ok(r["verdict"]) and ok(base_by_id[r["id"]]))
    p_tune = mcnemar(b, c)
    plan.update({"tune_candidate_acc": round(score(cand) / max(1, len(cand)), 4),
                 "tune_baseline_acc": round(score(tune_base) / max(1, len(tune_base)), 4),
                 "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": round(p_tune, 4),
                 "cost_tune_usd": round(sc.COST["usd"], 4)})

    if not (b > c and p_tune < 0.05):
        plan.update({"outcome": "rejected_on_tune", "confirmed_on_holdout": False})
        open(PROPOSALS, "a").write(json.dumps(plan) + "\n")
        state["tried"].append(setting["id"])
        json.dump(state, open(STATE, "w"))
        print("REJECTED on TUNE:", json.dumps(plan))
        return

    hold_items = [qs[r["id"]] for r in hold_base]
    cand_h = run_items(prov, hold_items, a.budget_usd)
    base_h = {r["id"]: r["verdict"] for r in hold_base}
    bh = sum(1 for r in cand_h if ok(r["verdict"]) and not ok(base_h[r["id"]]))
    ch = sum(1 for r in cand_h if not ok(r["verdict"]) and ok(base_h[r["id"]]))
    p_h = mcnemar(bh, ch)
    won = bh >= ch and p_h < 0.10
    plan.update({"holdout_candidate_acc": round(score(cand_h) / max(1, len(cand_h)), 4),
                 "holdout_baseline_acc": round(score(hold_base) / max(1, len(hold_base)), 4),
                 "mcnemar_b_holdout": bh, "mcnemar_c_holdout": ch,
                 "mcnemar_p_holdout": round(p_h, 4),
                 "cost_total_usd": round(sc.COST["usd"], 4),
                 "outcome": "won_both" if won else "won_tune_failed_holdout",
                 "confirmed_on_holdout": won})
    open(PROPOSALS, "a").write(json.dumps(plan) + "\n")
    state["tried"].append(setting["id"])
    json.dump(state, open(STATE, "w"))
    print("PROPOSAL", json.dumps(plan, indent=1))
    applied = False
    if won:
        from . import autoapply
        applied, msg = autoapply.maybe_apply(
            BASE, plan, {r["id"]: r["verdict"] for r in cand_h}, base_h)
        print(("PROPOSAL: " if applied else "AUTO-APPLY: ") + msg)
    if won and not applied:
        print("PROPOSAL: %s %s -> %s lifts TUNE %+.0f%% and HOLDOUT %+.0f%% (McNemar p=%.3f/%.3f) — say the word and I'll apply it."
              % (setting["id"], setting["current"], value,
                 100 * (plan["tune_candidate_acc"] - plan["tune_baseline_acc"]),
                 100 * (plan["holdout_candidate_acc"] - plan["holdout_baseline_acc"]),
                 p_tune, p_h))


if __name__ == "__main__":
    main()
