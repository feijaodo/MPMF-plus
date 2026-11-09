"""R85 scorecard — TUNE/HOLDOUT accuracy of the LIVE provider on a store snapshot.

For each frozen question: prefetch the injected block (the real plugin), answer
with a cheap model, judge with 3 independent models, majority vote. Writes one
JSON line per question plus a summary, and keeps a history for drift alerts.

    python3 scorecard.py --split both --budget-usd 0.10
    python3 scorecard.py --limit 6 --split tune      # smoke
"""
import argparse
import os
from .. import paths as _paths
_BASE = os.environ.get('MPMF_EVAL_DIR') or str(_paths.home() / "eval")
import json
import os
import random
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = _BASE
QUESTIONS = os.environ.get("MPMF_QUESTIONS") or os.path.join(BASE, "questions.json")
LIVE = str(_paths.store_path())
SNAPSHOT = os.path.join(BASE, "snapshot.mpmf")
OUT = os.path.join(BASE, "scorecard.jsonl")
SUMMARY = os.path.join(BASE, "scorecard_summary.json")
HISTORY = os.path.join(BASE, "scorecard_history.jsonl")

ANSWER_MODEL = "deepseek/deepseek-v4.1-flash"
JUDGES = ["deepseek/deepseek-v4.1-flash", "google/gemini-2.5-flash", "openai/gpt-4.1-mini"]
PRICE = {  # $ per 1M tokens (in, out) — conservative
    "deepseek/deepseek-v4.1-flash": (0.10, 0.40),
    "google/gemini-2.5-flash": (0.35, 2.90),
    "openai/gpt-4.1-mini": (0.45, 1.80),
}
WORKERS = 4
_LOCK = threading.Lock()
COST = {"usd": 0.0, "calls": 0, "ptok": 0, "ctok": 0, "empty_retries": 0, "empty_final": 0}




def api_key():
    """Same loader as the provider (athena_memory._load_env_file):
    existing env wins, then $MPMF_ENV_FILE, then ~/.hermes/.env.
    Round 88b: the legacy $MPMF_HOME/.env does not exist; reading it made the nightly die."""
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    path = os.environ.get("MPMF_ENV_FILE") or str(_paths.Path.home() / ".hermes" / ".env")
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                key, _, val = line.partition("=")
                if key.strip() == "OPENROUTER_API_KEY":
                    v = val.strip().strip('"').strip("'")
                    if v:
                        return v
    except FileNotFoundError:
        pass
    raise RuntimeError("no OPENROUTER_API_KEY (env, $MPMF_ENV_FILE, ~/.hermes/.env)")


# r92: resolved on first use, not at import — `nightly --dry-run` (and the deploy gate)
# import this module on hosts with no key and must not die; a real run still fails loudly.
_KEY = None


def key():
    global _KEY
    if _KEY is None:
        _KEY = api_key()
    return _KEY


def llm(model, prompt, max_tokens=220, temperature=0.0):
    """Retry on EMPTY completions, not just on transport errors.

    A reasoning model can spend the whole max_tokens budget on hidden reasoning and
    return content="".  The old code accepted that as a final answer, which is what
    produced the 9 empty answers in the first baseline.  Now an empty completion is a
    retryable failure and each retry gets 2.5x the token budget.
    """
    mt = max_tokens
    for attempt in range(4):
        was_empty = False
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              headers={"Authorization": "Bearer " + key()},
                              json={"model": model, "temperature": temperature,
                                    "max_tokens": mt,
                                    "messages": [{"role": "user", "content": prompt}]},
                              timeout=90)
            j = r.json()
            if "choices" not in j:
                raise RuntimeError(str(j)[:200])
            u = j.get("usage", {})
            pt, ct = int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
            pin, pout = PRICE.get(model, (0.5, 2.0))
            with _LOCK:
                COST["calls"] += 1
                COST["ptok"] += pt
                COST["ctok"] += ct
                COST["usd"] += (pt * pin + ct * pout) / 1e6
            content = (j["choices"][0]["message"].get("content") or "").strip()
            if not content:
                was_empty = True
                raise RuntimeError("empty completion finish_reason=%s"
                                   % j["choices"][0].get("finish_reason"))
            return content
        except Exception:
            if was_empty:
                with _LOCK:
                    COST["empty_retries"] += 1
            if attempt == 3:
                if was_empty:
                    with _LOCK:
                        COST["empty_final"] += 1
                return ""
            mt = int(mt * 2.5)
            time.sleep(1 + attempt)


ANS_P = """You answer questions from a memory assistant's injected context.
Use ONLY the context. If the context does not contain the answer, reply exactly DECLINED.
Be specific: names, numbers, dates as they appear in the context. One or two sentences.

CONTEXT:
{ctx}

QUESTION: {q}

ANSWER:"""

JUDGE_P = """Judge a single answer produced by a memory assistant. You do not know which system produced it.

QUESTION: {q}
ANSWER KEY (ground truth): {key}
ANSWER UNDER TEST: {ans}

Verdicts:
- "declined": the answer says DECLINED or refuses to answer
- "correct": it conveys the key's substance (specifics may be paraphrased)
- "partly": related but misses or blurs a specific
- "wrong": conflicts with the key or answers something else

Reply with JSON only: {{"verdict": "correct|partly|wrong|declined", "why": "<12 words"}}"""


def judge_one(model, q, key, ans):
    raw = llm(model, JUDGE_P.format(q=q, key=key, ans=ans[:1200]), 200)
    v = "unjudged"
    for tok in ("correct", "partly", "wrong", "declined"):
        if '"%s"' % tok in raw:
            v = tok
            break
    return v


def split_questions(qs, seed=85):
    """Stratified by (set, src): TUNE 65 / HOLDOUT 64."""
    buckets = defaultdict(list)
    for x in qs:
        buckets[(x.get("set") or "-", x.get("src") or "-")].append(x)
    rng = random.Random(seed)
    tune, hold = [], []
    for k in sorted(buckets):
        b = sorted(buckets[k], key=lambda x: x["id"])
        rng.shuffle(b)
        n_hold = int(round(len(b) * 64 / 129))
        hold += b[:n_hold]
        tune += b[n_hold:]
    # fix the exact counts deterministically
    tune.sort(key=lambda x: x["id"])
    hold.sort(key=lambda x: x["id"])
    while len(tune) > 65:
        hold.append(tune.pop())
    while len(hold) > 64:
        tune.append(hold.pop())
    return sorted(tune, key=lambda x: x["id"]), sorted(hold, key=lambda x: x["id"])


def load_provider():
    from mpmf_plus import provider
    provider.configure(PATH=SNAPSHOT)
    if hasattr(provider, "_invalidate_open_cache"):
        provider._invalidate_open_cache()
    from .. import hermes_adapter
    prov = hermes_adapter.MPMFPlusProvider()
    try:
        prov.initialize("r85-scorecard")
    except Exception:
        pass
    return provider, prov


def build_parser():
    """The scorecard flag surface, shared with nightly --dry-run."""
    ap = argparse.ArgumentParser(prog="mpmf-eval scorecard")
    ap.add_argument("--split", default="both", choices=["tune", "holdout", "both"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget-usd", type=float, default=0.10)
    ap.add_argument("--out", default=OUT)
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)

    os.makedirs(BASE, exist_ok=True)
    if not os.path.exists(SNAPSHOT) or os.path.getmtime(SNAPSHOT) < os.path.getmtime(LIVE) - 60:
        shutil.copy2(LIVE, SNAPSHOT)
    qs = json.load(open(QUESTIONS))
    tune, hold = split_questions(qs)
    todo = []
    if a.split in ("tune", "both"):
        todo += [("tune", x) for x in tune]
    if a.split in ("holdout", "both"):
        todo += [("holdout", x) for x in hold]
    if a.limit:
        todo = todo[: a.limit]

    provider, prov = load_provider()
    rows = []

    def work(item):
        split, x = item
        if COST["usd"] > a.budget_usd:
            return None
        t0 = time.time()
        try:
            ctx = prov.prefetch(x["q"]) or ""
        except Exception as e:
            ctx = ""
            rows.append({"id": x["id"], "error": repr(e)})
        pre_ms = (time.time() - t0) * 1000
        ans = llm(ANSWER_MODEL, ANS_P.format(ctx=ctx[:6000], q=x["q"]))
        with ThreadPoolExecutor(3) as ex:
            verdicts = list(ex.map(lambda m: judge_one(m, x["q"], x["key"], ans), JUDGES))
        c = Counter(v for v in verdicts if v != "unjudged")
        verdict = c.most_common(1)[0][0] if c else "unjudged"
        if c and len(c) > 1 and c.most_common(2)[0][1] == c.most_common(2)[1][1]:
            verdict = "partly"
        return {"id": x["id"], "split": split, "set": x.get("set"), "src": x.get("src"),
                "q": x["q"], "key": x["key"], "answer": ans[:600], "verdict": verdict,
                "verdicts": verdicts, "prefetch_ms": round(pre_ms, 1),
                "block_chars": len(ctx), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    with ThreadPoolExecutor(WORKERS) as ex:
        for r in ex.map(work, todo):
            if r:
                rows.append(r)
                if len(rows) % 10 == 0:
                    print("  %d/%d  $%.3f" % (len(rows), len(todo), COST["usd"]), flush=True)

    rows.sort(key=lambda r: (r.get("split") or "", r.get("id", 0)))
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summ = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n": len(rows), "cost_usd": round(COST["usd"], 4), "calls": COST["calls"],
            "empty_retries": COST["empty_retries"], "empty_final": COST["empty_final"],
            "tokens": {"in": COST["ptok"], "out": COST["ctok"]}, "splits": {}}
    for sp in ("tune", "holdout"):
        rs = [r for r in rows if r.get("split") == sp]
        if not rs:
            continue
        c = Counter(r.get("verdict") for r in rs)
        pre = sorted(r["prefetch_ms"] for r in rs if "prefetch_ms" in r)
        summ["splits"][sp] = {
            "n": len(rs),
            "correct": c["correct"], "partly": c["partly"], "wrong": c["wrong"],
            "declined": c["declined"], "unjudged": c["unjudged"],
            "accuracy": round((c["correct"] + 0.5 * c["partly"]) / len(rs), 4),
            "prefetch_p90_ms": round(pre[int(0.9 * (len(pre) - 1))], 1) if pre else None,
        }
    json.dump(summ, open(SUMMARY, "w"), indent=1)
    with open(HISTORY, "a") as f:
        f.write(json.dumps({k: summ[k] for k in ("ts", "n", "cost_usd", "splits")}) + "\n")
    print(json.dumps(summ, indent=1))
    print("cost per question: $%.5f" % (COST["usd"] / max(1, len(rows))))


if __name__ == "__main__":
    main()
