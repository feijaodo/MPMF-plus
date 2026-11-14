"""g7 — lessons test on frozen incident tasks: OFF vs ON-live vs ON-matched.

Eval-only glue around behaviour.run_task (task_lessons arm) and incident.py.  Nothing here changes live
answers or any default.  Task files, lesson text, transcripts and session ids stay outside the repo.

  retrieve   for every task prompt, ask the package's live lesson path (provider.recall_cognition with the
             config's cognition_top_k / cognition_min_cos / cognition_rank, vet on, rendered the way the
             adapter renders it) against a COPY of the store, and write {task: {block, names, ...}}.
  run        one arm (off | live | matched) over the frozen tasks: worker threads, per-run wall-clock timeout,
             shared dollar ceiling (a run stopped by the ceiling is `not_run`, never pass/fail), resume only
             tasks that have no row yet.
  analyze    classify every row (pass / fail / no_final / error / timeout / not_run), Wilson CIs, hook-fired
             validity per ON arm, exact McNemar pairs.

    python -m mpmf_plus.eval.lessons_test retrieve --frozen F --md5 M --out R.json
    python -m mpmf_plus.eval.lessons_test run --arm live --frozen F --md5 M --retrieval R.json --out rows.jsonl
    python -m mpmf_plus.eval.lessons_test analyze --frozen F --rows off=a.jsonl live=b.jsonl matched=c.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from . import mcnemar as _mcnemar

MODEL = "deepseek/deepseek-v4.1-flash"
MAX_CALLS = 60
RUN_TIMEOUT_S = 900
ARMS = ("off", "live", "matched")


# --------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.959964):
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_pairs(base: Dict[str, bool], cur: Dict[str, bool]) -> Dict[str, Any]:
    """b = base fail and cur pass, c = reverse, over the tasks present in both dicts."""
    common = [t for t in base if t in cur]
    b = sum(1 for t in common if cur[t] and not base[t])
    c = sum(1 for t in common if base[t] and not cur[t])
    return {"n": len(common), "b": b, "c": c, "both_pass": sum(1 for t in common if base[t] and cur[t]),
            "both_fail": sum(1 for t in common if not base[t] and not cur[t]), "p": round(_mcnemar(b, c), 4)}


# --------------------------------------------------------------------------- frozen file
def load_frozen(path: str, md5: str) -> List[dict]:
    with open(path, "rb") as fh:
        got = hashlib.md5(fh.read()).hexdigest()
    if got != md5:
        raise SystemExit("REFUSE: frozen file md5 %s != expected %s" % (got, md5))
    with open(path) as fh:
        d = json.load(fh)
    return d["tasks"] if isinstance(d, dict) else d


# --------------------------------------------------------------------------- live retrieval
def retrieve_live(prompt: str, top_k: Optional[int] = None, max_chars: Optional[int] = None) -> Dict[str, Any]:
    """What the live path hands the model for `prompt`: provider.recall_cognition on the configured store,
    mode all, vet on, no count bumps, rendered through rendering.cognition_block (the adapter's own steps).
    Returns {block, names, scores, status, pool_size, n}.  block == '' when nothing is retrieved."""
    from .. import paths as _paths, provider as _p, rendering as _r
    cfg = _paths.load_config()
    k = int(top_k if top_k is not None else cfg.get("cognition_top_k", 3))
    mc = int(max_chars if max_chars is not None else cfg.get("cognition_max_chars", 700))
    qv = None
    try:
        qv = _p.embed_query(prompt)
    except Exception:
        qv = None
    kw = {"query_vec": qv} if qv is not None else {}
    meta: Any = _p.recall_cognition(prompt, top_k=k, mode="all", bump_counts=False, vet=True,
                                    log_inject=True, return_meta=True, **kw)
    if isinstance(meta, tuple):                      # package-side port returns (block, meta)
        meta = {"block": meta[0], "items": [], **(meta[1] or {})}
    raw = (meta or {}).get("block") or ""
    items = (meta or {}).get("items") or []
    return {"block": _r.cognition_block(raw, max_chars=mc) if raw else "",
            "names": [i.get("name") for i in items], "scores": [i.get("score") for i in items],
            "status": (meta or {}).get("cosine_status") or ((meta or {}).get("retrieval") or "?"),
            "pool_size": (meta or {}).get("pool_size"), "n": len(items),
            "retrieval": (meta or {}).get("retrieval"), "query_vec": qv is not None}


def own_names(node_name: str) -> List[str]:
    """The task's own lesson node plus the pool lesson it was merged into / promoted to (if any):
    seeing either one among the retrieved lessons counts as the task's own lesson being shown."""
    if not node_name:
        return []
    out = [node_name]
    try:
        from .. import provider as _p
        mm = _p._open().mem
        nid = mm.get_node_id_by_name(node_name)
        w = (mm.get_wiki(nid) or {}) if nid is not None else {}
        for k in ("_merged_into", "_promoted_to"):
            if w.get(k):
                out.append(w[k])
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- spend meter + guarded caller
class SharedMeter:
    """Dollar total shared by several processes (flock on a small JSON file)."""

    def __init__(self, path: str, ceiling: float):
        self.path, self.ceiling = path, float(ceiling)
        if not os.path.exists(path):
            with open(path, "a") as fh:
                fh.write("")

    def _rw(self, delta: float) -> float:
        import fcntl
        with open(self.path, "r+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            raw = fh.read().strip()
            tot = float(json.loads(raw)["usd"]) if raw else 0.0
            tot += delta
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps({"usd": round(tot, 6)}))
            fh.flush()
            fcntl.flock(fh, fcntl.LOCK_UN)
        return tot

    def add(self, usd: float) -> float:
        return self._rw(usd)

    def total(self) -> float:
        return self._rw(0.0)

    def over(self) -> bool:
        return self.total() >= self.ceiling


_TLS = threading.local()


def guarded_caller(inner: Callable, meter: SharedMeter, price=(0.10, 0.40), timeout_s: float = RUN_TIMEOUT_S):
    """Wrap a model caller: refuse to start a call past the dollar ceiling (error text starts with CEILING) or
    past this run's wall-clock deadline (TimeoutError); add each call's list-price cost to the shared meter."""
    pin, pout = price

    def call(msgs, tools):
        if meter.over():
            raise RuntimeError("CEILING: shared spend ceiling $%.2f reached" % meter.ceiling)
        t0 = getattr(_TLS, "t0", None)
        if t0 is not None and time.time() - t0 > timeout_s:
            raise TimeoutError("run timeout after %ds" % timeout_s)
        m, u = inner(msgs, tools)
        u = u or {}
        meter.add((int(u.get("prompt_tokens") or 0) * pin + int(u.get("completion_tokens") or 0) * pout) / 1e6)
        return m, u
    return call


# --------------------------------------------------------------------------- one arm
def lesson_for(arm: str, task: dict, retrieval: Optional[dict]) -> str:
    if arm == "off":
        return ""
    if arm == "matched":
        return task.get("lesson") or ""
    return ((retrieval or {}).get(task["id"]) or {}).get("block") or ""


def make_row(arm: str, tr: dict, sc: dict, flag: bool, lesson_src: str, live: Optional[dict]) -> dict:
    err = tr.get("error")
    return {"arm": arm, "flag_seen": flag, "task": tr["task"], "pass": bool(sc["pass"]), "error": err,
            "timeout": bool(err and ("timeout" in err.lower() or "timed out" in err.lower())),
            "not_run": bool(err and err.startswith("RuntimeError('CEILING")),
            "score": {k: sc[k] for k in ("items", "items_met", "items_total", "tool_calls", "repeats", "flags")},
            "usage": tr.get("usage"), "events": tr["events"], "final": tr.get("final"), "hook": tr.get("hook"),
            "lesson_source": lesson_src, "live": live}


def run_arm(arm: str, tasks: List[dict], caller: Callable, out: str, workers: int = 5,
            retrieval: Optional[dict] = None, max_steps: int = MAX_CALLS, flag: Optional[bool] = None) -> Dict[str, Any]:
    from . import behaviour as B
    B.MAX_STEPS = max_steps
    on = arm != "off"
    done = set()
    if os.path.exists(out):                          # resume: only tasks without a row
        done = {json.loads(l)["task"] for l in open(out) if l.strip()}
    todo = [t for t in tasks if t["id"] not in done]
    lk = threading.Lock()
    fh = open(out, "a")

    def one(t):
        _TLS.t0 = time.time()
        tt = dict(t, lesson=lesson_for(arm, t, retrieval))
        live = (retrieval or {}).get(t["id"]) if arm == "live" else None
        try:
            tr = B.run_task(dict(B.TASK_BY_ID[t["id"]], lesson=tt["lesson"]), caller, task_lessons=on)
            sc = B.score(tr)
            row = make_row(arm, tr, sc, bool(on if flag is None else flag), arm, live and {k: v for k, v in live.items() if k != "block"})
        except Exception as e:                       # harness failure: an error row, never hidden
            row = {"arm": arm, "flag_seen": flag, "task": t["id"], "pass": False, "error": "harness: " + repr(e)[:200],
                   "timeout": False, "not_run": False,
                   "score": {"items": {}, "items_met": 0, "items_total": 0, "tool_calls": 0, "repeats": 0, "flags": {"no_final": True}},
                   "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "events": [], "final": None, "hook": None,
                   "lesson_source": arm, "live": None}
        if row["not_run"]:
            return row                               # not written: a ceiling stop leaves no row, so a resume runs it
        with lk:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
        return row
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, todo))
    fh.close()
    return {"arm": arm, "ran": sum(1 for r in rows if not r["not_run"]), "not_run": sum(1 for r in rows if r["not_run"]),
            "resumed_existing": len(done), "secs": round(time.time() - t0)}


# --------------------------------------------------------------------------- analysis
def classify(row: Optional[dict]) -> str:
    """not_run | error | timeout | no_final | pass | fail   (errors are never counted as no_final)."""
    if row is None or row.get("not_run"):
        return "not_run"
    if row.get("timeout"):
        return "timeout"
    if row.get("error"):
        return "error"
    if row.get("final") is None:
        return "no_final"
    return "pass" if row.get("pass") else "fail"


def hook_check(arm: str, row: dict) -> str:
    """OFF: none may show. ON: every claim shows the lesson text in the transcript.
    Returns ok | no_claim | no_lesson_retrieved | invalid_not_fired | invalid_off_has_lesson."""
    lesson_evs = [e for e in row.get("events", []) if e.get("t") == "lesson" and e.get("text")]
    hk = row.get("hook") or {}
    if arm == "off":
        return "invalid_off_has_lesson" if (lesson_evs or hk.get("fired")) else "ok"
    reason = hk.get("reason")
    if reason == "task_has_no_lesson":
        return "no_lesson_retrieved"
    if hk.get("fired") and lesson_evs and hk.get("text") == lesson_evs[0]["text"]:
        return "ok"
    if reason == "no_completion_claim" and row.get("final") is None:
        return "no_claim"
    return "invalid_not_fired"


def arm_summary(arm: str, rows: Dict[str, dict], ids: List[str]):
    cls = {t: classify(rows.get(t)) for t in ids}
    cnt = {k: sum(1 for v in cls.values() if v == k) for k in ("pass", "fail", "no_final", "error", "timeout", "not_run")}
    ran = len(ids) - cnt["not_run"]
    lo, hi = wilson(cnt["pass"], ran)
    fin = ran - cnt["no_final"]
    flo, fhi = wilson(cnt["pass"], fin)
    hk: Dict[str, int] = {}
    for t in ids:
        if rows.get(t) and cls[t] != "not_run":
            h = hook_check(arm, rows[t])
            hk[h] = hk.get(h, 0) + 1
    calls = [rows[t]["score"]["tool_calls"] for t in ids if rows.get(t) and cls[t] != "not_run"]
    return {"n_tasks": len(ids), "ran": ran, **cnt, "pass_rate": round(cnt["pass"] / ran, 3) if ran else None,
            "wilson95": [round(lo, 3), round(hi, 3)], "n_with_final": fin,
            "pass_rate_excl_no_final": round(cnt["pass"] / fin, 3) if fin else None,
            "wilson95_excl_no_final": [round(flo, 3), round(fhi, 3)], "hook": hk,
            "mean_tool_calls": round(sum(calls) / len(calls), 2) if calls else None}, cls


def analyze(frozen_ids: List[str], rows_by_arm: Dict[str, Dict[str, dict]], p11b_ids=(), live_meta=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_tasks": len(frozen_ids), "arms": {}}
    classes: Dict[str, Dict[str, str]] = {}
    for arm, rows in rows_by_arm.items():
        summ, classes[arm] = arm_summary(arm, rows, frozen_ids)
        out["arms"][arm] = summ

    def strict(arm):                                 # pass-or-not per task, tasks that ran only
        return {t: classes[arm][t] == "pass" for t in frozen_ids if classes[arm][t] != "not_run"}

    def pairs(a, b, ids=None, drop=()):
        sa, sb = strict(a), strict(b)
        keep = set(ids if ids is not None else frozen_ids)
        bad = {t for t in frozen_ids if any(classes[x][t] in drop for x in (a, b))}
        return mcnemar_pairs({t: v for t, v in sa.items() if t in keep and t not in bad},
                             {t: v for t, v in sb.items() if t in keep and t not in bad})
    comps = [("off", "live"), ("off", "matched"), ("live", "matched")]
    out["mcnemar"] = {}
    for a, b in comps:
        if a in rows_by_arm and b in rows_by_arm:
            k = "%s_vs_%s" % (b, a)                  # arm b relative to base a
            out["mcnemar"][k] = {"all": pairs(a, b),
                                 "excl_no_final_either_arm": pairs(a, b, drop=("no_final",)),
                                 "excl_error_timeout_either_arm": pairs(a, b, drop=("error", "timeout")),
                                 "p11b_subset": pairs(a, b, ids=set(p11b_ids)) if p11b_ids else None,
                                 "new_subset": pairs(a, b, ids=set(frozen_ids) - set(p11b_ids)) if p11b_ids else None}
    if "live" in rows_by_arm and "off" in rows_by_arm:
        m = out["mcnemar"]["live_vs_off"]["all"]
        out["effect"] = "helped" if (m["p"] < 0.05 and m["b"] > m["c"]) else "no_effect"
    if live_meta:
        names = [live_meta[t]["names"] for t in frozen_ids if t in live_meta]
        out["live_retrieval"] = {
            "tasks": len(names), "no_lesson_retrieved": sum(1 for n in names if not n),
            "mean_lessons_shown": round(sum(len(n) for n in names) / max(1, len(names)), 2),
            "pool_size": sorted({live_meta[t].get("pool_size") for t in frozen_ids if t in live_meta} - {None}),
            "distinct_lessons_shown": len({x for n in names for x in n}),
            "own_lesson_shown": sum(1 for t in frozen_ids if t in live_meta and live_meta[t].get("own_shown")),
        }
    return out


# --------------------------------------------------------------------------- CLI
def _read_rows(path: str) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    if os.path.exists(path):
        for l in open(path):
            if l.strip():
                r = json.loads(l)
                rows[r["task"]] = r
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mpmf-eval lessons_test")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("retrieve", "run", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--frozen", required=True)
        p.add_argument("--md5", required=True)
    sub.choices["retrieve"].add_argument("--out", required=True)
    r = sub.choices["run"]
    r.add_argument("--arm", required=True, choices=ARMS)
    r.add_argument("--out", required=True)
    r.add_argument("--retrieval", default="")
    r.add_argument("--workers", type=int, default=5)
    r.add_argument("--meter", required=True)
    r.add_argument("--ceiling-usd", type=float, required=True)
    a_ = sub.choices["analyze"]
    a_.add_argument("--rows", nargs="+", required=True, help="arm=path ...")
    a_.add_argument("--retrieval", default="")
    a_.add_argument("--out", default="")
    a = ap.parse_args(argv)
    specs = load_frozen(a.frozen, a.md5)
    if a.cmd == "retrieve":
        res = {}
        for s in specs:
            r_ = retrieve_live(s["prompt"])
            own = own_names((s.get("provenance") or {}).get("lesson_node", ""))
            r_["own_shown"] = bool(set(own) & set(r_["names"]))
            res[s["id"]] = r_
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1, sort_keys=True)
        print(json.dumps({"tasks": len(res), "empty": sum(1 for v in res.values() if not v["names"])}))
        return 0
    if a.cmd == "analyze":
        rows = {kv.split("=", 1)[0]: _read_rows(kv.split("=", 1)[1]) for kv in a.rows}
        live = json.load(open(a.retrieval)) if a.retrieval else None
        res = analyze([s["id"] for s in specs], rows, [s["id"] for s in specs if s.get("p11b_frozen")], live)
        txt = json.dumps(res, indent=1, sort_keys=True)
        if a.out:
            open(a.out, "w").write(txt)
        print(txt)
        return 0
    # run
    from . import behaviour as B, incident as I
    arm = a.arm
    os.environ["MPMF_LESSONS_AT_ACTION"] = "true" if arm != "off" else "false"
    from .. import paths as _paths
    seen = _paths.load_config().get("lessons_at_action")
    if seen is not (arm != "off"):
        raise SystemExit("REFUSE: arm=%s but the harness sees lessons_at_action=%r" % (arm, seen))
    retrieval = json.load(open(a.retrieval)) if a.retrieval else None
    if arm == "live" and retrieval is None:
        raise SystemExit("REFUSE: arm live needs --retrieval")
    # sandbox hardening (as p11b): an invalid regex given to the sandbox grep answers like grep, not a crash
    def _safe(fn):
        def g(w, m):
            try:
                return fn(w, m)
            except re.error as e:
                return "grep: %s" % e, False
        return g
    I._DEFAULTS = [(rx, _safe(fn) if fn is I._grep else fn) for rx, fn in I._DEFAULTS]
    I.DEFAULT_TABLE = [(rx, I._rec(fn)) for rx, fn in I._DEFAULTS] + [(r".*", I._catch_all)]
    tasks = I.register(specs)
    B.PRICE.update(I.PRICES)
    meter = SharedMeter(a.meter, a.ceiling_usd)
    caller = guarded_caller(B.make_model_caller(MODEL, 1e9), meter, I.PRICES[MODEL])
    res = run_arm(arm, tasks, caller, a.out, a.workers, retrieval, MAX_CALLS, bool(seen))
    res["meter_usd"] = round(meter.total(), 4)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
