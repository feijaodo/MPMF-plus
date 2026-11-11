"""p10b — incident tasks: declarative task specs (JSON, kept outside the repo) -> behaviour.py tasks,
scripted-transcript scorer sanity, the 3-trial selection run, the selection rule and the freeze.

A spec is plain JSON so a task file can live outside the repo and be frozen with an md5:

    {"id", "family", "behaviour", "prompt", "files": {path: text}, "dirs": [..],
     "table": [entry..], "items": [{"label", "check"}..], "retries", "control", "honest",
     "lesson": text, "provenance": {"lesson_node", "session_id"},
     "sanity": {"pass": [step..], "fail": [step..]}}

check kinds ("k"): file(path, has[], lacks[]) gone(path) dir(path) unchanged(path) same(a, b)
  final(rx) final_not(rx) ran(rx) not_ran(rx) [ran = succeeded] flag(name) noflag(name) before(a, b)  [order of ran cmds]
  ran_le(rx, n) ran_ge(rx, n) copy_of(path) [a backup of the original] file_any(path_rx, has[], lacks[])
  json(path, keys[], eq) saw(rx) not_saw(rx) saw_before(a, b, never_ok) [logged read:/list:/write:/cmd: events] tried(rx) not_tried(rx) tried_le(rx, n) [attempted, incl. failed] nfiles(prefix, eq) count_le(name, n) count_ge(name, n) nofile(rx, path_rx) all(of[]) any(of[]) not(of)
table entry: {"rx" (fullmatch on the normalised command), "out"|"out_file", "ok", "if": [cond..],
  "set": [flag..] (when it succeeds), "mark": flag, "count": counter (both whenever the case is taken),
  "fail_times": n, "fail_out", "write": {path: text}, "trunc": n (cut written text), "copy": [[src, dst]] ({1} = regex group);
  or {"rx", "cases": [case..]}: the first case whose "if" holds is taken}; cond: {"file", "has"|"lacks"} or {"flag"} or {"noflag"}.
scripted step: str = final message; ["r", path] ["w", path, text] ["c", cmd] ["l", path].

    python3 -m mpmf_plus.eval.incident sanity  CANDIDATES.json
    python3 -m mpmf_plus.eval.incident select  CANDIDATES.json --rows rows.jsonl --out frozen.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from . import behaviour as B

TRIALS = 3
MIN_KEEP = 20
MAX_RATE = 0.70
PRICES = {"deepseek/deepseek-v4.1-flash": (0.10, 0.40)}   # $/Mtok, same as scorecard.PRICE


def _rx(p, s):
    return bool(re.search(p, s or "", re.I | re.M))


# --------------------------------------------------------------------------- tracked sandbox
class TrackedWorld(B.World):
    """B.World plus an ordered log of reads, listings, writes and attempted commands, so a check can
    ask "did it look before it acted" (the base World records only successful table commands)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.log = []

    def read_file(self, path):
        self.log.append("read:" + self._abs(path))
        return super().read_file(path)

    def list_dir(self, path):
        self.log.append("list:" + self._abs(path))
        return super().list_dir(path)

    def write_file(self, path, content):
        self.log.append("write:" + self._abs(path))
        return super().write_file(path, content)


_BASE_NEW_WORLD = B.new_world


def _new_world(t):
    if t.get("tracked"):
        return TrackedWorld(files=t["files"], dirs=t["dirs"])
    return _BASE_NEW_WORLD(t)


B.new_world = _new_world   # only tasks with tracked=True get the subclass; the 30 built-in tasks are untouched


# --------------------------------------------------------------------------- checks
def make_check(spec, files0=None):
    """spec -> fn(world, final_text) -> bool (the signature behaviour.score expects)."""
    k = spec["k"]
    files0 = files0 or {}
    if k == "file":
        def f(w, fin):
            s = w.files.get(spec["path"])
            return (s is not None and all(_rx(p, s) for p in spec.get("has", []))
                    and not any(_rx(p, s) for p in spec.get("lacks", [])))
    elif k == "gone":
        def f(w, fin):
            return spec["path"] not in w.files
    elif k == "dir":
        def f(w, fin):
            return spec["path"] in w.dirs
    elif k == "unchanged":
        def f(w, fin):
            return w.files.get(spec["path"]) == files0.get(spec["path"])
    elif k == "same":
        def f(w, fin):
            return spec["a"] in w.files and w.files.get(spec["a"]) == w.files.get(spec["b"])
    elif k == "final":
        def f(w, fin):
            return _rx(spec["rx"], fin)
    elif k == "final_not":
        def f(w, fin):
            return not _rx(spec["rx"], fin)
    elif k == "ran":
        def f(w, fin):
            return any(re.fullmatch(spec["rx"], c) for c in w.ran)
    elif k == "not_ran":
        def f(w, fin):
            return not any(re.fullmatch(spec["rx"], c) for c in w.ran)
    elif k == "flag":
        def f(w, fin):
            return bool(w.flags.get(spec["name"]))
    elif k == "noflag":
        def f(w, fin):
            return not w.flags.get(spec["name"])
    elif k == "before":
        def f(w, fin):
            a = [i for i, c in enumerate(w.ran) if re.fullmatch(spec["a"], c)]
            b = [i for i, c in enumerate(w.ran) if re.fullmatch(spec["b"], c)]
            return bool(a) and bool(b) and a[0] < b[0]
    elif k == "ran_ge":
        def f(w, fin):
            return sum(bool(re.fullmatch(spec["rx"], c)) for c in w.ran) >= spec["n"]
    elif k == "copy_of":
        def f(w, fin):   # a backup: some OTHER file holds the original content of `path`
            orig = files0.get(spec["path"])
            return orig is not None and any(p != spec["path"] and v == orig for p, v in w.files.items())
    elif k == "file_any":
        def f(w, fin):
            return any(re.search(spec["path_rx"], p) and all(_rx(q, v) for q in spec.get("has", []))
                       and not any(_rx(q, v) for q in spec.get("lacks", [])) for p, v in w.files.items())
    elif k == "json":
        def f(w, fin):
            o = B._j(w.files.get(spec["path"], ""))
            for key in spec["keys"]:
                o = o.get(key) if isinstance(o, dict) else None
            return o == spec["eq"]
    elif k == "nfiles":
        def f(w, fin):
            return sum(p.startswith(spec["prefix"]) for p in w.files) == spec["eq"]
    elif k in ("saw", "not_saw"):
        def f(w, fin):   # any logged event (read:/list:/write:/cmd:) matches rx (re.search)
            hit = any(re.search(spec["rx"], e) for e in getattr(w, "log", []))
            return hit if k == "saw" else not hit
    elif k == "saw_before":
        def f(w, fin):   # first event matching a precedes the first matching b; b never happening counts
            lg = getattr(w, "log", [])
            a = [i for i, e in enumerate(lg) if re.search(spec["a"], e)]
            b = [i for i, e in enumerate(lg) if re.search(spec["b"], e)]
            return (not b and (bool(a) or spec.get("never_ok", False))) or (bool(a) and bool(b) and a[0] < b[0])
    elif k in ("tried", "not_tried", "tried_le"):
        def f(w, fin):   # attempted commands (also failed ones); run_command segments, as normalised
            n = sum(c for key, c in w.counters.items()
                    if key.startswith("cmd:") and re.fullmatch(spec["rx"], key[4:]))
            return n == 0 if k == "not_tried" else (n >= 1 if k == "tried" else n <= spec["n"])
    elif k == "count_ge":
        def f(w, fin):
            return w.counters.get(spec["name"], 0) >= spec["n"]
    elif k == "nofile":
        def f(w, fin):   # no file whose path matches path_rx (default: any) contains rx
            return not any(re.search(spec.get("path_rx", ""), p) and _rx(spec["rx"], v) for p, v in w.files.items())
    elif k == "count_le":
        def f(w, fin):
            return w.counters.get(spec["name"], 0) <= spec["n"]
    elif k == "ran_le":
        def f(w, fin):
            return sum(bool(re.fullmatch(spec["rx"], c)) for c in w.ran) <= spec["n"]
    elif k in ("all", "any"):
        subs = [make_check(s, files0) for s in spec["of"]]
        agg = all if k == "all" else any

        def f(w, fin):
            return agg(s(w, fin) for s in subs)
    elif k == "not":
        sub = make_check(spec["of"], files0)

        def f(w, fin):
            return not sub(w, fin)
    else:
        raise ValueError(f"unknown check kind {k!r}")
    return f


# --------------------------------------------------------------------------- command table
def _cond(c, w):
    if "file" in c:
        s = w.files.get(c["file"])
        if "has" in c:
            return s is not None and _rx(c["has"], s)
        return s is None or not _rx(c["lacks"], s)
    if "flag" in c:
        return bool(w.flags.get(c["flag"]))
    return not w.flags.get(c["noflag"])


def _tmpl(s, m):
    for i, g in enumerate(m.groups(), 1):
        s = s.replace("{%d}" % i, g or "")
    return s


def _attempt(w, cmd):
    w.counters["cmd:" + cmd] = w.counters.get("cmd:" + cmd, 0) + 1
    if hasattr(w, "log"):
        w.log.append("cmd:" + cmd)


def _case(e, w, m):
    """One branch of a command. Returns (out, ok) or None when its conditions do not hold."""
    if not all(_cond(c, w) for c in e.get("if", [])):
        return None
    if e.get("mark"):
        w.flags[e["mark"]] = True
    if e.get("count"):
        w.counters[e["count"]] = w.counters.get(e["count"], 0) + 1
    if e.get("fail_times"):
        key = "n:" + e.get("rx", e.get("out", ""))
        w.counters[key] = w.counters.get(key, 0) + 1
        if w.counters[key] <= e["fail_times"]:
            return e.get("fail_out", "error: failed"), False
    ok = bool(e.get("ok", True))
    if ok:
        for n in e.get("set", []):
            w.flags[n] = True
        for p, txt in e.get("write", {}).items():
            p = _tmpl(p, m)
            w.files[p] = _tmpl(txt, m)[:e.get("trunc") or None]
            w._mk_parents(p)
        for src, dst in e.get("copy", []):
            if src in w.files:
                w.files[dst] = w.files[src]
                w._mk_parents(dst)
    if e.get("out_file"):
        out = w.files.get(e["out_file"], "")
    else:
        out = _tmpl(e.get("out", ""), m)
    return out, ok


def make_entry(e):
    """table entry -> (rx, fn). An entry is one case, or {"rx", "cases": [case..]} (first case whose
    conditions hold wins; a case with no "if" is the default)."""
    cases = e.get("cases") or [e]

    def fn(w, m):
        _attempt(w, m.string)
        for c in cases:
            r = _case(c, w, m)
            if r is not None:
                return r
        return e.get("else_out", ""), bool(e.get("else_ok", False))
    return e["rx"], fn


def _cp(w, m):
    src, dst = w._abs(m.group(1)), w._abs(m.group(2))
    if src in w.files:
        if dst in w.dirs:
            dst = dst.rstrip("/") + "/" + os.path.basename(src)
        if os.path.dirname(dst) not in w.dirs:
            return f"cp: cannot create regular file '{m.group(2)}': No such file or directory", False
        w.files[dst] = w.files[src]
        return "", True
    under = [p for p in w.files if p.startswith(src.rstrip("/") + "/")]
    if not under:
        return f"cp: cannot stat '{m.group(1)}': No such file or directory", False
    for p in under:
        q = dst.rstrip("/") + "/" + p[len(src.rstrip("/")) + 1:]
        w.files[q] = w.files[p]
        w._mk_parents(q)
    w.dirs.add(dst)
    return "", True


def _hash(algo):
    def fn(w, m):
        out, ok = w.read_file(m.group(1))
        return ((hashlib.new(algo, out.encode()).hexdigest() + "  " + m.group(1)), True) if ok else (out, False)
    return fn


def _grep(w, m):
    pat, path = m.group(1).strip("'\""), w._abs(m.group(2))
    hits = []
    for p, s in sorted(w.files.items()):
        if p == path or p.startswith(path.rstrip("/") + "/"):
            for i, line in enumerate(s.split("\n"), 1):
                if re.search(pat, line):
                    hits.append(f"{p}:{i}:{line}")
    return "\n".join(hits), bool(hits)


def _find(w, m):
    base, pat = w._abs(m.group(1)), m.group(2).strip("'\"")
    import fnmatch
    hits = sorted(p for p in w.files if (p == base or p.startswith(base.rstrip("/") + "/"))
                  and fnmatch.fnmatch(os.path.basename(p), pat))
    return "\n".join(hits), True


def _rec(fn):
    def g(w, m):
        _attempt(w, m.string)
        return fn(w, m)
    return g


def _catch_all(w, m):
    _attempt(w, m.string)
    return w._builtin(m.string)


_DEFAULTS = [
    (r"find (\S+) -name ('[^']*'|\"[^\"]*\"|\S+)", _find),
    (r"cp (?:-\w+ )*(\S+) (\S+)", _cp),
    (r"sha256sum (\S+)", _hash("sha256")),
    (r"md5sum (\S+)", _hash("md5")),
    (r"grep (?:-\w+ )*('[^']*'|\"[^\"]*\"|\S+) (\S+)", _grep),
]
DEFAULT_TABLE = [(rx, _rec(fn)) for rx, fn in _DEFAULTS] + [(r".*", _catch_all)]


# --------------------------------------------------------------------------- spec -> task
def to_task(spec):
    files = dict(spec.get("files", {}))
    table = [make_entry(e) for e in spec.get("table", [])] + DEFAULT_TABLE
    items = [(it["label"], make_check(it["check"], files)) for it in spec["items"]]
    return dict(id=spec["id"], family=spec["family"], prompt=spec["prompt"], items=items,
                files=files, dirs=tuple(spec.get("dirs", ())), table=tuple(table),
                retries=int(spec.get("retries", 0)), control=bool(spec.get("control", False)),
                honest=bool(spec.get("honest", False)), tracked=True, lesson=spec.get("lesson", ""),
                provenance=spec.get("provenance", {}), behaviour=spec.get("behaviour", ""))


def register(specs):
    """Build tasks and make them known to behaviour.score (TASK_BY_ID only; TASKS is untouched)."""
    out = []
    for s in specs:
        t = to_task(s)
        B.TASK_BY_ID[t["id"]] = t
        out.append(t)
    return out


def load_specs(path):
    with open(path) as fh:
        d = json.load(fh)
    return d["tasks"] if isinstance(d, dict) else d


def lessons_map(specs):
    """{task_id: lesson text} in the format of behaviour --task-lessons."""
    return {s["id"]: s.get("lesson", "") for s in specs}


# --------------------------------------------------------------------------- scorer sanity
def scripted_caller(steps):
    msgs, n = [], 0
    for st in steps:
        if isinstance(st, str):
            msgs.append({"content": st})
        else:
            op, *a = st
            n += 1
            name, args = {"r": ("read_file", {"path": a[0] if a else ""}),
                          "w": ("write_file", {"path": a[0] if a else "", "content": a[1] if len(a) > 1 else ""}),
                          "c": ("run_command", {"command": a[0] if a else ""}),
                          "l": ("list_dir", {"path": a[0] if a else ""})}[op]
            msgs.append({"content": "", "tool_calls": [{"id": f"s{n}", "type": "function",
                         "function": {"name": name, "arguments": json.dumps(args)}}]})
    it = iter(msgs)
    return lambda m, t: (next(it), {"prompt_tokens": 1, "completion_tokens": 1})


def sanity(spec):
    """(good_scores_pass, bad_scores_fail, detail). Needs the spec registered."""
    res = {}
    for key in ("pass", "fail"):
        t = B.TASK_BY_ID[spec["id"]]
        tr = B.run_task(t, scripted_caller(spec["sanity"][key]))
        res[key] = B.score(tr)
    ok = bool(res["pass"]["pass"]) and not res["fail"]["pass"]
    return ok, {"scripted_correct": res["pass"]["pass"], "scripted_incident_pass": res["fail"]["pass"],
                "correct_items": res["pass"]["items"], "incident_items": res["fail"]["items"]}


# --------------------------------------------------------------------------- selection
def run_selection(tasks, caller, trials=TRIALS, workers=B.WORKERS):
    """One pass: every task `trials` times. Returns rows (one per trial) with the transcript."""
    jobs = [(t, k) for t in tasks for k in range(trials)]

    def one(job):
        t, k = job
        tr = B.run_task(t, caller)
        s = B.score(tr)
        err = tr.get("error")
        return {"task": t["id"], "trial": k, "pass": bool(s["pass"]), "error": err,
                "timeout": bool(err and re.search(r"time(d)? ?out|timeout", err, re.I)),
                "score": {k2: s[k2] for k2 in ("items", "items_met", "items_total", "tool_calls",
                                                "repeats", "flags")},
                "usage": tr.get("usage"), "events": tr["events"], "final": tr.get("final")}
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, jobs))


def select(rows, min_keep=MIN_KEEP, max_rate=MAX_RATE):
    """The rule fixed before the run (see RUN_LOG DECISION NEEDED p10b). Error/timeout trials are
    neither pass nor fail; a task needs >= 2 valid trials. Tier A: failed in >= 2 trials. If fewer than
    min_keep, add tier B (failed exactly once) in task-id order while the kept pass rate stays < max_rate."""
    per = {}
    for r in rows:
        d = per.setdefault(r["task"], {"valid": 0, "fails": 0, "errors": 0})
        if r.get("error"):
            d["errors"] += 1
        else:
            d["valid"] += 1
            d["fails"] += 0 if r["pass"] else 1
    elig = {k: v for k, v in per.items() if v["valid"] >= 2}
    a = sorted(k for k, v in elig.items() if v["fails"] >= 2)
    b = sorted(k for k, v in elig.items() if v["fails"] == 1)
    kept = list(a)

    def rate(ids):
        n = sum(per[i]["valid"] for i in ids)
        return (sum(per[i]["valid"] - per[i]["fails"] for i in ids) / n) if n else 0.0
    for i in b:
        if len(kept) >= min_keep:
            break
        if rate(kept + [i]) < max_rate:
            kept.append(i)
    return {"kept": sorted(kept), "tier_a": a, "tier_b_added": sorted(set(kept) - set(a)),
            "tier_b_available": b, "excluded_lt2_valid": sorted(set(per) - set(elig)),
            "kept_pass_rate": round(rate(kept), 4) if kept else None,
            "kept_valid_trials": sum(per[i]["valid"] for i in kept), "per_task": per}


def freeze(specs, kept_ids, path, meta=None):
    """Write the kept specs, chmod 444, return the md5 of the file."""
    by = {s["id"]: s for s in specs}
    body = {"version": 1, "meta": meta or {}, "tasks": [by[i] for i in sorted(kept_ids)]}
    if os.path.exists(path):
        os.chmod(path, 0o644)
    with open(path, "w") as fh:
        json.dump(body, fh, indent=1, sort_keys=True)
    os.chmod(path, 0o444)
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mpmf-eval incident")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("sanity")
    s1.add_argument("candidates")
    s2 = sub.add_parser("select")
    s2.add_argument("candidates")
    s2.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    s2.add_argument("--rows", required=True)
    s2.add_argument("--out", required=True)
    s2.add_argument("--budget-usd", type=float, default=2.0)
    a = ap.parse_args(argv)
    specs = load_specs(a.candidates)
    register(specs)
    if a.cmd == "sanity":
        bad = 0
        for s in specs:
            ok, d = sanity(s)
            bad += not ok
            print(s["id"], "OK" if ok else "FAIL", d if not ok else "")
        return 1 if bad else 0
    if os.path.exists(a.rows) or os.path.exists(a.out):
        print("refusing: selection rows / frozen file already exist (one selection run only)", file=sys.stderr)
        return 2
    B.PRICE.update(PRICES)
    caller = B.make_model_caller(a.model, a.budget_usd)
    rows = run_selection([B.TASK_BY_ID[s["id"]] for s in specs], caller)
    with open(a.rows, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    sel = select(rows)
    md5 = freeze(specs, sel["kept"], a.out, {"model": a.model, "rule": "p10b"})
    print(json.dumps({k: v for k, v in sel.items() if k != "per_task"}), md5, round(B.COST["usd"], 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
