"""b0 report: results.jsonl -> summary.md (+ results.json). Works on a truncated / partial results.jsonl.

The first line of summary.md is the status line the runner contract asks for:
``STATUS: IN PROGRESS n=<k> of <planned>`` | ``STATUS: COMPLETE`` | ``STATUS: PARTIAL, stopped at <reason>``.
Only questions scored by ALL three arms are used for the tests and the headline table (paired); questions with a
single arm scored so far are counted in ``scored so far`` and nowhere else. A verdict of null (judge/reader
failure) removes that question from every comparison it would enter and is listed, never counted wrong.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from . import prompts as PR
from . import stats as S

ARMS = ("bm25", "off", "on")
LABEL = {"bm25": "BM25 raw turns", "off": "MPMF history OFF", "on": "MPMF history ON"}


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001  (a line cut by a kill)
                continue
            if isinstance(r, dict) and "id" in r and "arm" in r:
                rows.append(r)
    return rows


def _load(out: str) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    pre: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}
    for name, dst in (("prereg.json", pre), ("meta.json", meta)):
        p = os.path.join(out, name)
        if os.path.exists(p):
            try:
                with open(p) as fh:
                    dst.update(json.load(fh))
            except Exception:  # noqa: BLE001
                pass
    return pre, meta, _read_jsonl(os.path.join(out, "results.jsonl"))


def _pct(x: float) -> str:
    return "%.1f%%" % (100 * x)


def _acc_cell(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = S.wilson(k, n)
    return "%s (%d/%d, CI %s-%s)" % (_pct(k / n), k, n, _pct(lo), _pct(hi))


def analyse(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pure function over the result lines: everything the report prints."""
    by: Dict[str, Dict[str, Dict[str, Any]]] = {a: {} for a in ARMS}
    cat: Dict[str, str] = {}
    cost = {a: 0.0 for a in ARMS}
    for r in rows:
        a = r.get("arm")
        if a not in by:
            continue
        by[a][r["id"]] = r
        cat[r["id"]] = r.get("category", "")
        cost[a] += float(r.get("cost") or 0.0)
    scored_by_arm = {a: sum(1 for x in by[a].values() if x.get("verdict") is not None) for a in ARMS}
    all3 = [i for i in by["bm25"] if i in by["off"] and i in by["on"]
            and all(by[a][i].get("verdict") is not None for a in ARMS)]
    all3.sort()
    ver = {a: {i: by[a][i]["verdict"] for i in all3} for a in ARMS}
    sco = {a: {i: by[a][i].get("score") for i in all3} for a in ARMS}
    has_score = any(v is not None for a in ARMS for v in sco[a].values())
    cats = sorted({cat[i] for i in all3})
    per_cat: Dict[str, Dict[str, Tuple[int, int]]] = {}
    for c in cats:
        ids = [i for i in all3 if cat[i] == c]
        per_cat[c] = {a: (sum(1 for i in ids if ver[a][i]), len(ids)) for a in ARMS}
    n = len(all3)
    tot = {a: (sum(1 for i in all3 if ver[a][i]), n) for a in ARMS}
    cmp = {}
    for name, (x, y) in {"on vs off (history)": ("off", "on"), "on vs bm25 (primary)": ("bm25", "on"),
                         "off vs bm25 (secondary)": ("bm25", "off")}.items():
        cmp[name] = S.paired_binary(ver[x], ver[y])
        if has_score:
            sc_x = {i: v for i, v in sco[x].items() if v is not None}
            sc_y = {i: v for i, v in sco[y].items() if v is not None}
            cmp[name]["score"] = S.paired_scores(sc_x, sc_y)
    means = {a: (S.mean(v for v in sco[a].values() if v is not None) if has_score else None) for a in ARMS}
    nulls = {a: sum(1 for x in by[a].values() if x.get("verdict") is None) for a in ARMS}
    return {"n_paired": n, "scored_by_arm": scored_by_arm, "per_cat": per_cat, "totals": tot, "cmp": cmp,
            "score_means": means, "has_score": has_score, "cost": cost, "nulls": nulls, "ids": all3,
            "verdicts": ver, "scores": sco}


def _bench_name(pre: Dict[str, Any], out: str) -> str:
    return pre.get("benchmark") or os.path.basename(os.path.normpath(out))


def render(out: str, status: str) -> str:
    pre, meta, rows = _load(out)
    A = analyse(rows)
    bench = _bench_name(pre, out)
    planned = int(pre.get("n_planned") or 0)
    n_questions_started = len({r["id"] for r in rows})
    if status.startswith("IN PROGRESS"):
        first = "STATUS: IN PROGRESS n=%d of %d" % (A["n_paired"], planned)
    else:
        first = "STATUS: " + status
    L: List[str] = [first, "", "# %s: BM25 raw turns vs MPMF history off vs MPMF history on" % bench, ""]
    spend = sum(A["cost"].values()) + float(meta.get("ingest_cost", 0.0))
    L += ["Reader %s (temperature 0), judge %s, extractor %s. Sampling seed %s, sample ids md5 %s, %d of %d pool questions planned."
          % (pre.get("reader", "?"), pre.get("judge", "?"), pre.get("extractor", "?"), pre.get("seed", "?"),
             pre.get("ids_md5", "?"), planned, pre.get("pool", 0)),
          "Primary metric: %s." % pre.get("primary_metric", PR.PRIMARY.get(bench, "?")),
          "Scored by all three arms (the paired set every test below uses): n = %d. Scored so far per arm: %s. "
          "Questions started: %d. Null verdicts (judge/reader failure, excluded, not counted wrong): %s."
          % (A["n_paired"], ", ".join("%s %d" % (a, A["scored_by_arm"][a]) for a in ARMS), n_questions_started,
             ", ".join("%s %d" % (a, A["nulls"][a]) for a in ARMS)),
          "The two MPMF arms share ONE ingest per conversation: history_mode is a recall-time switch.", ""]
    if A["n_paired"] == 0:
        L += ["No question has been scored by all three arms yet.", ""]
    else:
        L += ["## Accuracy (verdict = the binary label; 95% Wilson interval)", "",
              "| category | n | " + " | ".join(LABEL[a] for a in ARMS) + " |", "|---|---|" + "---|" * len(ARMS)]
        for c, d in A["per_cat"].items():
            L.append("| %s | %d | %s |" % (c, d["bm25"][1], " | ".join(_acc_cell(*d[a]) for a in ARMS)))
        L.append("| **all** | %d | %s |" % (A["n_paired"], " | ".join(_acc_cell(*A["totals"][a]) for a in ARMS)))
        L.append("")
        if A["has_score"]:
            L += ["Secondary numeric score, mean per arm over the same questions: " +
                  ", ".join("%s %.4f" % (LABEL[a], A["score_means"][a]) for a in ARMS) +
                  ". It is not binary, so its comparisons use a paired sign-flip permutation test, below.", ""]
        L += ["## Paired tests (same questions in every arm)", "",
              "b = first arm wrong and second right, c = first right and second wrong; McNemar exact two-sided.", "",
              "| comparison (first -> second) | n | b | c | p (McNemar exact) |", "|---|---|---|---|---|"]
        for name, r in A["cmp"].items():
            L.append("| %s | %d | %d | %d | %s |" % (name, r["n"], r["b"], r["c"], r["p"]))
        if A["has_score"]:
            L += ["", "| comparison | n | mean diff (second - first) | wins / losses | p (%s) |" % (
                  "paired sign-flip permutation, %d perms, seed %d" % (S.PERMUTATIONS, S.SEED)), "|---|---|---|---|---|"]
            for name, r in A["cmp"].items():
                s = r.get("score")
                if s:
                    L.append("| %s | %d | %+.4f | %d / %d | %s |" % (name, s["n"], s["mean_diff"], s["wins"], s["losses"], s["p"]))
        L.append("")
    L += ["## Cost", "",
          "Reader + judge by arm: " + ", ".join("%s $%.4f" % (a, A["cost"][a]) for a in ARMS) +
          "; ingest (extraction calls) $%.4f; total $%.4f." % (float(meta.get("ingest_cost", 0.0)), spend), ""]
    ing = meta.get("ingest") or {}
    if ing:
        bad = sum(v.get("failed", 0) for v in ing.values())
        L += ["Ingest: %d conversation store(s), %d turns, %d facts written, %d dedup-skipped, %d failed turns%s."
              % (len(ing), sum(v.get("turns", 0) for v in ing.values()), sum(v.get("written", 0) for v in ing.values()),
                 sum(v.get("skipped", 0) for v in ing.values()), bad,
                 " (first errors: %s)" % "; ".join(e for v in ing.values() for e in v.get("errors", [])[:1][:1])[:300] if bad else ""), ""]
    L += ["## How our reader and judge differ from the published setup", "", PR.DIFFERENCES.get(bench, "(no notes for this benchmark)"), ""]
    L += ["## Caveats", "",
          "- Same reader and prompt for all arms; the arms differ only in the retrieved context. BM25 is a small pure-python Okapi "
          "BM25 (k1 1.5, b 0.75) over raw turns with the same item count and character budget as the MPMF block.",
          "- A smaller true result is reported as it came out; no arm was re-run.",
          "- Judge verdicts come from a single judge call per answer (temperature 0); judge noise is not measured here."]
    if status.startswith("PARTIAL") or status.startswith("IN PROGRESS"):
        L.append("- This report covers a pre-registered PREFIX of the seeded order (a uniform random subsample), not the full planned sample.")
    return "\n".join(L) + "\n"


def write_summary(out: str, status: str = "IN PROGRESS") -> str:
    text = render(out, status)
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, ".summary.md.tmp")
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, os.path.join(out, "summary.md"))
    return text


def write_results_json(out: str) -> str:
    pre, meta, rows = _load(out)
    answers = {(r["id"], r["arm"]): r for r in _read_jsonl(os.path.join(out, "answers.jsonl"))}
    per: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        q = per.setdefault(r["id"], {"id": r["id"], "category": r.get("category", ""), "arms": {}})
        a = answers.get((r["id"], r["arm"]), {})
        q["arms"][r["arm"]] = {"answer": a.get("answer", ""), "verdict": r.get("verdict"), "score": r.get("score"),
                               "cost": r.get("cost")}
    p = os.path.join(out, "results.json")
    with open(p, "w") as fh:
        json.dump({"prereg": {k: v for k, v in pre.items() if k != "ids"}, "meta": meta,
                   "questions": list(per.values())}, fh, indent=1, ensure_ascii=False)
    return p
