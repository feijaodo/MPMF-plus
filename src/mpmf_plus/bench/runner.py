"""b0 run loop: pre-registration, per-corpus ingest, three arms, judge, results.jsonl, resumable.

Layout of the output directory (``--out``):
  prereg.json      the pre-registration (written once, before the first model call; never rewritten)
  meta.json        run facts that change as work is done (ingest stats per corpus, spend, status)
  results.jsonl    ONE line per (question, arm): {"id","category","arm","verdict","score","cost"}  (source of truth)
  answers.jsonl    {"id","arm","answer","judge_raw","context_chars"}  (the text behind each result line)
  results.json     assembled at the end from the two files above
  summary.md       regenerated from results.jsonl by ``summarize`` every SUMMARY_EVERY scored questions
  work/            throwaway MPMF_HOME and the bench stores (a store is deleted when its corpus is scored)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from . import datasets as D
from . import judges as J
from . import llm as L
from . import prompts as PR

ARMS = ("bm25", "off", "on")
SUMMARY_EVERY = 10
DEFAULT_SEED = 20261114


def jl_append(path: str, row: Dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with open(path, "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()


def jl_read(path: str) -> List[Dict[str, Any]]:
    """Tolerant reader: a truncated / garbled last line (killed mid-write) is skipped, never raised."""
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(r, dict) and "id" in r and "arm" in r:
                rows.append(r)
    return rows


def prereg(bench: str, qs: List[D.Question], planned: List[D.Question], seed: int, n: Optional[int], args: Any) -> Dict[str, Any]:
    cats: Dict[str, int] = {}
    for q in planned:
        cats[q.category] = cats.get(q.category, 0) + 1
    return {
        "benchmark": bench, "seed": seed, "pool": len(qs), "n_planned": len(planned), "n_requested": n,
        "sampling": "sorted ids -> random.Random(seed).shuffle -> first n (a prefix of the order is itself a uniform random sample)",
        "ids_md5": D.ids_md5(planned), "ids": [q.qid for q in planned], "category_counts": cats,
        "arms": {"bm25": "BM25 over raw turns, date-prefixed, top-%d, <= %d chars" % (0, 0),
                 "off": "MPMF per-turn ingest, history_mode=0 at recall", "on": "MPMF per-turn ingest (same store), history_mode=1 at recall"},
        "reader": L.READER_MODEL, "reader_temperature": 0, "reader_max_tokens": args.reader_tokens,
        "extractor": L.READER_MODEL, "judge": PR.JUDGE_MODELS[bench] if not args.judge else args.judge,
        "primary_metric": PR.PRIMARY[bench],
        "comparisons": ["on vs off (history)", "on vs bm25 (primary MPMF-vs-BM25)", "off vs bm25 (secondary)"],
        "tests": "McNemar exact two-sided on the binary verdict; paired sign-flip permutation (10000 perms, fixed seed) on numeric scores; 95% Wilson on accuracies",
        "cids_filter": args.cids, "cap_usd": args.cap, "ceiling_usd": round(0.8 * args.cap, 4), "rerank": bool(args.rerank),
        "fake_model": bool(args.fake_model),
    }


def write_prereg_once(out: str, evidence: Optional[str], pre: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-registration BEFORE any model call. An existing prereg.json wins (resume must not change it)."""
    p = os.path.join(out, "prereg.json")
    if os.path.exists(p):
        with open(p) as fh:
            old = json.load(fh)
        if old.get("ids_md5") != pre["ids_md5"]:
            raise SystemExit("REFUSE: prereg.json exists with ids_md5 %s, this run would use %s" % (old.get("ids_md5"), pre["ids_md5"]))
        return old
    os.makedirs(out, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(pre, fh, indent=1)
    if evidence:
        with open(evidence, "a") as fh:
            fh.write("\n=== PRE-REGISTRATION (%s) written %s before the first model call ===\n" % (
                pre["benchmark"], time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            slim = {k: v for k, v in pre.items() if k != "ids"}
            fh.write(json.dumps(slim, indent=1) + "\n")
            fh.write("sample ids (%d), md5 %s:\n%s\n" % (len(pre["ids"]), pre["ids_md5"], "\n".join(pre["ids"])))
    return pre


def _safe_unlink(path: str, root: str) -> None:
    rp, rr = os.path.realpath(path), os.path.realpath(root)
    if os.path.commonpath([rp, rr]) == rr and os.path.isfile(rp):
        os.unlink(rp)


def run(args: Any) -> int:
    """The ``run`` subcommand body (executes inside the bench environment, see __main__)."""
    from . import pipeline as PL
    out = args.out
    work = os.path.join(out, "work")
    home = os.environ["MPMF_HOME"]
    fake = bool(args.fake_model)
    if fake:
        from .. import provider as P, memory as M
        P._embed_floats_local = PL.fake_embed       # type: ignore[assignment]
        M._embed_floats_local = PL.fake_embed       # type: ignore[assignment]
    chat = L.get_chat(fake)
    corpora, qs = D.load(args.bench, args.data, include_cat5=args.include_cat5, lme_file=args.lme_file)
    if args.cids:
        keep = {c.strip() for c in args.cids.split(",") if c.strip()}
        qs = [q for q in qs if q.cid in keep]
        if not qs:
            raise SystemExit("--cids matched no question")
    planned = D.sample(qs, args.n, args.seed)
    pre = prereg(args.bench, qs, planned, args.seed, args.n, args)
    pre["arms"]["bm25"] = "BM25 over raw turns (date-prefixed), top-%d, <= %d chars, item cap %d" % (PL.BM25_K, PL.BM25_BUDGET, PL.BM25_ITEM)
    judge_model = pre["judge"]

    if args.dry_run:
        by_c: Dict[str, int] = {}
        for q in planned:
            by_c[q.cid] = by_c.get(q.cid, 0) + 1
        ncalls = sum(len(PL.ingest_calls(corpora[c])) for c in by_c)
        est = ncalls * 0.0003 + len(planned) * 3 * 0.0008 + len(planned) * 0.001
        print(json.dumps({"benchmark": args.bench, "pool": len(qs), "planned": len(planned), "corpora": len(by_c),
                          "ingest_calls": ncalls, "ids_md5": pre["ids_md5"], "categories": pre["category_counts"],
                          "reader": L.READER_MODEL, "judge": judge_model,
                          "est_cost_usd_rough": round(est, 2)}, indent=1))
        return 0

    pre = write_prereg_once(out, args.evidence, pre)
    os.makedirs(work, exist_ok=True)
    res_p, ans_p, meta_p = (os.path.join(out, f) for f in ("results.jsonl", "answers.jsonl", "meta.json"))
    meta: Dict[str, Any] = {"ingest": {}, "ingest_cost": 0.0}
    if os.path.exists(meta_p):
        with open(meta_p) as fh:
            meta = json.load(fh)
    done = {(r["id"], r["arm"]) for r in jl_read(res_p)}
    prior = sum(float(r.get("cost") or 0) for r in jl_read(res_p)) + float(meta.get("ingest_cost", 0.0))
    meter = L.Meter()
    meter.usd = prior
    ceiling = float(pre["ceiling_usd"])
    t_end = time.time() + args.time_limit if args.time_limit else None
    stop_reason: List[str] = []

    def stop() -> bool:
        if stop_reason:
            return True
        if meter.usd >= ceiling:
            stop_reason.append("spend ceiling $%.2f reached (80%% of the $%.2f cap)" % (ceiling, args.cap))
        elif t_end and time.time() > t_end:
            stop_reason.append("time limit %ds reached" % args.time_limit)
        return bool(stop_reason)

    lock = threading.Lock()
    scored = [0]
    extractor_cache = os.path.join(out, "extractions.jsonl")
    ext = PL.Extractor(chat, L.READER_MODEL, meter, extractor_cache, fake=fake)
    recaller = None

    def regen() -> None:
        from . import report
        try:
            report.write_summary(out, status="IN PROGRESS")
        except Exception as e:  # noqa: BLE001  (a report bug must never stop the run)
            print("summary regen failed: %s" % e, file=sys.stderr)

    by_corpus: Dict[str, List[D.Question]] = {}
    for q in planned:
        by_corpus.setdefault(q.cid, []).append(q)

    def answer_one(q: D.Question, arm: str, ctx: str) -> None:
        if stop():
            return
        prompt = PR.reader_prompt(q, ctx)
        cost = 0.0
        try:
            r = chat(prompt, L.READER_MODEL, max_tokens=args.reader_tokens, meter=meter)
            ans, cost = r["text"], r["cost"]
        except Exception as e:  # noqa: BLE001
            jl_append(res_p, {"id": q.qid, "category": q.category, "arm": arm, "verdict": None, "score": None, "cost": cost}, lock)
            jl_append(ans_p, {"id": q.qid, "arm": arm, "answer": "", "judge_raw": "reader error: %s" % str(e)[:200],
                              "context_chars": len(ctx)}, lock)
            return
        j = J.judge(q, ans, chat, judge_model)
        meter.add(j["cost"])             # reader + extractor calls are metered inside chat(); the judge's here
        cost += j["cost"]
        jl_append(ans_p, {"id": q.qid, "arm": arm, "answer": ans, "judge_raw": j["raw"], "context_chars": len(ctx)}, lock)
        jl_append(res_p, {"id": q.qid, "category": q.category, "arm": arm, "verdict": j["verdict"], "score": j["score"],
                          "cost": round(cost, 8)}, lock)
        with lock:
            scored[0] += 1
            if scored[0] % SUMMARY_EVERY == 0:
                regen()

    for cid, cqs in by_corpus.items():
        if stop():
            break
        todo = [q for q in cqs if any((q.qid, a) not in done for a in ARMS)]
        if not todo:
            continue
        corpus = corpora[cid]
        store_path = os.path.join(work, "stores", hashlib.md5(cid.encode()).hexdigest()[:12] + ".mpmf")
        t0, c0 = time.time(), meter.usd
        if os.path.exists(store_path):
            _safe_unlink(store_path, work)            # a partial store from a killed run: ours, rebuilt (extractions are cached)
        st = PL.ingest_corpus(corpus, store_path, ext, home=home, workers=args.workers, stop=stop, fake=fake,
                              progress=lambda i, n, cid=cid: print("ingest %s %d/%d spend=$%.3f" % (cid, i, n, meter.usd), flush=True))
        st["cost"] = round(meter.usd - c0, 6)
        st["seconds"] = round(time.time() - t0, 1)
        meta["ingest"][cid] = st
        meta["ingest_cost"] = round(float(meta.get("ingest_cost", 0.0)) + st["cost"], 6)
        with open(meta_p, "w") as fh:
            json.dump(meta, fh, indent=1)
        if st["stopped"] or stop():
            _safe_unlink(store_path, work)
            break
        if st["nodes"] <= 0:
            print("WARNING: corpus %s ingested into 0 nodes: MPMF arms will be empty for it" % cid, file=sys.stderr)
        if recaller is None:
            recaller = PL.Recaller()
        bm = PL.bm25_index(corpus)
        ex = ThreadPoolExecutor(max_workers=max(1, args.workers))
        futs = []
        for q in todo:
            if stop():
                break
            try:
                ctxs = {"bm25": PL.bm25_context(bm, q.question)}
                if any((q.qid, a) not in done for a in ("off", "on")):
                    ctxs["off"] = recaller.context(q.question, history=False)
                    ctxs["on"] = recaller.context(q.question, history=True)
            except Exception as e:  # noqa: BLE001  (a recall failure is recorded as an unscored pair, not hidden)
                print("recall failed for %s: %s" % (q.qid, e), file=sys.stderr)
                for a in ARMS:
                    if (q.qid, a) not in done:
                        jl_append(res_p, {"id": q.qid, "category": q.category, "arm": a, "verdict": None, "score": None, "cost": 0.0}, lock)
                continue
            for a in ARMS:
                if (q.qid, a) not in done and a in ctxs:
                    futs.append(ex.submit(answer_one, q, a, ctxs[a]))
        ex.shutdown(wait=True)
        if not args.keep_store:
            _safe_unlink(store_path, work)
        done = {(r["id"], r["arm"]) for r in jl_read(res_p)}
        with open(meta_p, "w") as fh:
            meta["spend_total"] = round(meter.usd, 6)
            json.dump(meta, fh, indent=1)

    from . import report
    complete = all((q.qid, a) in {(r["id"], r["arm"]) for r in jl_read(res_p)} for q in planned for a in ARMS)
    status = "COMPLETE" if complete else "PARTIAL, stopped at " + (stop_reason[0] if stop_reason else "an unfinished corpus")
    meta["spend_total"] = round(meter.usd, 6)
    meta["status"] = status
    with open(meta_p, "w") as fh:
        json.dump(meta, fh, indent=1)
    report.write_results_json(out)
    report.write_summary(out, status=status)
    print(status, "spend=$%.4f" % meter.usd)
    return 0
