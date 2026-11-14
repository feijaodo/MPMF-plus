"""python -m mpmf_plus.bench {run,summarize} ...

    python -m mpmf_plus.bench run locomo --data DIR --out OUT --n 120 --seed 7 --cap 8 --fake-model --dry-run
    python -m mpmf_plus.bench summarize locomo --out OUT

``run`` re-executes itself in a child process whose MPMF_HOME / recall config / health dir point at a throwaway
directory under OUT/work, so no live config or store is ever read (the env must be set before mpmf_plus is
imported). Datasets and output dirs are arguments; nothing is hard-coded.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Optional

from . import datasets as D

DEFAULT_SEED = 20261114


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m mpmf_plus.bench",
                                description="b0 benchmark harness: BM25 raw turns vs MPMF history off vs MPMF history on.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="ingest, answer with all three arms, judge, append results.jsonl")
    r.add_argument("bench", choices=D.BENCHMARKS)
    r.add_argument("--data", required=True, help="directory holding the downloaded datasets")
    r.add_argument("--out", required=True, help="output directory (results.jsonl, summary.md, work/)")
    r.add_argument("--n", type=int, default=None, help="sample size (prefix of the seeded shuffle); default all")
    r.add_argument("--seed", type=int, default=DEFAULT_SEED)
    r.add_argument("--cap", type=float, default=8.0, help="step cap in USD; the loop stops at 80%% of it")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--time-limit", type=int, default=0, help="seconds; 0 = none")
    r.add_argument("--reader-tokens", type=int, default=400)
    r.add_argument("--judge", default=None, help="override the judge slug (recorded in the pre-registration)")
    r.add_argument("--evidence", default=None, help="evidence file that receives the pre-registration")
    r.add_argument("--lme-file", default="longmemeval_oracle.json")
    r.add_argument("--include-cat5", action="store_true", help="LoCoMo: also the adversarial category (default off, as Mem0)")
    r.add_argument("--rerank", action="store_true", help="use the cross-encoder rerank in MPMF recall (default off: deterministic)")
    r.add_argument("--cids", default=None, help="comma list: restrict the pool to these conversation/corpus ids (smoke runs)")
    r.add_argument("--keep-store", action="store_true")
    r.add_argument("--dry-run", action="store_true", help="load, sample, print the plan and a rough cost; no model call, no results")
    r.add_argument("--fake-model", action="store_true", help="deterministic stand-in reader/judge/extractor and hash embedder (tests)")
    s = sub.add_parser("summarize", help="regenerate summary.md from results.jsonl (works on a truncated file)")
    s.add_argument("bench", choices=D.BENCHMARKS)
    s.add_argument("--out", required=True, help="directory with results.jsonl")
    s.add_argument("--status", default="IN PROGRESS", help="IN PROGRESS | COMPLETE | 'PARTIAL, stopped at <reason>'")
    return p


def _child_env(out: str) -> dict:
    from . import pipeline as PL
    home = os.path.join(os.path.abspath(out), "work", "home")
    env = dict(os.environ)
    env.update(PL.bench_env(home))
    return env


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "summarize":
        from . import report
        pre_p = os.path.join(args.out, "prereg.json")
        if not os.path.exists(pre_p):          # a bare results.jsonl: still summarise it, name from the argument
            os.makedirs(args.out, exist_ok=True)
        report.write_summary(args.out, status=args.status)
        print(os.path.join(args.out, "summary.md"))
        return 0
    if os.environ.get("MPMF_BENCH_CHILD") != "1":
        from . import pipeline as PL
        home = os.path.join(os.path.abspath(args.out), "work", "home")
        PL.write_recall_config(home, rerank=bool(args.rerank))
        cmd = [sys.executable, "-m", "mpmf_plus.bench"] + (argv if argv is not None else sys.argv[1:])
        return subprocess.call(cmd, env=_child_env(args.out))
    from . import runner
    return runner.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
