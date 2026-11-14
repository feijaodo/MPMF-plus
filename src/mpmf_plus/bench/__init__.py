"""mpmf_plus.bench (b0): benchmark harness. BM25 over raw turns vs MPMF history off vs MPMF history on.

    python -m mpmf_plus.bench --help

Modules: datasets (loaders, seeded sample), pipeline (real per-turn ingest, BM25, recall), prompts (reader prompts and
the published-setup notes), judges (official judge prompts), stats (Wilson, McNemar, sign-flip), textmetrics (token F1,
ROUGE-L, BM25), llm (OpenRouter client + fake), runner (resumable run loop), report (summary.md from results.jsonl).
Nothing is imported at package import time: the bench environment (MPMF_HOME ...) must be set before mpmf_plus loads.
"""
