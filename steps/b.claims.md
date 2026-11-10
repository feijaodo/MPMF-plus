# Step B claim table (README.md / CHANGELOG.md, as at HEAD bfceabe)

Sources (only): D = /root/ops-mpmf/r92_5b/derived_dev.json, R = /root/ops-mpmf/RETIRED_HELDOUT.md.
Code check for the auto-revert threshold (item 3): src/mpmf_plus/eval/nightly.py (not edited).
Status: verified / WRONG (fixed from rows) / unsupported (removed from README; CHANGELOG: "evidence not retained").

## WRONG (4, corrected from D)
| doc | old | new | source |
|---|---|---|---|
| README order table, off prefetch p90 | 1,464 ms | 1,533 ms | D order.off.bench_prefetch_ms.p90 = 1532.7 |
| README order table, on prefetch p90 | 1,257 ms | 1,265 ms | D order.on.bench_prefetch_ms.p90 = 1265.3 |
| README live latency (n, p50, p90, max) | n=155, 1.6 s, 4.7 s, 6.9 s ("real sessions") | n=25, 1.8 s, 4.3 s, 5.0 s | D prefetch_latency.real_sessions_only; the old figure was the "all_except_scorecard" class (n=158, incl. r90e bench + watchdog probes) |
| README order set "at least 3 dated facts" | >= 3 | 2 to 5 distinct key dates (2:4, 3:30, 4:23, 5:7) | D order.distinct_key_dates_per_question |
Knock-on corrections of the same figures: "146 of 155 from one day" -> "15 of 25" (D real_sessions_by_day); "p90 4.7 s over 155 calls" -> "4.3 s over 25 calls" in Deliberately-not-claimed; idle-host footnote "1.75-1.96 s earlier runs" removed (unsupported).

## Auto-revert (item 3) - docs vs code
nightly.py: auto-revert (_check, REVERT_DROP = 0.054, p < 0.10, MIN_PAIRS 32) covers history_mode and the reranker; the separate nightly drift ALERT uses DROP_PTS = 0.08 (8 points), p < 0.10. README previously said only ">= 5.4" for revert and CHANGELOG ">= 5.4" - both match the code's revert path. README now states both: revert 5.4 pts, alert 8 pts. The task text's "8 points" matches the alert only, not the revert; docs follow the code. No code changed. See RUN_LOG DECISION NEEDED.

## Verified
| doc | claim | source |
|---|---|---|
| README | order score off 0.8% / on 68.0% | D order.*.lenient_score_pct 0.78 / 67.97 |
| README | strict 0/64 (0%) vs 32/64 (50%) | D strict_correct 0 / 32, strict_pct |
| README | verdicts off 0/1/59/4, on 32/23/5/4 (correct/partly/declined/wrong) | D order.*.verdicts |
| README | Wilson 95% 38.1-61.9% (on), 0-5.7% (off) | D strict_wilson95_pct |
| README | McNemar b=32, c=0, p<1e-6 | D mcnemar_strict p=4.7e-10 |
| README | 64 questions, 59 of 64 declines (off) | D n_questions, off.verdicts.declined |
| README + CHANGELOG | 6,326 of 24,339 nodes (26.0%), cos<0.99; 4,350 <0.90; mean 0.954; byte-exact | D vector_audit |
| README | latency: 25 calls, 3 sessions, p50 1.8 s, p90 4.3 s, max 5.0 s, nearest-rank, since 2026-11-06, 15 on 2026-11-06 | D prefetch_latency |
| README | idle-host p90 1,265 / 1,533 ms | D order.*.bench_prefetch_ms |
| README + CHANGELOG | auto-revert >= 5.4 pts, p<0.10; drift alert >= 8 pts; 32 pair minimum (CHANGELOG) | nightly.py (code) |
| CHANGELOG | 7 cron dry-runs | deploy.sh (code) |
| README | held-out retired (harness bug, coverage 3/10; dev control 9/10) - now stated, nothing reported from it | R |
| README | requires Python >=3.10, max_length=512, 60 s lock timeout, 1024-d mxbai, port 11434 | code/pyproject (config values, not results) |

## Unsupported (removed from README; CHANGELOG marked)
| doc | claim | action |
|---|---|---|
| README | 3,402-memory deployment; 56%/59% vs 37%/30% (intro) | removed; intro now uses order figures |
| README | 55.8% vs 36.7% (n=49), 58.6% vs 30.1%, McNemar 22/2 p<0.0001 | removed (not in D/R) |
| README | 129 questions, md5 186e1076..., 27.9% vs 5.4%, 33/4, dated 42-58% vs 5%, order 0-7%, n=15-60 | removed |
| README | extraction prompt 0.99x facts, rerun 1.03x, n=30 | removed |
| README | HOLDOUT 2026-11-09: 40.6% vs 42.2%, b=2 c=1 p=1.0; baseline corrected 46.9%->42.2%; "64-question HOLDOUT" | removed |
| README | earlier 32.8% figure | removed |
| README | torch 1.2 GB, sentence-transformers 4.7 MB, transformers 56 MB, weights ~88 MB | removed |
| README | write-scaling table (7,069/14,138/28,298 nodes; 40.7/76.8/149.0 MB; 1,722/3,964/8,632 ms; 2.3x/5.0x) | removed |
| README | 2 procs x 25 writes, +50 nodes (6522->6572); "98/98 identical" | numbers removed; artifacts still linked |
| README | "numbers above measured before repair" sentence | removed with the older results |
| CHANGELOG | "paired 0 of 64 questions" (nightly auto-revert bug) | kept; marked "evidence not retained" |
