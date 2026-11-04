# Round 87 — ingest namespace + rerank experiments (propose-only)

## Ingest facts -> memory
Bug: the plugin calls `store(..., namespace="transcript")` for the raw turn; `ingest.ingest`
copied that namespace onto every extracted fact (`_ns` + `f_turn_...` name), so the user's
facts landed in transcript and default recall (`namespaces=('memory',)`) never saw them.

Fix (`src/mpmf_plus/ingest.py`, md5 9c93071e81fa192b9d1b56d56beb871a):
- `fact_ns` = memory unless the caller asks for ruminate/cognition/meta; transcript is remapped to memory.
- fact names are now `memfact_<conv>_<msg>_<n>` (classifier-safe: not `turn_`/`sess_`, not `fact_`).
- raw turn nodes (`turn_*`, 938 of them) stay in transcript.

Copy test: ingest(namespace="transcript") -> 2 facts, `_ns=memory`, names `memfact_...`,
`classify_namespace` = memory.

Live migration: 73 ingest facts since 2026-11-03 moved transcript -> memory in ONE
`write_txn` (0.91 s), all 73 verified in memory; 938 raw turn nodes untouched.
Re-score of the 12 recent questions on live: **0.000 -> 0.625** (7 correct, 1 partly,
4 declined), cost $0.0051. The prior transcript-enabled upper bound was 0.833.

Note: gateway still runs the old plugin/import; the code fix needs a restart. The
migration makes existing facts visible immediately.

## Rerank on the live path
Path: `plugin.prefetch` -> `_wrapper_recall` (`mod.recall`) -> `provider_entry` ->
`provider.recall`, `memory_arm()=="d20"` -> `_recall_d20` -> `_rerank_with_timeout` ->
`rerank()` cross-encoder.

Traced live recall (MPMF_TRACE_RERANK=1, store live):
status=done, 20 cosine candidates, rerank 5.6 s, scores 7.27 / 7.19 / 6.49 / ... / -0.71,
final 8 from RRF of cosine+rerank. The cross-encoder IS running.

Logging bug: `provider_entry` star-imports the provider, and `import *` skips names
starting with `_`; the plugin's `getattr(mod, "_LAST_TIMINGS", {})` was therefore always
empty -> rerank 0 / status "?". Fixed by exporting `_LAST_TIMINGS` from provider_entry
(md5 63a1b23df93c6ebb115ebf51ac3df621). Needs a gateway restart to take effect.

## Rerank experiments (copies, core TUNE n=110; baseline 0.550, p90 980 ms)
| variant | acc | delta | b/c | p | p90 ms | cost |
|---|---|---|---|---|---|---|
| reranker off (cosine only) | 0.5591 | +0.91 | 3/1 | 0.625 | 815 | $0.062 |
| rrf_k 20 -> 60 | 0.5273 | -2.27 | 2/3 | 1.000 | 770 | $0.059 |
| cos_pin_k=3 | 0.5455 | -0.45 | 1/2 | 1.000 | 1041 | $0.064 |

Effect probe (10 queries): rrf60 changes 9/10 orders, pin3 1/10, off 0/10 (order
unchanged, latency drops ~5.5 s — the reranker was not reordering those probes).
None beats the 5.4-pt floor or McNemar -> no HOLDOUT, no live config change.
`cos_pin_k` is implemented behind a key defaulting to 0; live config unchanged.
