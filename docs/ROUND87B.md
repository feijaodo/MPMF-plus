# Round 87b — restart proof, E2E, full-message ingest, reranker-off non-inferiority

## 1. Restart proof
Gateway pid 375397 started **2026-11-04 19:03:23Z** (log: Stopping 19:03:16 -> Starting 19:03:28).
Loaded files (both mtimes before the restart):
- `src/mpmf_plus/ingest.py` md5 **9c93071e81fa192b9d1b56d56beb871a**, mtime 16:25:31
- `/root/.mpmf/provider_entry.py` md5 **63a1b23df93c6ebb115ebf51ac3df621**, mtime 16:28:03
First real turn after restart (19:22:06, qlen 1348): prefetch 6908.4 ms, **rerank 3909.76 ms,
status=timeout** — rerank is now non-zero in coverage, so the provider_entry `_LAST_TIMINGS`
export is live. Its store.log line lands when the turn closes; the live E2E below exercised
the same path and produced `ingest_ms=4037 facts=2 skipped=0 dropped_ctx=0 conv=r87b-live`.

## 2. End-to-end test (suite + live)
`tests/test_r87b_ingest_e2e.py` (md5 **0228af71...**) — 1 passed in 6.16 s.
Live run: `sync_turn` -> `store(auto_extract=False, transcript)` + `store(full user, memory)`
-> ingest. Store.log: `ingest_ms=4037 facts=2 skipped=0 dropped_ctx=0`.
Nodes `memfact_r87b-live_0_0/1`, `_ns=memory`, text excludes `ATHENA-SECRET-87B`.
Prefetch("What is the Round 87b live ingest marker?") returns a 2,363-char block containing
`purple-heron-87b`; secret not present.

## 3. Full-message ingest + session id
Plugin md5 **1db828c2...**: raw transcript node keeps the 600+600 digest with `auto_extract=False`;
facts come from `user_content[:6000]`; assistant text is transcript-only; `set_ingest_session(sid)`
binds one conv id per Hermes session.
12-turn parity on copies (same store snapshot, real extraction):
| variant | facts | facts/turn | grounded (judged) | assistant leaks |
|---|---|---|---|---|
| old (600-char digest) | 27 | 2.25 | 70.4% | 0 |
| new (full user text) | 52 | 4.33 | **90.4%** | 0 |
Both write `_ns=memory` (ingest namespace fix shared); the parity isolates cap/assistant/session.

## 4. Recent re-score (12 questions, live)
**0.708** — 8 correct, 1 partly, 3 declined (was 0.625 / 4 declined; one is now partly).
Declines and why:
- 2026-11-03T23:51 OLLAMA_NUM_PARALLEL: key coverage in store 0.43; nearest node is a generic
  layout/minimal-changes fact -> **not extracted**.
- 2026-11-03T15:23 round 83c manual step (/restart): only a generic "restart with a new version
  of OpenGrok" node matches -> **merged/deduped**, no 83c fact.
- 2026-11-03T01:14 Espy swap verification: best hit is the raw transcript turn itself
  (`[session=... USER: ...]`), no memory fact -> **extraction skipped**, raw transcript only.

## 5. Reranker off — non-inferiority
| split | n | base | off | delta | b/c | p | p90 |
|---|---|---|---|---|---|---|---|
| TUNE | 110 | 0.5500 | 0.5591 | +0.91 | 3/1 | 0.625 | 815 ms |
| HOLDOUT | 110 | 0.5409 | 0.5455 | +0.45 | 1/1 | 1.000 | 1018 ms |
| **COMBINED** | **220** | **0.5455** | **0.5523** | **+0.68** | **4/2** | **0.6875** | **975 ms** |

Point estimate is non-inferior (+0.69 pts) but the CI is wide (McNemar p=0.69, 4 wins /
2 losses); it buys ~165 ms of p90 and removes a 5.6 s worst case. Propose only — live config
unchanged (`recall.config.json` md5 09771691...).

## Rollback
Backups in `/root/r87b/backup/` (plugin, provider, ingest, provider_entry, recall config) plus
`/root/r87b/live_store.mpmf`. Plugin is outside git; restore the backup copy and restart.
