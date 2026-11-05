# Round 88 — temporal inventory + history renderer (propose-only)

## 0. Production state
- Gateway pid 375397 started **2026-11-04 19:03:23Z**; plugin patched at 19:1x and 02:2x,
  so the running gateway still executes the OLD `sync_turn`.
- Proof: real turn ending 19:37 logged `ingest_ms=6336 facts=5 skipped=0 dropped_ctx=0
  conv=turn_1793821019 msg=0` — conv is still `turn_<epoch>`, not a session id.
- **Reranker NOT switched off**: no explicit approval recorded in this round; live
  `recall.config.json` md5 09771691... unchanged (`rerank_skip_after_ms` stays 4500).
  Rollback if approved later: set `rerank_skip_after_ms: 0`, restart, restore 4500.
- Hermes plugin now under git: `/root/.hermes/plugins/mpmf_plus_user` repo, commit
  **95a0501** (today's sync_turn + history prepend).

## 1. Temporal inventory (live store, 24,415 nodes)
- Dated facts: **16,366** have `_fact_date`, **16,365** `_said_at`; `_source=ingest` 85
  (all 85 dated). `_fact_date_src` 15,564.
- Entities with >= 3 distinct dated days: **459** (car 51 days/160 facts, job 48/101,
  daughter 42/101, tesla 33/61, ...).
- Hub states with dated history: **0**. `catstate`/history wiki keys: **1**. No
  `_state`/`current_state` values exist.
- Edges: 46,569. Labels are mostly interned/unresolved; resolved ones are
  `shared_tags:*` (e.g. `shared_tags:location,residence` 3,600). **supersedes 0,
  coref:Before 0, DERIVED_FROM 0**.
- Live prefetch renders **only the top-8 relevance block**. There are no r69
  "Current state" / "Then (year)" sections in the live provider/plugin paths.

## 2. Order set
- 80 hubs attempted, **64 questions written** (writer saw only dated source messages;
  16 rejected by the parser), frozen at `questions_order.json`
  md5 **b1353104e875c623af3150da92cf70dd**.
- Baseline (64): acc **0.0469**, sequence-right **0.0312**, p90 1,648 ms, $0.106.

## 3. History renderer (config `history_mode`, default off)
- `provider.history_block()`: regex intent detector (when did / how has X changed /
  in what order / timeline / history of / used to / originally), entity resolution over
  `_entity`+query tokens, gathers dated facts, dedupes, sorts by `_fact_date` fallback
  `_said_at`, renders `## History of X` capped at 1,500 chars, prepended ahead of the
  relevance block in `plugin.prefetch`.
- Non-temporal identity: **20/20 blocks byte-identical** after normalising the
  `(N top hits, Xs)` latency line.
- Temporal smoke: "When did my car start overheating?" -> 1,460-char history, 157 dated facts listed.

## 4. Evaluation (copies, propose only)
| arm | n | base | history | delta | seq-right | hist fired | p90 |
|---|---|---|---|---|---|---|---|
| order (first 32) | 32 | 0.0312 | **0.2031** | **+17.19** | 0.0312 -> 0.0625 | 32/32 | 7,875 ms |
| core TUNE | 85 | 0.5882 | 0.5824 | -0.59 | - | - | 834 ms |

- Order McNemar on the 32: b=5 / c=1, **p=0.219** — above the 5.4-pt floor but not
  significant, so **no HOLDOUT** and no live config change.
- Core per-set (budget-truncated): history 55 = 0.0, now 20 = +2.5, then 10 = -10.0
  (one question — small n).
- History render costs ~6-7 s on 24k nodes (full store scan per temporal query).

## Budget
Cap $0.40. Spent ~$0.33 recorded + ~$0.05 partial killed run ≈ **$0.38**. The 64-question
order arm and the 110-question core arm were truncated (32 / 85) to stay inside the cap.

## Rollback
`/root/r88/backup/` for config; plugin repo commit 95a0501 can be reverted; history_mode
defaults to off in code and is absent from the live config.
