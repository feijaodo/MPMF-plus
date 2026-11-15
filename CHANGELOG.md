# Changelog

## Unreleased

### Added
- **Model-agnostic core (r98): capability is read from the REPLY, never from a model name.** New `mpmf_plus/llm_caps.py` is the only place that knows how a provider shapes a reply: `reasoning_text(msg)` reads `reasoning`, `reasoning_content`, `reasoning_details` (list or dict), `reasoning_text`, `thinking`, `reasoning_items`; `usage_cost(data)` returns the `usage.cost` the provider reported; `post_with_option_fallback(...)` POSTs a body, and when the endpoint answers 400/404/422/501 naming a request option ("Unsupported parameter: reasoning") it drops that option, retries once, and remembers it per endpoint. Fixed call sites that had decided behaviour from a slug: `bench/llm.chat` no longer pre-emptively disables reasoning for slugs containing `deepseek` (it switches the channel off only after a reply whose content is empty and whose trace is present); `provider.call_extractor` and `memory._llm_complete` keep sending the extraction hint but survive endpoints without the option and no longer hard-code which field carries the trace; `capture.py` reads whichever reasoning column the state.db and the provider use (schema-tolerant); `thinking.LLM_MODEL` is env-overridable (`MPMF_EXTRACT_MODEL`) like `EXTRACT_MODEL`. Test `test_no_name_based_model_logic_left_in_the_core` refuses any future `"slug" in model` / `model.startswith("slug")` / `model == "slug"` in `src/mpmf_plus`. Tests: `tests/test_r98_model_agnostic.py`.
- **Every trace and every fact records the model that produced it (r98).** Thinking nodes gain `model` in their wiki (`store_thinking(..., model=)`; capture passes the session model and tolerates an older shim without the kwarg) and `recall_thinking` returns it; fact nodes already carried `_ingest_model`/`_extraction_model`.
- **Host scripts moved into the package (r98).** `~/.hermes/scripts/mpmf_daily_checks.py` is now a shim over `mpmf_plus.daily` (`python3 -m mpmf_plus.daily [--dry-run]`, same exit code, capture counts line kept), and `~/.mpmf/gate_eval/vet40.py` a shim over `mpmf_plus.eval.vet40` (caches resolve from `MPMF_GATE_EVAL` / `$MPMF_HOME/gate_eval`; no literal home path). `cognition._panel_prompt` and the intake gate now import the shipped judge prompt. Tests: `tests/test_r98_package_moves.py`.
- **Thinking capture has no model allowlist (r97).** `mpmf_plus.capture` used to take only final replies from `deepseek-v4.1-flash` / `deepseek-v4-flash` sessions, so every other model's reasoning was silently never captured. The `models` default and the SQL model clause are gone (a stale `models` key in `thinking_capture.config.json` is ignored); any final reply (no tool calls, real user turn, source not cron/subagent) with reasoning >= `min_reasoning_chars` is captured. Everything the cursor passes over but does not capture is counted by reason (`no_reasoning`, `reasoning_too_short`, `not_final_reply`, `excluded_source`, `synthetic_user_turn`, `duplicate_reasoning`) and model: in `thinking_capture_state.json` (`skips`, per UTC day, 14 days kept), as one line per run in `health/thinking_capture.log`, in the `--once` result, and in the `--dry-run` report. The daily check prints one info line (`health.capture_skip_report`, via `health.daily_info`) with captured and skipped counts; it never changes the exit code. Tests: `tests/test_r97_capture_any_model.py`.
- **`append_segments` is now PER STORE: `append_segments_stores` (a list of store paths).** A store uses append-only segments only if its resolved real path is in that list (or a caller passes `append_segments=` to `MemoryManager`, or the process sets `MPMF_APPEND_SEGMENTS`; tests, `mpmf.compact` and `mpmf.rollback` keep working). **The old global config key `append_segments: true` no longer switches any store** (it used to switch every store any process opened, Espy/Ayu/shared/thinking included; it was never turned on anywhere, so no store changes). Code from before this change ignores the new key (proved on a deploy-r95 worktree, with the old key as the negative control), so a late arming cannot hurt an old process; the one thing arming must wait for is a gateway that is still running old code, so every provider process now writes a stamp `health/loaded_rev/<pid>.json` (`code_rev`, `per_store_scope`, process start; failure-proof, dead pids pruned) which the segments guard checks before it adds a store to the list. `./deploy.sh rollback` now compacts the store and then removes THAT store from the list (other entries kept; the legacy key is written `false`). Default stays OFF. Tests: `tests/test_a1_per_store.py`.
- **Safe rollback: `./deploy.sh rollback <tag> [--dry-run]`.** Code from before `append_segments` cannot read a segmented store, so the rollback first compacts a segment-format store to the plain legacy image under the store's write lock (`mpmf.rollback`: counts, name index and the full node/edge/vector signature are compared before the atomic rename; a difference aborts and the live tree is not moved), writes an explicit `"append_segments": false` into the config, and only then checks the tag out. A tag the repo does not have is refused. `append_segments` itself stays default OFF. Tests: `tests/test_g3_rollback.py` (including a real v0.2.0 worktree reading the compacted store and refusing the segmented one).
- **`history_mode` is now ON by default** (`paths.DEFAULTS`, `config.example.json`); one resolver (`paths.history_mode`: env `MPMF_HISTORY_MODE` > file > default) for every reader; the nightly auto-revert writes `"history_mode": false` instead of removing the key; README *Switches* table lists every flag with its measured result, cost, default and toggle; non-temporal blocks byte-identical off vs on (23 of 23, `steps/g2.notes.md`).
- README: dev set and held-out v2 side by side, strict vs strict (dev 0/64 -> 32/64, held-out 5/64 -> 30/64); the intro now compares the strict pair.
- **Second-pass retrieval (`second_pass`, default OFF, RESEARCH).** With the flag on, the
  `mpmf_recall` tool the model can call while reasoning returns the same rendering as
  prefetch (not 300-char snippets), never repeats a memory already in the turn's block,
  and is capped at `second_pass_max_calls` (2) per turn; its description tells the model
  when to use it. Flag off = tool schema and handler exactly as before. TUNE (98 q, two
  full runs): no tool 0.434/0.424 -> new tool 0.510/0.510, but the already-shipped tool
  gives 0.520/0.526, so the flag adds nothing over it; stays off. See steps/p18.notes.md.
- **Append-only write segments (`append_segments`, default OFF).** With the flag on,
  a save appends one CRC-framed delta segment to the end of the store instead of
  rewriting the whole image (measured on the 131 MB store: median write 776 ms ->
  42 ms, 131 MB -> 4 KB written). The base image keeps its layout and checksum;
  flag OFF writes exactly the old bytes. A torn tail from a crash is ignored on
  read and truncated by the next writer. After `compact_after_segments` (default
  5000) the writer folds the segments into a new base itself, and
  `python -m mpmf.compact STORE [--legacy] [--dry-run]` does it on demand
  (`--legacy` writes a plain image that pre-P17 code can read; older code refuses
  a file that still has segments with `Checksum mismatch`, never a silent stale
  read). Set with `"append_segments": true` in recall.config.json or
  `MPMF_APPEND_SEGMENTS=1`. Design: `src/mpmf/segments.py`.

## 0.3.0

Everything since 0.2.0. The package is now self-contained: the pieces that used to
live in host-side wrapper scripts are in `mpmf_plus`, and the host imports nothing
from outside it.

### Added
- **Self-contained package.** New modules: `rendering` (memory block), `thinking`
  (thinking-trace store), `eval` (scorecard, nightly, proposer; `python -m mpmf_plus.eval`),
  `health` (daily checks), `capture` (thinking capture), `cognition` (intake gate and
  `<prior-reasoning>` recall), `provider` and `hermes_adapter` (the Hermes memory-provider
  contract), `paths` (everything resolves from `MPMF_HOME`; no absolute host path in `src/`).
- **Per-turn ingest** (`mpmf_plus.ingest`): one turn in, facts out, one write transaction,
  one `store.log` line, bound to the host session id. A failed extraction is loud
  (`failures.log`), never silent.
- **History in order** (`history_mode`, default off): entity -> dated-fact index,
  relevance-then-date render (`## History of X`). See *Measured results* for the numbers.
- **float32 vectors**: one matrix, file format unchanged.
- **Fast dedup**: the near-duplicate scan is one matrix-vector product, candidate masks
  kept across store versions; embedding and dedup run outside the write transaction.
- **`swap_store`**: atomic store replacement with a lock held only for the rename.
- **One embed rule** (`mpmf.embed_text`): a single canonical embed-text function used by
  ingest, the thinking writer, the wrapper and the rebuild tool.
- **Dimension guard** everywhere: a store stamped with another embedder or dimension raises.
- **Strict store resolution**: no implicit store path; a missing store raises in `attach`
  mode; store resolution is lazy (nothing resolves at import time); only known stores are
  accepted by the daily check.
- **Deploy gate** (`deploy.sh`): host contract test + end-to-end store growth + the
  auto-revert pairing test + the real-turns-only test + the re-enrich accounting test +
  **7 cron dry-runs** (nightly, daily, watchdog, capture, intake, historical, re-enrich),
  including the intake gate counts. A candidate that fails any of them is not deployed.
- **Watchdog** (post-restart provider check): silent when healthy, alerts with rollback
  commands; the ingest rule is turn-based.
- **Disk guard**: a store save is refused when free space is below 2x the store size.
- **Auto-revert** of `history_mode` and of reranker-off when HOLDOUT drops >= 5.4 points
  with McNemar p < 0.10 against a recorded baseline; both verdicts are written to
  `autorevert_last.txt` every night.

### Fixed
- Nightly auto-revert paired 0 of 64 questions (integer ids vs string JSON keys) and
  reported SKIPPED (evidence not retained); ids are now compared as strings and fewer than 32 pairs exits non-zero
  with an alert.
- Silence checks counted cron prompts and subagent sessions as the user's turns; they now
  count real user sessions only, and the thinking-store age is measured against the user's
  last real turn instead of the wall clock.
- The re-enrich sweep line now accounts for every pending node
  (`enriched + failed + deferred == pending`).
- The historical-rumination job gated lessons through a module that no longer existed;
  it now uses `mpmf_plus.cognition`.
- Vectors: 26.0% of the previous production store's vectors (6,326 / 24,339) disagreed with
  a fresh embedding of their own text (cosine < 0.99). They were re-embedded and verified
  byte-exact; evidence in `CHANGES/evidence/r83e_vector_audit.json`.
- `evidence_sentence` parsing, ingest attribution, `store.log` default path, and several
  contract regressions found by the new gate.

### Changed
- `rerank_skip_after_ms: 0` is the deployed setting (reranker off on prefetch).
- Planned for a later release, **not** in 0.3.0: append-only record log (write cost still
  grows with store size, see *Write-time scaling*).

## 0.2.0
Embedder stamp in the store, shared-bank `create | attach`, write lock, no silent fallbacks.
