# Changelog

## Unreleased

### Added
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
