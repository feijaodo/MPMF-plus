# Round 75 — 0.2.0 contents

**Every script from rounds r69–r74 that this release replaces, one line each.**
The pipeline is now a library + one CLI (`mpmf-rebuild`), not a pile of round
scripts.

| round script (research tree) | replaced by |
|---|---|
| `r69/rebuild.py` | `mpmf_plus.rebuild.extraction` + `cli.py` |
| `r69/rebuild2.py` | `extraction.run_extraction` (retry + FAILED + second pass) |
| `r69/rebuild3.py` | `extraction.extract_conversation` (67e prompt, retry, `_msg_index`/`_evidence`/`_fact_date`) |
| `r69/dedup.py` | `mpmf_plus.memory` near-dup rule + `_drop_superseded` |
| `r69/restore69e.py` | covered by provenance fields (`_msg_index`, `_evidence`) |
| `r69/kinds69f.py`, `kinds69f2.py` | `stage2.classify_text` (anchored verb rules) |
| `r69/restate69f.py`, `restate69f2.py` | `stage2.current_state` + `stage2.build_states` |
| `r69/cats69f.py`, `orphans69f.py` | `stage2.build_states` (IS_A in the same txn) |
| `r69/fixcar69f.py` | superseded by ground-truth corrections + `_superseded_by` |
| `r70/prep70.py` | `extraction.persist_facts` (schema asserted on every node) |
| `r71/retag71.py` | `mpmf_plus.rebuild.retag` (context-aware mode re-tag) |
| `r72/merge72.py`, `r73/delta73.py` | merge is a documented procedure; swap safety now lives in `_open()` re-baseline |
| `r73/supersede74.py`, `r74/fix74*.py` | `mpmf_plus.memory._drop_superseded` + `_superseded_by` |
| `r72/diag72.py` (rerank timing) | `rerank_clamp_chars` + governor logs |
| `r74/watch73.py`, `r74/health74.py` | `CHANGES/evidence/r70-r73/` + the 24h health line script |

## Package changes in this release

1. **Superseded nodes never become recall candidates** — `_superseded_by` is
   honoured by the d20 matrix and by a post-filter on `recall()`. They stay on
   disk as an audit trail. Test: `tests/test_r74_guards.py` (4 cases).
2. **`rerank_clamp_chars`** — config key, default `0` (off); clamps only the
   cross-encoder input, never the injected text. Test: 2 cases.
3. **Unthrottled governor trip logging** — `_prewarm_log()` writes every trip
   with rss / baseline / margin / store path / node count, bypassing the 60 s
   throttle that silently swallowed the post-cutover trip. Test: 2 cases.
4. **Re-baseline on store swap** — `_store_key` carries the inode and
   `_is_store_swap()` uses a 1.25× size ratio, so a rebuild renamed into place
   takes a fresh RSS baseline instead of inheriting one from the old store. A
   per-turn save never re-baselines. Test: 2 cases.
5. **`src/mpmf_plus/rebuild/`** — extraction, stage 2 and re-tag as a library,
   plus the `mpmf-rebuild` CLI (`--input --out --cap --workers --model`).
   Tests: `tests/test_rebuild_pipeline.py` (9 cases) on a 3-conversation fixture.

## Versioning

`0.2.0` was already committed at `2a258bc` and never pushed. This round **adds a
commit on top** rather than amending it: the earlier commit is the published
description of the 0.2.0 *port*, and rewriting it would lose that history for
anyone who already fetched. The version stays `0.2.0` — the round-75 work is
what makes 0.2.0 shippable (rebuild tool + the four guards), not a new feature
line.
