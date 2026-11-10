# MPMF+

A 5-plane binary graph memory for LLM agents: an on-disk format (nodes, edges,
per-node raw text / summary / wiki / vector), a retrieval layer, and a recall
layer that renders a memory block for injection into a prompt. This package is a
port of the code that was actually measured, not a re-implementation.

**Why:** on the author's deployment, 64 "in what order…" questions scored 0.8% without the
history block and 68.0% with it (dev set, not held out); on a second frozen question set
(held-out, same method as dev set) 11.7% vs 69.5%; details and caveats in *Measured results*.

> **Installing with an agent?** Follow [`AGENT_INSTALL.md`](AGENT_INSTALL.md)
> step by step — it covers host detection, `create` vs `attach`, the two hooks,
> and the mandatory self-test.

## Measured results

Earlier evaluations (rounds 56 and 70–73: summary-pipeline comparison, rebuilt-store
head-to-head, render and rerank ablations) are kept as aggregate-only artifacts in
`CHANGES/evidence/`. Their figures are **not restated here**: this release's figures are
checked against its own derived result rows, and those older numbers are not.

### 0.3.0 results

**Order questions ("in what order…").** 64 questions, the same question text and ids in both arms,
`history_mode` off vs on:

**Dev set, not held out:** the 68.0% (and every number in this table) is on a development
set. The history feature was built and tuned against these 64 questions.

| dev set, not held out | off | on |
|---|---|---|
| score (correct + ½ partly) | **0.8%** | **68.0%** |
| strict (fully correct only) | **0 / 64 (0%)** | **32 / 64 (50%)** |
| correct / partly / declined / wrong | 0 / 1 / 59 / 4 | 32 / 23 / 5 / 4 |
| sequence check (answer's event order judged right) | **3 / 64** | **24 / 64** |

**Held-out result: RETIRED.** The held-out set failed its harness check (both arms read the
live `history_mode`, so the "off" arm was not off) and its store coverage was 3/10 (dev control
9/10). No held-out number is reportable, and none is quoted here: the held-out result was retired because of the harness bug and the low store coverage.

**Held-out v2 (a second, frozen question set; the first held-out attempt above stays retired).**
64 questions (md5 `2e3da59e7bbe9cca7a1d74f4daf126e8`) whose entities are disjoint from the dev set,
generated and filtered before any scoring, same store snapshot for both arms, same scoring code as the
dev set. Each arm's config was read back and checked before scoring (`ARM_CONFIG` lines in the evidence);
the store md5 was identical before and after. Scored once per arm.

| held-out, same method as dev set | off | on |
|---|---|---|
| score (correct + ½ partly) | **11.7%** | **69.5%** |
| strict (fully correct only) | **5 / 64 (7.8%)**, Wilson 95% CI 3.4–17.0% | **30 / 64 (46.9%)**, Wilson 95% CI 35.2–58.9% |
| correct / partly / declined / wrong | 5 / 5 / 53 / 1 | 30 / 29 / 1 / 4 |
| declined | 53 / 64 | 1 / 64 |
| sequence check (answer's event order judged right) | **0 / 64** | **33 / 64** |

Strict exact McNemar (held-out): b=26 (off not-correct, on correct), c=1, p = 4.2e-07. The held-out
strict gain (+25 questions, 7.8% to 46.9%) is in the same range as the dev-set gain (0% to 50%); the
intervals overlap. Remaining caveats: the on arm still leaves 29 of 64 answers only "partly" correct. The on-arm run was interrupted once by an operator-side tool timeout
after 8 questions were scored; it was resumed with the harness's row log, so those 8 rows were kept and
not re-scored (the other 56 were scored in the resumed run, same config read-back).

Strict McNemar (dev set) b=32, c=0, p < 1e-6. Caveats, all of which matter:

- The questions are drawn from topics with **2 to 5 distinct key dates** (2 dates: 4 questions;
  3: 30; 4: 23; 5: 7); this measures ordering where history exists, not a random question.
- n = 64, one author's data, one run per arm. This is a development set (the history feature was built and tuned against it), not a held-out set; the first held-out attempt is retired (see above) and the v2 held-out result is reported above. The 95% Wilson score interval on the strict 32/64 is **38.1–61.9%** (on the 0/64 off arm: 0–5.7%); computed by `CHANGES/evidence/r92_stats.py`.
- The 0.8% "off" figure is mostly *declines* (59 of 64): without the history block the
  system says it does not know. It is not mostly wrong answers.
- A nightly guard watches a separate HOLDOUT and reverts `history_mode` (and the
  reranker setting) on a >= 5.4-point drop with McNemar p < 0.10 (`REVERT_DROP` in
  `eval/nightly.py`). The separate nightly drift *alert* fires at >= 8 points, p < 0.10. No
  HOLDOUT result is quoted here.

Aggregates: `CHANGES/evidence/r92_order_history.json` (no question or answer text).

**Latency (prefetch), real sessions only.** On the author's deployment, the first 25 prefetches
made in real sessions, 2026-11-06 to 2026-11-10 (n = 25 calls from 3 sessions; benchmark,
scorecard, held-out harness, watchdog-probe and inspect calls excluded), took
**p50 1.81 s, p90 4.31 s**, max 5.02 s. That is the number to plan around: it includes the box
doing other work (cron jobs, the embedder shared with ingest). Caveats: 15 of the 25 calls are
from a single day (2026-11-06), the sample is one deployment, and percentiles are nearest-rank.
The log has grown since this window was cut; later calls are not included.

> Footnote (benchmark, not real sessions) — idle-host evaluation harness on the 64 dev-set order
> questions: prefetch p90 was 1,265 ms (history on) and 1,533 ms (off). This is a lower bound for an
> unloaded host and is not the latency to expect.

**The previous store had 26% mismatched vectors.** Before round 83e, 6,326 of 24,339 nodes
(26.0%) had a stored vector that disagreed with a fresh embedding of the node's own text
(cosine < 0.99; 4,350 below 0.90; mean 0.954). All were re-embedded and verified byte-exact.
Evidence:
`CHANGES/evidence/r83e_vector_audit.json`.

The evaluation data is personal and not published; aggregates only, with no questions,
answers, facts or conversation text, are in `CHANGES/evidence/`.

Artifacts, kept in this repository but **not shipped in the package**:
`CHANGES/evidence/r56_stats.json` (Round 56 judge results, aggregates only) and
`CHANGES/evidence/r59d_parity.json` + `r59d_parity.log` (Round 59d parity;
redacted to per-probe flags and totals).

## Install

Python **3.10 or newer** (`requires-python = ">=3.10"`).

```bash
pip install .                 # core only (needs requests)
pip install ".[rerank]"       # + sentence-transformers, for the cross-encoder
pip install ".[fast]"         # + numpy, for the matrix path
```

From git (tagged, no PyPI publication yet):

```bash
pip install "git+https://github.com/feijaodo/MPMF-plus.git@v0.3.0"
```

### What gets installed

| path | what it is |
|---|---|
| `src/mpmf/` | the core format + retrieval (`format.py`, `memory.py`, `retrieval.py`, `query.py`, `embedding.py`, `openai_embed.py`, `ingestion/`) |
| `src/mpmf_plus/memory.py` | port of the live store/recall wrapper (`CODE_REV r55`); path/name changes only, no logic changes — `CHANGES/diff/01_memory_port.diff` |
| `src/mpmf_plus/block.py` | port of the live block builder (`render_cfg`, `wrapper_recall`, the render loop); mechanical extraction, no logic changes — `CHANGES/diff/02_block_and_adapter.md` |
| `src/mpmf_plus/store.py` | store resolution and `store_mode = create \| attach` (shared memory bank) |
| `src/mpmf_plus/selftest.py` | `python -m mpmf_plus.selftest --store PATH` — the install gate |
| `src/mpmf_plus/hermes_adapter.py` | the only Hermes-coupled file; holds no memory logic |
| `src/mpmf_plus/errors.py` | the three failure types below |
| `CHANGES/` | port diff, before/after sources, manifest, `evidence/` (in the repo, not shipped in the package) |
| `examples/store_and_recall.py` | offline end-to-end example (temp store, no model) |

`memory.py` contains unused lesson-injection internals inherited from the author's agent; they are inactive and are scheduled for removal.

## Requirements

**Embedder — Ollama, `mxbai-embed-large` (1024-d), required for real recall.**
The client posts to `http://127.0.0.1:11434` and uses the canonical model name
with no padding. Point it at another host or port with the `OLLAMA_URL`
environment variable (there is no config-file key: it is env-only, read at
import time). The model must already be pulled:

```bash
ollama pull mxbai-embed-large
```

**Reranker — `cross-encoder/ms-marco-MiniLM-L-6-v2`, downloaded on first use.**
It runs on CPU (`max_length=512`) through `sentence-transformers`, and the
weights are fetched from the Hugging Face Hub the first time a rerank
happens, so that one call needs network access; afterwards it loads from
`~/.cache/huggingface`. Without the optional dependency the rerank stage is
skipped and recall degrades to cosine order (arm C behaviour), it does not
silently fail.

**Disk:** `sentence-transformers` itself is small but it pulls torch and
transformers, which are large (a CPU-only torch wheel is smaller). The core (no rerank extra)
needs only `requests`.

**Not required:** numpy (optional, `[fast]`), a GPU (everything is CPU), any API
key. The example and the test suite run with no model and no network.

## Quickstart

The whole offline flow — write three facts to a throwaway store, reopen it, recall
and render the block. Runs as written, no Ollama, no network:

```python
import json, os, tempfile
d = tempfile.mkdtemp(); os.environ["MPMF_HOME"] = d
os.environ["MPMF_STORE_PATH"] = os.path.join(d, "demo.mpmf")
json.dump({"memory_render": "full"}, open(os.path.join(d, "recall.config.json"), "w"))
from mpmf.memory import MemoryManager
from mpmf_plus import block
store = MemoryManager(os.environ["MPMF_STORE_PATH"], vector_dimensions=64); store.load()
store.create_node(name="backup_time", node_type=0, plane_flags=1, text="The nightly backup runs at 02:00 UTC.",
                  summary="backup at 02:00 UTC", wiki={"_ns": "memory"}, importance=0.5)
store.periodic_save()
store = MemoryManager(os.environ["MPMF_STORE_PATH"], vector_dimensions=64); store.load()
nid = [i for i in store.all_node_ids() if store.get_name(i) == "backup_time"][0]
shim = block.RecallShim(); shim.top_nodes = [nid]
shim.raw_texts = shim.summaries = {nid: "The nightly backup runs at 02:00 UTC."}
print(block.render_memory_block("when does the backup run?", shim, store, top_k=1)[0])
```

The recall here is a hand-built shim on purpose: real recall is
`mpmf_plus.memory.recall()`, which needs the Ollama embedder above. See
`examples/store_and_recall.py` for the full commented version.

## Configuration

Paths come from the environment or from the package's own location; none is
hardcoded to a filesystem path:

| variable | default | meaning |
|---|---|---|
| `MPMF_HOME` | `~/.mpmf` | base directory for store, cache, state, health |
| `MPMF_STORE` | `$MPMF_HOME/store.mpmf` | the store file (preferred name) |
| `MPMF_STORE_PATH` | `$MPMF_HOME/store.mpmf` | legacy alias for `MPMF_STORE`; `MPMF_STORE` wins |
| `MPMF_STORE_MODE` | `create` | `create` or `attach` — see *Shared memory bank* |
| `MPMF_STORE_LOCK_TIMEOUT` | `60` | seconds to wait for the cross-process write lock before `LockTimeoutError` |
| `MPMF_RECALL_CONFIG` | `$MPMF_HOME/recall.config.json` | recall/render config |
| `MPMF_CORE_PATH` | the bundled `mpmf` core | override only; by default the core shipped inside this package is used, and nothing outside it is imported |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | where the embedder lives |

The same two store settings also exist as config keys in
`$MPMF_HOME/recall.config.json`: `"store_path"` and `"store_mode"`.

**Nothing fails silently.** A missing or unreadable store raises
`StoreNotFoundError`, a missing/invalid config raises `ConfigError`, and a dead
or empty embedder raises `EmbedderError`. There is no fallback to another file
and no "return empty on error" (`src/mpmf_plus/errors.py`;
`tests/test_no_silent_fallbacks.py`).

### Shared memory bank

Several agents — Hermes, OpenClaw, a cron script — can use one store. Two
settings decide which file it is and whether this process may create it:

```bash
export MPMF_STORE=/srv/shared/bank.mpmf      # the one file everyone opens
export MPMF_STORE_MODE=attach                # join an existing bank
# first agent only: MPMF_STORE_MODE=create
```

| `store_mode` | behaviour when the file is missing |
|---|---|
| `create` | creates it once, with a line on stderr; refuses to overwrite an existing store |
| `attach` | raises `StoreNotFoundError` and creates **nothing** — no empty store, no new file, no fallback path |

Precedence: argument (`--store`, `configure_store()`) → `MPMF_STORE` /
`MPMF_STORE_PATH` / `MPMF_STORE_MODE` → config keys `store_path` / `store_mode`
→ the defaults above. An unknown or blank mode raises `ConfigError`; a blank
path raises `ConfigError` rather than silently becoming the default.

Writers take an exclusive `flock` on `<store>.lock` and re-read the image if
another process saved in the meantime, so two agents writing at once do not
erase each other's nodes (stress logs in `CHANGES/evidence/`). A single
`store()` holds that lock across an embed round-trip, hence the 60 s default
wait; a genuinely stuck lock still raises `LockTimeoutError`.

Same machine only — `flock` over a network filesystem is not safe. All agents
must use the same embedder. Since 0.2.0 the store records the model and
dimension it was built with (a reserved wiki node, `_mpmf_embedder`, namespace
`meta`, so it never appears in recall): a store with **no** record — every
v0.1.x store — is accepted and stamped on its first write, and a store recorded
with a different model or dimension raises `ConfigError` naming both. Nothing is
padded, converted or silently re-embedded.

### Write-time scaling

`store()` rewrites the whole image, so the cost of one write grows with the
store rather than with what changed. Two costs dominate: the O(N) dedup scan over every
node, and serialising + fsyncing + renaming the whole file. The Ollama embedding
round-trip is paid outside the write lock. No timing table is given here (the earlier
one is not backed by this release's evidence). Planned, **not in 0.3.0**: an append-only
record log with an in-memory index — a write appends one record instead of re-writing the
image, and full compaction moves to a background task. Until then, keep a shared bank
in the low tens of thousands of nodes, or accept multi-second writes.

## Claims and their artifacts

Every claim below is traceable to a file. Nothing else is claimed.

| # | claim | artifact |
|---|---|---|
| 1 | The port is mechanical: the shipped core and wrapper are the measured code, not a rewrite | `CHANGES/manifest.md`, `CHANGES/diff/01_memory_port.diff`, `CHANGES/diff/02_block_and_adapter.md`, `CHANGES/port_report.json` |
| 2 | Failures are loud: the three exception types are raised, not swallowed | `src/mpmf_plus/errors.py`, `src/mpmf_plus/memory.py` (`_open`), `tests/test_no_silent_fallbacks.py` |
| 3 | **The write-lock works at package level.** Concurrent writers through `mpmf_plus.memory._write_txn_open` into one store copy lost no nodes (see the artifact), and a writer holding a stale image is refused with `StaleImageError` instead of silently overwriting another writer's nodes | `CHANGES/evidence/r59e_stress.log` (Round 59e, step 1) |
| 4 | Config is environment-driven: store/recall defaults are the historical ones, and the core path defaults to the core bundled in this package | `src/mpmf_plus/memory.py`, `src/mpmf_plus/block.py`, `config.example.json` |

Claim 3 covers only concurrent writers against one file, not multi-host
deployment.

## Deliberately not claimed

- No claim that the graph improves answers.
- No latency guarantee: real-session prefetch p90 was 4.31 s over 25 calls (see *0.3.0 results*); the
  idle-host bench is roughly a third of that.
- No multi-process deployment claim beyond the write-lock test in claim 3.
- No quality claim about edges or the "Related" hop; the eval harnesses used in
  earlier rounds live outside this package and are not shipped, so their
  numbers are not restated here.

## Contributing

Contributions require a signed Contributor License Agreement granting the author
the right to relicense — see `CONTRIBUTING.md`. Unsigned contributions are not
merged.

## Licence

PolyForm Noncommercial License 1.0.0 — free for noncommercial use; commercial use
requires a separate licence. Full text in [`LICENSE`](LICENSE).

## Install, deploy and roll back

### Install from the wheel (fresh environment)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install mpmf_plus-0.3.0-py3-none-any.whl          # or: pip install .
export MPMF_HOME=/path/to/state                       # store, config, logs, eval live here
python -m mpmf_plus.selftest --store "$MPMF_HOME/store.mpmf" --mode create
```

`selftest` writes a marker, reads it back, recalls it and renders it; exit 0 = pass.
Write `{"memory_render": "full"}` to `$MPMF_HOME/recall.config.json` first (the renderer refuses to
guess a missing config; `AGENT_INSTALL.md` §3). Real recall needs the Ollama embedder (see
*Requirements*); the selftest takes it from `OLLAMA_URL`. Nothing is read from or written to any directory outside `MPMF_HOME` and
the Python environment.

Useful entry points: `python -m mpmf_plus.ingest` (one turn -> facts),
`python -m mpmf_plus.eval {scorecard,nightly,proposer}`, `mpmf-rebuild`.

### Deploy (host-side, with the gate)

The deployed tree is a git worktree pinned to a tag; the running agents import from it and
never execute the development checkout. Develop only in the release repository.

```bash
./deploy.sh --check-only          # run the gate on the working tree, move nothing
./deploy.sh deploy-<n> [<commit>] # gate the commit in a detached worktree, tag it, switch the live worktree
```

The gate runs, in order: host-contract test, auto-revert pairing/revert test, real-turns-only
test, re-enrich accounting test, end-to-end store growth (`sync_turn` must grow the store),
then the 7 cron dry-runs. Any failure prints `DEPLOY REFUSED` and the live tree is left
untouched; deploy also refuses while the live worktree has local edits. `deploy.sh` is
host-specific: set `MPMF_LIVE` (deploy-target worktree) and `MPMF_PY` (interpreter).

### Roll back

```bash
git -C "$MPMF_LIVE" checkout --detach <previous deploy tag>   # two commands: this one
# then restart the host process that imports the package (e.g. the gateway /restart)
```

Back up a store before any change that touches it, with a copy kept outside the live
directory; a store swap (`mpmf_plus.swap_store`) keeps the old file.
