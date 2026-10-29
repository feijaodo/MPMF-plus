# MPMF+

A 5-plane binary graph memory for LLM agents: an on-disk format (nodes, edges,
per-node raw text / summary / wiki / vector), a retrieval layer, and a recall
layer that renders a memory block for injection into a prompt. This package is a
port of the code that was actually measured, not a re-implementation.

**Why:** on a 3,402-memory deployment the shipped configuration answered 56% of
natural and 59% of paraphrased questions correctly, against 37% / 30% for the
summary-based pipeline it replaced (details in *Measured results*).

## Install

Python **3.10 or newer** (`requires-python = ">=3.10"`).

```bash
pip install .                 # core only (needs requests)
pip install ".[rerank]"       # + sentence-transformers, for the cross-encoder
pip install ".[fast]"         # + numpy, for the matrix path
```

From git (tagged, no PyPI publication yet):

```bash
pip install "git+https://github.com/feijaodo/mpmf-plus.git@v0.1.0"
```

### What gets installed

| path | what it is |
|---|---|
| `src/mpmf/` | the core format + retrieval (`format.py`, `memory.py`, `retrieval.py`, `query.py`, `embedding.py`, `openai_embed.py`, `ingestion/`) |
| `src/mpmf_plus/memory.py` | port of the live store/recall wrapper (`CODE_REV r55`); path/name changes only, no logic changes — `CHANGES/diff/01_memory_port.diff` |
| `src/mpmf_plus/block.py` | port of the live block builder (`render_cfg`, `wrapper_recall`, the render loop); mechanical extraction, no logic changes — `CHANGES/diff/02_block_and_adapter.md` |
| `src/mpmf_plus/hermes_adapter.py` | the only Hermes-coupled file; holds no memory logic |
| `src/mpmf_plus/errors.py` | the three failure types below |
| `CHANGES/` | port diff, before/after sources, manifest, `evidence/` (in the repo, not shipped in the package) |
| `examples/store_and_recall.py` | offline end-to-end example (temp store, no model) |

`memory.py` contains unused lesson-injection internals inherited from the author's agent; they are inactive and will be removed in 0.2.

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
weights (~88 MB) are fetched from the Hugging Face Hub the first time a rerank
happens, so that one call needs network access; afterwards it loads from
`~/.cache/huggingface`. Without the optional dependency the rerank stage is
skipped and recall degrades to cosine order (arm C behaviour), it does not
silently fail.

**Disk:** `sentence-transformers` is small (4.7 MB) but it pulls torch and
transformers — measured on the author's box: torch **1.2 GB** as installed
(a CPU-only wheel is smaller), transformers 56 MB. The core (no rerank extra)
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
| `MPMF_STORE_PATH` | `$MPMF_HOME/store.mpmf` | the store file |
| `MPMF_RECALL_CONFIG` | `$MPMF_HOME/recall.config.json` | recall/render config |
| `MPMF_CORE_PATH` | the bundled `mpmf` core | override only; by default the core shipped inside this package is used, and nothing outside it is imported |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | where the embedder lives |

**Nothing fails silently.** A missing or unreadable store raises
`StoreNotFoundError`, a missing/invalid config raises `ConfigError`, and a dead
or empty embedder raises `EmbedderError`. There is no fallback to another file
and no "return empty on error" (`src/mpmf_plus/errors.py`;
`tests/test_no_silent_fallbacks.py`).

## Claims and their artifacts

Every claim below is traceable to a file. Nothing else is claimed.

| # | claim | artifact |
|---|---|---|
| 1 | The port is mechanical: the shipped core and wrapper are the measured code, not a rewrite | `CHANGES/manifest.md`, `CHANGES/diff/01_memory_port.diff`, `CHANGES/diff/02_block_and_adapter.md`, `CHANGES/port_report.json` |
| 2 | Failures are loud: the three exception types are raised, not swallowed | `src/mpmf_plus/errors.py`, `src/mpmf_plus/memory.py` (`_open`), `tests/test_no_silent_fallbacks.py` |
| 3 | **The write-lock works at package level.** Two processes x 25 writes through `mpmf_plus.memory._write_txn_open` into one store copy produced **exactly +50 nodes, 0 missing** (6522 -> 6572), and a writer holding a stale image is refused with `StaleImageError` instead of silently overwriting another writer's nodes | `CHANGES/evidence/r59e_stress.log` (Round 59e, step 1) |
| 4 | Config is environment-driven: store/recall defaults are the historical ones, and the core path defaults to the core bundled in this package | `src/mpmf_plus/memory.py`, `src/mpmf_plus/block.py`, `config.example.json` |

Claim 3 covers only concurrent writers against one file, not multi-host
deployment.

## Measured results

On the author's deployment (3,402 memories), the shipped configuration (cosine
top-20 + cross-encoder + rank fusion, full-text rendering) answered **56%** of
natural questions and **59%** of paraphrased questions correctly, against **37%**
and **30%** for the previous summary-based pipeline (49 questions each, 3-model
judge panel, judge shown the true fact). The package was then shown to reproduce
that deployment exactly: identical top-8 results and rendered blocks on all 98
questions. The evaluation data is personal and not published, so these numbers
cannot be rerun by others.

Artifacts, kept in this repository but **not shipped in the package**:
`CHANGES/evidence/r56_stats.json` (Round 56 judge results, aggregates only) and
`CHANGES/evidence/r59d_parity.json` + `r59d_parity.log` (Round 59d, 98/98 identical;
redacted to per-probe flags and totals).

## Deliberately not claimed

- No claim that the graph improves answers.
- No latency claim: the port keeps the same code, but no end-to-end timing was
  measured in this package.
- No multi-process deployment claim beyond the write-lock test in claim 3.
- No quality claim about edges or the "Related" hop; the eval harnesses used in
  earlier rounds live outside this package and are not shipped, so their
  numbers are not restated here.

## Licence

Free for noncommercial use under the **PolyForm Noncommercial License 1.0.0**
(full text in `LICENSE`, SPDX identifier `PolyForm-Noncommercial-1.0.0`).

**Commercial licensing:** commercial use is not granted by that licence. For a
commercial licence, contact the author (feijaodo).

## Contributing

Contributions require a signed Contributor License Agreement granting the author
the right to relicense — see `CONTRIBUTING.md`. Unsigned contributions are not
merged.
