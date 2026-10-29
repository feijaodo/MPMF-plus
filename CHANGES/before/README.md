# MPMF+

**MPMF+** is a 5-plane binary graph memory for LLM agents: a compact on-disk
format (nodes, edges, per-node raw text / summary / wiki / vector), a retrieval
layer, and a recall layer that turns a query into a rendered memory block.

This repository is **one installable package** containing both layers:

| Import | Contents |
|--------|----------|
| `mpmf` | core library: binary format, `MemoryManager`, `MPMFMemory`, four-stage `RetrievalPipeline`, embedders |
| `mpmf_plus` | the MPMF+ layer: config, embedder/reranker interfaces, recall arm C, write-lock + intake gate, gauntlet harness |

The HTTP server is **not part of this package**. **No background thread is
started unless the host opts in** (`background_thread` in the config). Every
path is configurable via environment variables or a config file; there is no
hardcoded installation path.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install .            # add [fast] for numpy, [rerank] for the cross-encoder
```

## Quick start

```python
from mpmf_plus import Store          # reads $MPMF_STORE_PATH or ~/.mpmf/store.mpmf
s = Store()
s.store("The nightly backup runs at 02:00 UTC", name="backup_time")
for hit in s.recall("when does the backup run?"):
    print(hit.name, hit.text)
```

See `examples/store_and_recall.py` for a full offline run.

## Configuration

Settings resolve in this order: explicit argument → environment variable →
config file (`$MPMF_CONFIG`, default `~/.mpmf/config.json`) → built-in default.
Copy `config.example.json` to get started. Key variables:

| Variable | Meaning | Default |
|----------|---------|---------|
| `MPMF_HOME` | home dir for stores/config | `~/.mpmf` |
| `MPMF_STORE_PATH` | store file | `$MPMF_HOME/store.mpmf` |
| `MPMF_ARM` | recall arm: `c` (`d20`) or `production` | `c` |
| `MPMF_RENDER` | block render: `full` or `summary` | `full` |
| `MPMF_EMBEDDER` | `auto` / `ollama` / `openai` / `hash` | `auto` |
| `MPMF_RERANKER` | `auto` / `cross-encoder` / `null` | `auto` |

### Recall arms

**Arm C (`d20`) is the DEFAULT.** For each query it takes the cosine top-**20**
over the allowed nodes, runs **one cross-encoder pass**, fuses the cosine order
and the rerank order by **reciprocal-rank fusion with k=20**, and returns the
top-k. When no cross-encoder is installed the reranker is a no-op and arm C
degrades cleanly to cosine ranking.

Rendering defaults to **full text**: each hit's raw text with a per-item cap of
**1200** chars inside a block capped at **3000** chars. `summary` mode uses the
node summary with 600-char items.

Embedders and rerankers sit behind two small interfaces
(`Embedder.embed/pack`, `Reranker.rerank`), so either can be swapped without
touching the retrieval code.

## Safety features

- **Write-lock.** Every mutation runs inside an exclusive cross-process
  transaction (a POSIX `flock` on `<store>.lock` plus a process-local mutex, with
  a reload-if-stale on entry). The store is a whole-file image, so without this a
  second writer's nodes would be silently destroyed.
- **Intake gate.** A fact whose text digest already exists is merged rather than
  duplicated, so parallel writes of the same fact converge to one node. Staged
  lessons are merged/promoted/rejected by cosine near-copy and a session-count
  rule, with every decision written to an append-only JSONL log.

## Measured results

These are the only performance claims this README makes, and all come from the
project's own isolation runs. Nothing here is extrapolated.

**Correct-answer rate (arm C vs. the production pipeline), r56 isolation.**
Judged on the same probe sets, arm C (d20 + full-text render) raised the
correct-answer rate over the production arm:

| Probe set | production | arm C |
|-----------|-----------:|------:|
| v1 (n=49) | 37% | **56%** |
| v2 (n=49) | 30% | **59%** |

**Write-lock stress test.** With many threads writing concurrently, the
cross-process transaction + intake gate produced **exactly one node** for a
repeated identical write, and no lost writes for distinct parallel writes.
Reproduce with `tests/test_mpmf_plus_store.py`.

## The graph: what it does and does not do yet

Honest status of the graph layer, from the project's own audits:

- **Tag-overlap edges are mostly shallow.** Of the edges built by the
  tag-overlap rule, **89.6%** were sampled; judges rated **36% meaningful**,
  **56% shallow**, and **8% wrong**. Most such edges connect nodes that share a
  tag but do not add retrieval value.
- **A 1-hop "Related" section showed no measured gain.** Adding a one-hop
  expansion section to the rendered block did not measurably improve answer
  quality in the runs performed.
- **A fair two-fact test is still pending.** A controlled test that requires
  combining two facts across a hop has not yet been run, so the multi-hop value
  of the graph is **not** established.

Treat graph traversal as a useful index, not as a proven reasoning aid.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

**Not chosen.** See `LICENSE` — it is a placeholder. Replace it before any
public release.
