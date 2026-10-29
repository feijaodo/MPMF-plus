# CHANGES — Round 59b

Changed files in <repo> (md5 before → after). 'n/a' = the file was
rewritten or created in this round, so no pre-round hash exists in the release tree.

| file | md5 before | md5 after |
|---|---|---|
| `.gitignore` | n/a | `58aba31c0e77df8c3d58b31f167eefc8` |
| `README.md` | n/a | `f59c4a329f4c06db24c318231b3b408d` |
| `pyproject.toml` | n/a | `35dd57f711d5006a4033128a2a13c15b` |
| `src/mpmf/__init__.py` | 410080c2 | `410080c2987e9e467f39844894e0a852` |
| `src/mpmf/query.py` | c6ffc1d6 | `c6ffc1d6d480e81082b0d887dfc3909d` |
| `src/mpmf_plus/config.py` | bb7da410 | `708bc32f7aee5e8e242e6165c34063db` |
| `src/mpmf_plus/recall.py` | 1667ff3d | `2296b91d2da0bce359d0afda2e99ae14` |
| `src/mpmf_plus/store.py` | a3626b36 | `a3ef72a9f7e9d53b7f227084a0560ea2` |
| `tests/test_cas.py` | n/a | `bc7053bb1e34c31b848d69205137c88f` |
| `tests/test_graph_value.py` | n/a | `c1f93bdf8f19ce279adceb016bda245d` |
| `tests/test_integration.py` | n/a | `d02323ddaa37858c6b0dcade99fcba19` |
| `tests/test_mpmf_plus_store.py` | n/a | `9cdbacdb04887ae48f3bed0c41d94007` |

## Diffs

- `diffs/store-write-lock.diff` — **real bug fix**: `WriteLock` allocated a fresh
  `threading.Lock()` per instance, so the shared per-path mutex helper
  (`_process_write_lock`, already present but unused) never applied. 20 parallel
  writes to one store produced 9 nodes; the test `test_parallel_distinct_writes_are_not_lost`
  is the regression guard.
- `diffs/recall-unwrap.diff` — **packaging bug**: `RecallEngine` assumed a `MemoryManager`
  and broke when handed an `MPMFMemory`; now unwraps either.
- `diffs/config-path-wording.diff` — docstring no longer contains an absolute `<home>/` path.

## Rewritten tests

- `tests/test_cas.py` — the previous file called `MemoryManager.read_version_by_name`, an API
  that has never existed (the real CAS API is attached to `MPMFMemory` in `mpmf/query.py:224-225`).
  24 failures were test-side. Rewritten against the real API (6 tests).
- `tests/test_graph_value.py`, `tests/test_integration.py` — personal names and the
  `store.mpmf` filename replaced with neutral ones.

## Process note

md5 'before' values for files edited in this round were captured from the Round 59b step-1
scan, not from a snapshot taken immediately before editing; the hunks above are exact.

## Suite

`MPMF_EMBEDDER=hash MPMF_RERANKER=null pytest -q` → **49 passed, 1 xfailed** (xfail = known
core bug, see below).


## Known core bugs reported, not patched

- `mpmf/query.py:186` — CAS version is `str(int(time.time()*1000))`, so two CAS calls inside
  the same millisecond return the same version and a stale-version check can pass. Test
  `test_version_increments_on_update` is xfail(strict) until the core is fixed.
