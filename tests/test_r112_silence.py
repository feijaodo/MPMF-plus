"""r112: delete the silence — embedding fallbacks, gate ownership, zero-value config.

Six failure modes that all turn an error into a plausible-looking value. Each
test is written against the behaviour the code SHOULD have, so it fails on the
release before the fix (see the commit message for the before/after run).

  T1  empty text must be refused, not embedded as a zero vector
  T2  a context-length error must not be retried with HALF the text
  T3  a dead backend must raise, not fall back to a hash vector
  T4  a gate holder must not unlink a marker another holder now owns
  T5  a marker whose pid is dead must not block the embedder
  T6  the rebuild CLI's factory must accept the output path (it raised TypeError)
  T7  a config value of 0 / false must survive the read (no `or` default)
"""

import json
import os
import time

import pytest

from mpmf_plus.errors import EmbedderError


# --------------------------------------------------------------------------- T1
def test_empty_text_is_refused_by_the_local_embedder():
    """`[0.0] * 1024` is a valid-looking vector for a node with no content.

    It passes the dim check, sits in the matrix, and ranks against every query
    at 0.0 — a silent hole. Refuse the input instead.
    """
    import mpmf_plus.memory as M
    import mpmf_plus.provider as P

    for mod in (P, M):
        with pytest.raises(EmbedderError):
            mod._embed_floats_local("")
        with pytest.raises(EmbedderError):
            mod._embed_floats_local("   \n\t ")


# --------------------------------------------------------------------------- T2
def test_context_length_error_is_not_retried_with_half_the_text(monkeypatch):
    """Halving the source on a ctx-length error stores a vector for a document
    that does not exist. The text is already capped once; a second, silent cut
    is the bug. One call, then raise."""
    import mpmf_plus.provider as P

    seen = []

    def boom(text):
        seen.append(text)
        raise RuntimeError("input length exceeds context length")

    monkeypatch.setattr(P, "_embed_http_once", boom)

    with pytest.raises(EmbedderError):
        P._embed_floats_local("word " * 800)

    assert len(seen) == 1, "the embedder was called %d times: %r" % (
        len(seen), [len(s) for s in seen])


# --------------------------------------------------------------------------- T3
def test_a_dead_backend_raises_instead_of_returning_a_hash_vector(monkeypatch):
    """The hash embedder produced plausible 1024-d vectors during outages, so a
    store written while Ollama was down passed every structural check and ranked
    on noise. It must raise."""
    from mpmf.embedding import EmbeddingProvider

    e = EmbeddingProvider(dim=64)
    monkeypatch.setattr(e, "_try_ollama", lambda text: None)
    monkeypatch.setattr(e, "_try_openai_compat", lambda text: None)

    with pytest.raises(Exception) as ei:
        e.embed("the nightly backup runs at 02:00 UTC")
    assert "embed" in str(ei.value).lower()


# --------------------------------------------------------------------------- T4
def test_a_gate_holder_does_not_unlink_a_marker_another_holder_owns(tmp_path, monkeypatch):
    """Two holders share one file. If A's exit deletes the marker B just wrote,
    B's embed batch starts while A is still embedding."""
    monkeypatch.setenv("MPMF_HOME", str(tmp_path))
    from mpmf_plus import embed_gate as eg

    path = eg.marker_path("ingest")
    path.parent.mkdir(parents=True, exist_ok=True)

    with eg._holder("ingest"):
        # Another process takes the marker over while we hold it.
        path.write_text(json.dumps({"pid": os.getpid() + 1, "ts": time.time(),
                                    "token": "holder-B"}))

    assert path.exists(), "holder A deleted holder B's live marker"
    assert json.loads(path.read_text())["token"] == "holder-B"


# --------------------------------------------------------------------------- T5
def test_a_dead_pids_marker_does_not_block_the_embedder(tmp_path, monkeypatch):
    """A holder that was SIGKILLed leaves its marker behind; every batch job then
    waits out the full staleness window for a process that no longer exists."""
    monkeypatch.setenv("MPMF_HOME", str(tmp_path))
    from mpmf_plus import embed_gate as eg

    path = eg.marker_path("ingest")
    path.parent.mkdir(parents=True, exist_ok=True)

    dead = 4194303                       # above the default pid_max; never a live pid
    assert not os.path.exists("/proc/%d" % dead)
    path.write_text(json.dumps({"pid": dead, "ts": time.time()}))

    assert eg.ingest_waiting() is False, "a dead holder still blocks the gate"


# --------------------------------------------------------------------------- T6
def test_the_default_memory_factory_accepts_the_output_path(tmp_path):
    """rebuild/cli.py calls the factory as `mem_factory(out_path)`; the shipped
    `_open_rw()` took no arguments, so the CLI raised TypeError before it wrote
    anything and its zero-vector fallback was unreachable."""
    from mpmf.memory import MemoryManager           # noqa: F401  (import order)
    from mpmf_plus.memory import _open_rw

    mem = _open_rw(str(tmp_path / "rebuilt.mpmf"))
    mem.save()
    assert (tmp_path / "rebuilt.mpmf").exists()


# --------------------------------------------------------------------------- T7
def test_zero_and_false_config_values_survive_the_read(tmp_path, monkeypatch):
    """A switch set to false must mean false, and a key that is read must exist.

    Scope note: `rerank_skip_after_ms` / `prefetch_budget_ms` are deliberately NOT
    asserted here. In this code 0 means "unset" (`elapsed < 0` = never rerank) and
    the live config sets 0, so making 0 literal would disable the reranker — a
    product decision, tracked in BACKLOG B5, not changed in a silence round.
    """
    from mpmf_plus import block, paths

    cfg_file = tmp_path / "recall.config.json"
    cfg_file.write_text(json.dumps({"facts_dedupe": "false", "show_said_date": True}))
    # paths.home() is resolved once at import, so pin the reader's path directly
    # rather than relying on MPMF_HOME being seen late.
    monkeypatch.setattr(paths, "config_path", lambda: cfg_file)

    assert block.injection_opts()["facts_dedupe"] is False, '"false" read as True'
    assert block.render_cfg().get("show_said_date") is True, "show_said_date unreachable"
