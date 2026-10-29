"""Compare-and-swap tests against the REAL API.

The core exposes CAS on ``MPMFMemory`` (mpmf/query.py:224-225 attaches
``read_version`` / ``compare_and_swap``), not on ``MemoryManager``. The previous
version of this file called ``MemoryManager.read_version_by_name``, an API that
has never existed, so all 24 of its tests failed for a test-side reason.

These tests exercise: create-only semantics, stale-version rejection, version
increment on update, and exactly-one-winner under concurrent creations.
"""
import threading

import pytest

from mpmf.query import MPMFMemory  # defines the class and attaches read_version / compare_and_swap

DIM = 64


def _mem(tmp_path):
    return MPMFMemory(str(tmp_path / "m.mpmf"), vector_dimensions=DIM)


def test_read_version_missing_returns_none(tmp_path):
    assert _mem(tmp_path).read_version("nope") is None


def test_create_only_succeeds_once(tmp_path):
    m = _mem(tmp_path)
    ok, ver = m.compare_and_swap("fact1", None, text="first")
    assert ok and ver
    ok2, _ = m.compare_and_swap("fact1", None, text="second")
    assert not ok2


def test_stale_version_rejected(tmp_path):
    m = _mem(tmp_path)
    ok, ver = m.compare_and_swap("fact1", None, text="first")
    assert ok
    ok2, _ = m.compare_and_swap("fact1", "stale-version", text="second")
    assert not ok2


def test_version_increments_on_update(tmp_path):
    m = _mem(tmp_path)
    _, v1 = m.compare_and_swap("fact1", None, text="first")
    ok, v2 = m.compare_and_swap("fact1", v1, text="second")
    assert ok and v2 != v1
    assert m.read_version("fact1") == v2


def test_concurrent_creations_exactly_one_wins(tmp_path):
    m = _mem(tmp_path)
    wins, lock = [], threading.Lock()

    def worker(i):
        ok, _ = m.compare_and_swap("same-name", None, text=f"body {i}")
        with lock:
            wins.append(ok)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sum(wins) == 1


def test_cas_survives_save_reload(tmp_path):
    p = str(tmp_path / "m.mpmf")
    m = MPMFMemory(p, vector_dimensions=DIM)
    _, v1 = m.compare_and_swap("fact1", None, text="first")
    m.save()
    m2 = MPMFMemory(p, vector_dimensions=DIM)
    assert m2.read_version("fact1") == v1
    ok, _ = m2.compare_and_swap("fact1", v1, text="second")
    assert ok


def test_version_monotonic_within_same_millisecond(tmp_path):
    """The r59d fix: version = max(previous + 1, now_ms) must strictly increase
    even when many CAS calls land inside one millisecond."""
    m = _mem(tmp_path)
    ok, ver = m.compare_and_swap("f", None, text="v0")
    assert ok
    seen = [int(ver)]
    for i in range(40):
        ok, ver = m.compare_and_swap("f", ver, text=f"v{i+1}")
        assert ok, f"CAS failed at iteration {i}"
        seen.append(int(ver))
    assert seen == sorted(set(seen)), "versions must be strictly increasing"
    assert all(b > a for a, b in zip(seen, seen[1:]))
