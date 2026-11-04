"""r83f: swap_store must not lose writes that land during the swap.

Three cases:
  * plain swap — delta written after the snapshot is carried over;
  * mid-swap write — a writer that already held the old inode lands after the
    replace; the tool must notice (old-file mtime) and replay it;
  * rollback path is printed and the old file is intact.
"""
import hashlib
import os
import shutil
import sys
import threading
import time

import pytest

sys.path.insert(0, "/root/mpmf-release/src")
from mpmf.memory import MemoryManager  # noqa: E402
from mpmf_plus.swap_store import swap_store  # noqa: E402

DIM = 1024


def fake_embed(text):
    """Deterministic 1024-d vector derived from the text (no ollama)."""
    h = hashlib.sha256(text.encode()).digest()
    buf = (h * (DIM // len(h) + 1))[:DIM]          # exactly DIM bytes -> DIM floats
    return [b / 255.0 - 0.5 for b in buf]


def pack(v):
    import numpy as np
    return np.asarray(v, dtype="<f4").tobytes()


def make_store(path, n=6, tag="base"):
    for p in (path, path + ".lock"):
        if os.path.exists(p):
            os.remove(p)
    m = MemoryManager(path, vector_dimensions=DIM)
    with m.write_txn():
        for i in range(n):
            t = "%s fact %d about memory vectors" % (tag, i)
            m.create_node(name="n_%s_%d" % (tag, i), node_type=1, plane_flags=1,
                          text=t, summary=t[:40], wiki={"_ns": "memory"},
                          vector=pack(fake_embed(t)), importance=0.5)
    return path


def add_node(path, name, text):
    m = MemoryManager(path, vector_dimensions=DIM)
    m.load()                      # must load: write_txn saves the whole image
    with m.write_txn():
        m.create_node(name=name, node_type=1, plane_flags=1, text=text,
                      summary=text[:40], wiki={"_ns": "memory"},
                      vector=pack(fake_embed(text)), importance=0.5)


def test_swap_carries_delta_and_prints_rollback(tmp_path, monkeypatch):
    monkeypatch.setattr("mpmf_plus.swap_store.DEFAULT_PAUSED", str(tmp_path / "PAUSED.md"))
    (tmp_path / "PAUSED.md").write_text("| 2026-01-01T00:00Z | seed | x | y | z | w | CLOSED |\n")
    live = str(tmp_path / "live.mpmf")
    make_store(live, 6)
    new = str(tmp_path / "new.mpmf")
    shutil.copy2(live, new)

    add_node(live, "n_after_snapshot", "written after the snapshot was taken")
    add_node(live, "n_after_snapshot2", "second late node")

    res = swap_store(live, new, embed_fn=fake_embed, label="test", quiet_s=0.3, log=lambda *a: None)

    assert res["ok"] and res["delta"]["nodes_added"] == 2
    m = MemoryManager(live, vector_dimensions=DIM)
    m.load()
    names = {m.get_name(i) for i in m.all_node_ids()}
    assert {"n_after_snapshot", "n_after_snapshot2"} <= names
    assert len(m.all_node_ids()) == 8
    assert res["verify"]["sample_ok"] == res["verify"]["sample_n"]
    assert os.path.exists(res["backup"]) and "mv " in res["rollback"]


def test_write_landing_mid_swap_is_replayed(tmp_path, monkeypatch):
    monkeypatch.setattr("mpmf_plus.swap_store.DEFAULT_PAUSED", str(tmp_path / "PAUSED.md"))
    (tmp_path / "PAUSED.md").write_text("| 2026-01-01T00:00Z | seed | x | y | z | w | CLOSED |\n")
    live = str(tmp_path / "live.mpmf")
    make_store(live, 6)
    new = str(tmp_path / "new.mpmf")
    shutil.copy2(live, new)

    stop = threading.Event()

    def late_writer():
        # land ~40 ms into the swap: after the replace, on the stale inode
        time.sleep(0.04)
        try:
            add_node(live, "n_mid_swap", "this write landed during the swap window")
        except Exception:
            pass
        stop.set()

    t = threading.Thread(target=late_writer)
    t.start()
    res = swap_store(live, new, embed_fn=fake_embed, label="test", quiet_s=1.0, log=lambda *a: None)
    t.join(timeout=5)

    m = MemoryManager(live, vector_dimensions=DIM)
    m.load()
    names = {m.get_name(i) for i in m.all_node_ids()}
    assert "n_mid_swap" in names, "mid-swap write was lost (names=%s)" % sorted(names)
    assert res["verify"]["nodes"] >= 7
