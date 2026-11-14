"""a1: append_segments is a PER-STORE setting (``append_segments_stores``), plus the loaded-code stamp.

Synthetic stores only. Throwaway files NAMED like the real ones (athena / espy / ayu / shared /
athena-thinking) in a temp dir; no real path is opened.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from mpmf import rollback as R
from mpmf.format import HEADER_SIZE, FileHeader
from mpmf import segments as SEG
from mpmf.memory import MemoryManager, _append_segments_for
from mpmf_plus import loaded_rev

REPO = Path(__file__).resolve().parent.parent
DIM = 1024       # the call sites in src/ hard-code 1024
NAMES = ["athena.mpmf", "espy.mpmf", "ayu.mpmf", "shared.mpmf", "athena-thinking.mpmf"]


def _vec(seed):
    return np.random.RandomState(seed).rand(DIM).astype("<f4").tobytes()


def _plain(path, n=3):
    """A plain legacy image with n nodes (explicit override: the config is not consulted)."""
    m = MemoryManager(str(path), vector_dimensions=DIM, append_segments=False)
    m.load()
    for i in range(n):
        with m.write_txn():
            m.create_node("n%d" % i, 1, 1, text="t%d" % i, summary="s%d" % i, vector=_vec(i))
    return m


def _is_segmented(path):
    with open(path, "rb") as fh:
        return SEG.read_base_end(FileHeader.unpack(fh.read(HEADER_SIZE))) is not None


def _nodes(path):
    m = MemoryManager(str(path), vector_dimensions=DIM, append_segments=False)
    m.load()
    return len(m._nodes)


@pytest.fixture
def env(tmp_path, monkeypatch):
    for k in ("MPMF_APPEND_SEGMENTS", "MPMF_APPEND_SEGMENTS_STORES"):
        monkeypatch.delenv(k, raising=False)
    cfg = tmp_path / "recall.config.json"
    monkeypatch.setenv("MPMF_RECALL_CONFIG", str(cfg))
    monkeypatch.setenv("MPMF_HOME", str(tmp_path))
    monkeypatch.setenv("MPMF_HEALTH_DIR", str(tmp_path / "health"))
    d = tmp_path / "stores"
    d.mkdir()

    def set_cfg(**kw):
        cfg.write_text(json.dumps(kw))
    return d, cfg, set_cfg


# ---------------------------------------------------------------- the flag rule

def test_default_off_everywhere(env):
    d, cfg, set_cfg = env
    set_cfg()
    for n in NAMES:
        assert _append_segments_for(d / n) is False
        assert MemoryManager(str(d / n), vector_dimensions=DIM)._segments_on is False


def test_old_global_key_no_longer_switches_any_store(env):
    d, cfg, set_cfg = env
    set_cfg(append_segments=True)
    for n in NAMES:
        assert MemoryManager(str(d / n), vector_dimensions=DIM)._segments_on is False
    set_cfg(append_segments=True, append_segments_stores=[])
    assert MemoryManager(str(d / "athena.mpmf"), vector_dimensions=DIM)._segments_on is False


def test_list_switches_only_the_listed_store(env):
    d, cfg, set_cfg = env
    set_cfg(append_segments_stores=[str(d / "athena.mpmf")])
    got = {n: MemoryManager(str(d / n), vector_dimensions=DIM)._segments_on for n in NAMES}
    assert got == {"athena.mpmf": True, "espy.mpmf": False, "ayu.mpmf": False,
                   "shared.mpmf": False, "athena-thinking.mpmf": False}


def test_same_basename_in_another_directory_is_not_listed(env):
    d, cfg, set_cfg = env
    other = d / "other"
    other.mkdir()
    set_cfg(append_segments_stores=[str(d / "athena.mpmf")])
    assert MemoryManager(str(other / "athena.mpmf"), vector_dimensions=DIM)._segments_on is False


def test_resolved_real_path_comparison(env):
    d, cfg, set_cfg = env
    (d / "real").mkdir()
    (d / "link").symlink_to(d / "real")
    set_cfg(append_segments_stores=[str(d / "link" / "athena.mpmf")])
    assert MemoryManager(str(d / "real" / "athena.mpmf"), vector_dimensions=DIM)._segments_on is True
    assert MemoryManager(str(d / "real" / ".." / "real" / "athena.mpmf"), vector_dimensions=DIM)._segments_on is True


def test_bad_values_mean_off(env):
    d, cfg, set_cfg = env
    for bad in (None, 5, True, {"a": 1}, [1, None, ""]):
        set_cfg(append_segments_stores=bad)
        assert _append_segments_for(d / "athena.mpmf") is False
    cfg.write_text("{not json")
    assert _append_segments_for(d / "athena.mpmf") is False


def test_env_list_and_env_global_override(env, monkeypatch):
    d, cfg, set_cfg = env
    set_cfg()
    monkeypatch.setenv("MPMF_APPEND_SEGMENTS_STORES", json.dumps([str(d / "athena.mpmf")]))
    assert _append_segments_for(d / "athena.mpmf") is True and _append_segments_for(d / "espy.mpmf") is False
    monkeypatch.delenv("MPMF_APPEND_SEGMENTS_STORES")
    monkeypatch.setenv("MPMF_APPEND_SEGMENTS", "1")          # explicit process-wide override (tools, tests)
    assert _append_segments_for(d / "espy.mpmf") is True
    monkeypatch.setenv("MPMF_APPEND_SEGMENTS", "0")
    set_cfg(append_segments_stores=[str(d / "athena.mpmf")])
    assert _append_segments_for(d / "athena.mpmf") is False


def test_constructor_override_still_wins(env):
    d, cfg, set_cfg = env
    set_cfg(append_segments_stores=[str(d / "athena.mpmf")])
    assert MemoryManager(str(d / "espy.mpmf"), vector_dimensions=DIM, append_segments=True)._segments_on is True
    assert MemoryManager(str(d / "athena.mpmf"), vector_dimensions=DIM, append_segments=False)._segments_on is False


def test_defaults_carry_the_new_key():
    from mpmf_plus import paths
    assert paths.DEFAULTS["append_segments_stores"] == [] and paths.DEFAULTS["append_segments"] is False


# ---------------------------------------------------------------- 5-store matrix

def test_five_store_matrix_memorymanager(env):
    """athena on, the other four off; after a real write the four stay plain (no segment bytes)."""
    d, cfg, set_cfg = env
    for n in NAMES:
        _plain(d / n)
    before = {n: ((d / n).read_bytes(), _nodes(d / n)) for n in NAMES}
    set_cfg(append_segments_stores=[str(d / "athena.mpmf")])
    kinds = {}
    for n in NAMES:
        m = MemoryManager(str(d / n), vector_dimensions=DIM)
        m.load()
        for k in range(2):      # athena: write 1 upgrades the plain file to a segment-capable base, write 2 appends
            with m.write_txn():
                m.create_node("new%d-%s" % (k, n), 1, 1, text="x", summary="y", vector=_vec(99 + k))
        kinds[n] = (m._segments_on, m.last_save_kind)
        assert _nodes(d / n) == before[n][1] + 2
    assert kinds["athena.mpmf"] == (True, "segment")
    for n in NAMES[1:]:
        assert kinds[n] == (False, "full"), n
        assert not _is_segmented(d / n), n
    assert _is_segmented(d / "athena.mpmf")
    # the four others: whole-image rewrite, header flags untouched
    for n in NAMES[1:]:
        h = FileHeader.unpack((d / n).read_bytes()[:HEADER_SIZE])
        assert h.flags == 0 and bytes(h._reserved) == bytes(80)


def test_five_store_matrix_mpmfmemory_wrapper(env):
    """The wrapper the provider uses (mpmf.MPMFMemory), driven through remember()+save()."""
    from mpmf import MPMFMemory
    d, cfg, set_cfg = env
    for n in NAMES:
        _plain(d / n)
    set_cfg(append_segments_stores=[str(d / "athena.mpmf")])
    flags = {}
    for n in NAMES:
        w = MPMFMemory(str(d / n), vector_dimensions=DIM, use_ollama=False)
        flags[n] = w.mem._segments_on
        for k in range(2):
            with w.mem.write_txn():
                w.remember("wrapper text %d for %s" % (k, n), name="w%d-%s" % (k, n), summary="s")
    assert flags == {"athena.mpmf": True, "espy.mpmf": False, "ayu.mpmf": False,
                     "shared.mpmf": False, "athena-thinking.mpmf": False}
    assert _is_segmented(d / "athena.mpmf")
    for n in NAMES[1:]:
        assert not _is_segmented(d / n), n


def test_entry_points_each_kind(env, monkeypatch):
    """Every kind of store-opening call site in src/: none of them segments a store that is not listed."""
    d, cfg, set_cfg = env
    from mpmf_plus import swap_store, store as pstore, thinking
    set_cfg(append_segments_stores=[str(d / "athena.mpmf")])
    # swap_store._open
    for n in NAMES:
        _plain(d / n)
        assert swap_store._open(str(d / n))._segments_on == (n == "athena.mpmf")
    # store.create_empty (explicit create of a NEW store file)
    fresh = d / "new" / "espy.mpmf"
    pstore.create_empty(pstore.StoreSettings(path=str(fresh), mode="create", origins={}), vector_dimensions=DIM)
    assert fresh.exists() and not _is_segmented(fresh)
    # thinking._open_rw / _open on the thinking store path
    monkeypatch.setattr(thinking, "STORE_PATH", str(d / "athena-thinking.mpmf"))
    monkeypatch.setattr(thinking, "VECTOR_DIM", DIM)
    assert thinking._open_rw()._segments_on is False
    monkeypatch.setattr(thinking, "STORE_PATH", str(d / "athena.mpmf"))
    assert thinking._open_rw()._segments_on is True       # same rule, the list is the only switch
    # provider._open_rw (MPMFMemory wrapper through configure(PATH=))
    from mpmf_plus import provider
    prev = provider.configure(PATH=str(d / "espy.mpmf"), VECTOR_DIM=DIM)
    try:
        assert provider._open_rw().mem._segments_on is False
        provider.configure(PATH=str(d / "athena.mpmf"))
        assert provider._open_rw().mem._segments_on is True
    finally:
        provider.configure(**prev)
    # compact / rollback tools never depend on the list (explicit False)
    from mpmf import compact
    assert compact.__doc__ and "append_segments=False" in Path(compact.__file__).read_text()
    for n in NAMES[1:]:
        assert not _is_segmented(d / n)


# ---------------------------------------------------------------- rollback per store

def test_rollback_removes_only_the_store_from_the_list(tmp_path):
    cfg = tmp_path / "c.json"
    a, b = tmp_path / "athena.mpmf", tmp_path / "other.mpmf"
    cfg.write_text(json.dumps({"append_segments_stores": [str(a), str(b)], "keep": 1}))
    res = R.set_flag_false(str(cfg), str(tmp_path / "bk"), "t", str(a))
    out = json.loads(cfg.read_text())
    assert out == {"append_segments_stores": [str(b)], "keep": 1, "append_segments": False}
    assert res["still_listed"] == [str(b)] and res["backup"] and os.path.exists(res["backup"])


def test_rollback_compacts_segmented_store_and_delists_it(tmp_path, monkeypatch):
    s = tmp_path / "athena.mpmf"
    m = MemoryManager(str(s), vector_dimensions=DIM, append_segments=True)
    m.load()
    for i in range(4):
        with m.write_txn():
            m.create_node("n%d" % i, 1, 1, text="t", summary="s", vector=_vec(i))
    assert _is_segmented(s)
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"append_segments_stores": [str(s)]}))
    monkeypatch.setenv("MPMF_HEALTH_DIR", str(tmp_path / "h"))
    assert R.main(["--config", str(cfg), "--store", str(s), "--tag", "t"]) == 0
    assert not _is_segmented(s) and _nodes(s) == 4
    assert json.loads(cfg.read_text())["append_segments_stores"] == []
    assert "append_segments off" in (tmp_path / "h" / "PAUSED.md").read_text()


# ---------------------------------------------------------------- old code ignores the new key

def _tag_exists(tag):
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", "--quiet", "refs/tags/%s^{commit}" % tag],
                          capture_output=True).returncode == 0


_PROBE = r"""
import sys, json
sys.path.insert(0, sys.argv[1])
from mpmf.memory import MemoryManager
m = MemoryManager(sys.argv[2], vector_dimensions=1024)
m.load()
print(json.dumps({"on": m._segments_on, "nodes": len(m._nodes)}))
"""


@pytest.mark.skipif(not _tag_exists("deploy-r95"), reason="deploy-r95 tag not in this repo")
def test_pre_a1_code_ignores_the_new_key_and_honours_the_old_one(tmp_path):
    wt = tmp_path / "wt"
    subprocess.run(["git", "-C", str(REPO), "worktree", "add", "--detach", "-q", str(wt), "deploy-r95"], check=True)
    try:
        store = tmp_path / "athena.mpmf"
        _plain(store)
        probe = tmp_path / "probe.py"
        probe.write_text(_PROBE)

        def run(cfgd):
            c = tmp_path / "c.json"
            c.write_text(json.dumps(cfgd))
            e = dict(os.environ, MPMF_RECALL_CONFIG=str(c), PYTHONPATH="")
            e.pop("MPMF_APPEND_SEGMENTS", None)
            r = subprocess.run([sys.executable, str(probe), str(wt / "src"), str(store)], capture_output=True,
                               text=True, env=e, timeout=120)
            assert r.returncode == 0, r.stderr
            return json.loads(r.stdout.strip().splitlines()[-1])
        assert run({"append_segments_stores": [str(store)]})["on"] is False        # old code: ignored
        assert run({"append_segments": True})["on"] is True                        # negative control
        assert not _is_segmented(store)
    finally:
        subprocess.run(["git", "-C", str(REPO), "worktree", "remove", "--force", str(wt)], capture_output=True)


# ---------------------------------------------------------------- loaded-code stamp

def test_stamp_written_with_the_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("MPMF_LOADED_REV_DIR", str(tmp_path / "lr"))
    p = loaded_rev.write_stamp("9.9 vX-1-gabc1234")
    assert p is not None
    rec = json.loads(Path(p).read_text())
    assert os.path.basename(p) == "%d.json" % os.getpid()
    assert rec["pid"] == os.getpid() and rec["code_rev"] == "9.9 vX-1-gabc1234" and rec["per_store_scope"] is True
    me = loaded_rev.proc_start(os.getpid())
    assert me is not None and abs(loaded_rev.parse_iso(rec["started"]) - me) < 2.0
    assert rec["ts"]
    assert not [f for f in os.listdir(tmp_path / "lr") if f.startswith(".stamp_")]


def test_stamp_removes_dead_pids_and_keeps_live_ones(tmp_path, monkeypatch):
    d = tmp_path / "lr"
    d.mkdir()
    monkeypatch.setenv("MPMF_LOADED_REV_DIR", str(d))
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        # a stamp of a LIVE process (the child), a dead pid, a pid reused by another process (wrong start)
        cst = loaded_rev.proc_start(child.pid)
        assert cst is not None
        live = {"pid": child.pid, "started": loaded_rev.iso(cst),
                "code_rev": "x", "per_store_scope": True, "ts": "t"}
        (d / ("%d.json" % child.pid)).write_text(json.dumps(live))
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        (d / ("%d.json" % dead.pid)).write_text(json.dumps(dict(live, pid=dead.pid)))
        reused = dict(live, pid=os.getppid(), started="2001-01-01T00:00:00Z")
        (d / ("%d.json" % os.getppid())).write_text(json.dumps(reused))
        (d / "garbage.json").write_text("{")
        loaded_rev.write_stamp("r")
        names = sorted(os.listdir(d))
        assert "%d.json" % os.getpid() in names and "%d.json" % child.pid in names
        assert "%d.json" % dead.pid not in names
        assert "%d.json" % os.getppid() not in names
        assert "garbage.json" in names            # not a pid file: left alone
    finally:
        child.kill()
        child.wait()


def test_readonly_health_dir_does_not_break_init(tmp_path, monkeypatch):
    ro = tmp_path / "ro"
    ro.mkdir()
    monkeypatch.setenv("MPMF_LOADED_REV_DIR", str(ro / "lr"))
    ro.chmod(0o500)
    try:
        if os.access(str(ro), os.W_OK):
            pytest.skip("running as a user that ignores directory modes")
        assert loaded_rev.write_stamp("r") is None
        from mpmf_plus import hermes_adapter
        prov = hermes_adapter.MPMFPlusProvider()
        monkeypatch.setattr(prov, "_warm_start", lambda: None)
        prov.initialize("ro-test")                      # must not raise
        assert prov._session_id == "ro-test"
    finally:
        ro.chmod(0o700)


def test_unwritable_dir_simulated_does_not_break_init(tmp_path, monkeypatch):
    """Root ignores directory modes, so also simulate EACCES on the write itself."""
    monkeypatch.setenv("MPMF_LOADED_REV_DIR", str(tmp_path / "lr"))

    def deny(*a, **k):
        raise PermissionError(13, "denied")
    monkeypatch.setattr(loaded_rev.tempfile, "mkstemp", deny)
    assert loaded_rev.write_stamp("r") is None
    monkeypatch.setattr(loaded_rev.os, "makedirs", deny)
    assert loaded_rev.write_stamp("r") is None
    from mpmf_plus import hermes_adapter
    prov = hermes_adapter.MPMFPlusProvider()
    monkeypatch.setattr(prov, "_warm_start", lambda: None)
    prov.initialize("deny-test")
    assert prov._session_id == "deny-test"


def test_stamp_path_is_a_file_does_not_raise(tmp_path, monkeypatch):
    f = tmp_path / "afile"
    f.write_text("x")
    monkeypatch.setenv("MPMF_LOADED_REV_DIR", str(f))
    assert loaded_rev.write_stamp("r") is None


def test_provider_and_adapter_write_the_stamp_on_init(tmp_path, monkeypatch):
    from mpmf_plus import hermes_adapter
    d = tmp_path / "lr"
    monkeypatch.setenv("MPMF_LOADED_REV_DIR", str(d))
    prov = hermes_adapter.MPMFPlusProvider()
    monkeypatch.setattr(prov, "_warm_start", lambda: None)
    prov.initialize("stamp-test")
    rec = json.loads((d / ("%d.json" % os.getpid())).read_text())
    from mpmf_plus._rev import CODE_REV
    assert rec["code_rev"] == CODE_REV and rec["per_store_scope"] is True


def test_stamp_skipped_under_pytest_without_the_env(tmp_path, monkeypatch):
    monkeypatch.delenv("MPMF_LOADED_REV_DIR", raising=False)
    monkeypatch.setenv("MPMF_HOME", str(tmp_path))
    assert loaded_rev.write_stamp("r") is None and not (tmp_path / "health").exists()
