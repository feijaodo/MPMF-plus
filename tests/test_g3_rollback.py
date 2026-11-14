"""g3: rollback = compact a segmented store to the legacy plain image, then flag off, then move the tree.

Synthetic stores only; no live paths. The "old reader" check is a byte-level re-implementation of what
pre-P17 code enforces (checksum over everything after the header to EOF), plus -- when the repo carries
the v0.2.0 tag -- the real old code from a temp worktree.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mpmf import rollback as R
from mpmf.format import HEADER_SIZE, FileHeader, Plane
from mpmf.memory import MemoryManager

REPO = Path(__file__).resolve().parent.parent
DIM = 16


def _vec(seed):
    return np.random.RandomState(seed).rand(DIM).astype("<f4").tobytes()


def _mk(path, on):
    m = MemoryManager(str(path), vector_dimensions=DIM, append_segments=on)
    m.load()
    return m


def _add(m, i):
    with m.write_txn():
        n = m.create_node("n%d" % i, 1, 1, text="text %d" % i, summary="s%d" % i,
                          wiki={"k": "v%d" % i}, vector=_vec(i))
        if i > 0:
            m.create_edge(n.node_id, max(1, n.node_id - 1), Plane.ASSOCIATIVE, "rel%d" % i)


def _segmented(path, n=12):
    m = _mk(path, True)
    for i in range(n):
        _add(m, i)
    assert m._seg_count > 0
    return m


def _old_reader_accepts(path) -> bool:
    """What pre-P17 MemoryManager.load() enforces: checksum over data[256:] to EOF."""
    data = Path(path).read_bytes()
    h = FileHeader.unpack(data)
    return hashlib.sha256(memoryview(data)[HEADER_SIZE:]).digest() == h.checksum


def _tag():
    tags = subprocess.run(["git", "-C", str(REPO), "tag"], capture_output=True, text=True).stdout.split()
    if not tags:
        pytest.skip("no tags in this clone")
    return "v0.2.0" if "v0.2.0" in tags else tags[0]


def test_old_reader_rejects_segmented_and_accepts_compacted(tmp_path):
    p = tmp_path / "s.mpmf"
    m = _segmented(p)
    nodes, edges = len(m._nodes), len(m._edges)
    assert R.detect(str(p))["segmented"] is True
    assert _old_reader_accepts(p) is False          # negative control: the proof can fail
    res = R.compact_to_legacy(str(p), DIM)
    assert res["segments_folded"] > 0 and res["nodes"] == nodes and res["edges"] == edges
    assert R.detect(str(p))["segmented"] is False
    assert _old_reader_accepts(p) is True
    r = _mk(p, False)
    assert len(r._nodes) == nodes and len(r._edges) == edges
    assert r._seg_count == 0 and r._seg_base_end is None


def test_compaction_keeps_name_index_and_content(tmp_path):
    p = tmp_path / "s.mpmf"
    m = _segmented(p)
    before = (dict(m._name_index), {i: (m.get_name(i), m.get_text(i), m.get_wiki(i), bytes(m.get_vector(i)))
                                     for i in m._nodes})
    R.compact_to_legacy(str(p), DIM)
    r = _mk(p, False)
    after = (dict(r._name_index), {i: (r.get_name(i), r.get_text(i), r.get_wiki(i), bytes(r.get_vector(i)))
                                    for i in r._nodes})
    assert before == after


def test_verification_failure_aborts_and_leaves_store_untouched(tmp_path, monkeypatch):
    p = tmp_path / "s.mpmf"
    _segmented(p)
    raw = p.read_bytes()
    real = R._signature
    calls = {"n": 0}

    def lying(m):
        s = real(m)
        calls["n"] += 1
        if calls["n"] == 2:               # the re-opened rewritten image "loses" a name
            s["names"] -= 1
        return s
    monkeypatch.setattr(R, "_signature", lying)
    with pytest.raises(R.RollbackAbort):
        R.compact_to_legacy(str(p), DIM)
    assert p.read_bytes() == raw
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".mpmf_rollback_")]


def test_plain_store_is_not_rewritten_by_main(tmp_path, monkeypatch):
    p = tmp_path / "s.mpmf"
    m = _mk(p, False)
    _add(m, 1)
    cfg = tmp_path / "recall.config.json"
    cfg.write_text(json.dumps({"keep": 1}))
    monkeypatch.setenv("MPMF_HEALTH_DIR", str(tmp_path / "h"))
    raw = p.read_bytes()
    assert R.main(["--config", str(cfg), "--store", str(p), "--tag", "t"]) == 0
    assert p.read_bytes() == raw
    assert json.loads(cfg.read_text()) == {"keep": 1, "append_segments": False}


def test_main_dry_run_changes_nothing(tmp_path, capsys):
    p = tmp_path / "s.mpmf"
    _segmented(p)
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"append_segments": True}))
    raw, craw = p.read_bytes(), cfg.read_text()
    assert R.main(["--config", str(cfg), "--store", str(p), "--dry-run"]) == 0
    assert p.read_bytes() == raw and cfg.read_text() == craw
    assert "SEGMENT-CAPABLE" in capsys.readouterr().out


def test_main_missing_store_refuses(tmp_path, capsys):
    cfg = tmp_path / "c.json"
    cfg.write_text("{}")
    assert R.main(["--config", str(cfg), "--store", str(tmp_path / "nope.mpmf")]) == 2
    assert "ROLLBACK ABORTED" in capsys.readouterr().out
    assert cfg.read_text() == "{}"


def test_main_bad_config_json_aborts_without_touching_it(tmp_path, monkeypatch):
    monkeypatch.setenv("MPMF_HEALTH_DIR", str(tmp_path / "h"))
    p = tmp_path / "s.mpmf"
    _add(_mk(p, False), 1)
    cfg = tmp_path / "c.json"
    cfg.write_text("{not json")
    assert R.main(["--config", str(cfg), "--store", str(p)]) == 2
    assert cfg.read_text() == "{not json"


@pytest.mark.skipif(subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", "--quiet", "refs/tags/v0.2.0"],
                                   capture_output=True).returncode != 0, reason="v0.2.0 tag not in this clone")
def test_real_old_tag_reads_compacted_store_and_refuses_segmented(tmp_path):
    wt = tmp_path / "wt"
    subprocess.run(["git", "-C", str(REPO), "worktree", "add", "--detach", "-q", str(wt), "v0.2.0"], check=True)
    try:
        seg, plain = tmp_path / "seg.mpmf", tmp_path / "plain.mpmf"
        _segmented(seg)
        shutil.copy(seg, plain)
        R.compact_to_legacy(str(plain), DIM)
        n = len(_mk(plain, False)._nodes)
        code = ("import sys; sys.path.insert(0, %r)\n"
                "from mpmf.memory import MemoryManager\n"
                "m = MemoryManager(sys.argv[1], vector_dimensions=%d); m.load()\n"
                "print('LOADED', len(m._nodes), len(m._edges))\n" % (str(wt / "src"), DIM))
        ok = subprocess.run([sys.executable, "-c", code, str(plain)], capture_output=True, text=True)
        assert ok.returncode == 0 and ok.stdout.startswith("LOADED %d " % n), ok.stderr[-300:]
        bad = subprocess.run([sys.executable, "-c", code, str(seg)], capture_output=True, text=True)
        assert bad.returncode != 0 and "Checksum mismatch" in bad.stderr
    finally:
        subprocess.run(["git", "-C", str(REPO), "worktree", "remove", "--force", str(wt)], capture_output=True)
        subprocess.run(["git", "-C", str(REPO), "worktree", "prune"], capture_output=True)


# ---- deploy.sh rollback: the real script, a throwaway live clone, a throwaway store/config ----

def _live_clone(tmp_path):
    live = tmp_path / "live"
    subprocess.run(["git", "clone", "-q", "--shared", str(REPO), str(live)], check=True, capture_output=True)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    subprocess.run(["git", "-C", str(live), "checkout", "-q", "--detach", head], check=True, capture_output=True)
    return live


def _head(live):
    return subprocess.run(["git", "-C", str(live), "rev-parse", "HEAD"], capture_output=True, text=True).stdout


def _run_deploy(tmp_path, live, *args, store, cfg):
    env = dict(os.environ, MPMF_LIVE=str(live), MPMF_HOME=str(tmp_path / "home"),
               MPMF_RECALL_CONFIG=str(cfg), MPMF_HEALTH_DIR=str(tmp_path / "health"),
               MPMF_ROLLBACK_BACKUP_DIR=str(tmp_path / "bk"), MPMF_STORE_PATH=str(store))
    return subprocess.run(["bash", str(REPO / "deploy.sh"), *args], capture_output=True, text=True,
                          env=env, timeout=300)


@pytest.fixture
def rig(tmp_path):
    store = tmp_path / "s.mpmf"
    _segmented(store)
    cfg = tmp_path / "recall.config.json"
    cfg.write_text(json.dumps({"append_segments": True, "keep": "me"}))
    return store, cfg, _live_clone(tmp_path)


def test_rollback_unknown_tag_refused_nothing_touched(tmp_path, rig):
    store, cfg, live = rig
    raw, craw, head = store.read_bytes(), cfg.read_text(), _head(live)
    r = _run_deploy(tmp_path, live, "rollback", "no-such-tag-xyz", store=store, cfg=cfg)
    assert r.returncode == 1 and "ROLLBACK REFUSED" in r.stdout
    assert store.read_bytes() == raw and cfg.read_text() == craw and _head(live) == head


def test_rollback_usage_and_option_like_tag_refused(tmp_path, rig):
    store, cfg, live = rig
    assert _run_deploy(tmp_path, live, "rollback", store=store, cfg=cfg).returncode == 2
    r = _run_deploy(tmp_path, live, "rollback", "--force", store=store, cfg=cfg)
    assert r.returncode == 1 and "REFUSED" in r.stdout
    assert _run_deploy(tmp_path, live, "rollback", _tag(), "--bogus", store=store, cfg=cfg).returncode == 2


def test_rollback_dry_run_prints_plan_and_changes_nothing(tmp_path, rig):
    store, cfg, live = rig
    raw, craw, head = store.read_bytes(), cfg.read_text(), _head(live)
    r = _run_deploy(tmp_path, live, "rollback", _tag(), "--dry-run", store=store, cfg=cfg)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SEGMENT-CAPABLE" in r.stdout and "DRY RUN: nothing changed" in r.stdout
    assert store.read_bytes() == raw and cfg.read_text() == craw and _head(live) == head


def test_rollback_aborts_before_moving_live_when_store_unusable(tmp_path, rig):
    store, cfg, live = rig
    store.write_bytes(store.read_bytes()[:100])         # shorter than a header
    head = _head(live)
    r = _run_deploy(tmp_path, live, "rollback", _tag(), store=store, cfg=cfg)
    assert r.returncode == 1 and "ROLLBACK ABORTED" in r.stdout
    assert json.loads(cfg.read_text())["append_segments"] is True
    assert _head(live) == head


def test_rollback_happy_path_compacts_then_flags_then_moves(tmp_path, rig):
    store, cfg, live = rig
    tag = _tag()
    n = len(_mk(store, True)._nodes)
    r = _run_deploy(tmp_path, live, "rollback", tag, store=store, cfg=cfg)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert out.index("compacted:") < out.index("config: append_segments false") < out.index("ROLLED BACK")
    assert "RESTART" in out
    assert R.detect(str(store))["segmented"] is False and _old_reader_accepts(store)
    assert len(_mk(store, False)._nodes) == n
    assert json.loads(cfg.read_text()) == {"append_segments": False, "keep": "me"}
    want = subprocess.run(["git", "-C", str(REPO), "rev-parse", tag + "^{commit}"],
                          capture_output=True, text=True).stdout
    assert _head(live) == want
    assert (tmp_path / "health" / "PAUSED.md").read_text().count("append_segments off") == 1
