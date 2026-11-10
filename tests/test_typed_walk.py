"""P12: typed-link walk — flag-off identity, 1-hop only, label filter, no eviction of seeds,
superseded / foreign-namespace / low-cosine neighbours never injected.

Throwaway store, no Ollama, no network.
"""
import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mpmf.memory import MemoryManager        # noqa: E402
from mpmf_plus import typed_walk as TW        # noqa: E402
from mpmf_plus import paths                   # noqa: E402

DIM = 1024


def vec(text):
    v = np.zeros(DIM, dtype=np.float32)
    for w in text.lower().split():
        v[int(hashlib.sha256(w.encode()).hexdigest(), 16) % DIM] += 1.0
    return (v / (float(np.linalg.norm(v)) or 1.0)).tolist()


def pack(v):
    import struct
    return struct.pack("<%df" % DIM, *v)


@pytest.fixture()
def world(tmp_path, monkeypatch):
    cfgp = tmp_path / "recall.config.json"
    cfgp.write_text('{"typed_walk": true, "typed_walk_k": 2, "typed_walk_seeds": 2, "typed_walk_min_cos": 0.2}')
    monkeypatch.setenv("MPMF_RECALL_CONFIG", str(cfgp))
    monkeypatch.setenv("MPMF_HOME", str(tmp_path))
    m = MemoryManager(str(tmp_path / "t.mpmf"), vector_dimensions=DIM)
    ids = {}
    texts = {
        "seed": "The user leases a green Tesla model three now",
        "old": "The user drove a Hyundai Accent for fifteen years before the Tesla",   # supersedes target
        "far": "Completely unrelated quantum chromodynamics lecture notes today",       # low cosine
        "gone": "The user leases a green Tesla and a superseded duplicate row",         # marked superseded
        "foreign": "The user leases a green Tesla in the ruminate namespace",
        "hop2": "The user drove a Hyundai Accent to Brooklyn every single weekend",     # 2 hops away
        "other": "Something about the user and a Tesla lease contract",
    }
    with m.write_txn():
        for k, t in texts.items():
            wiki = {"_ns": "ruminate" if k == "foreign" else "memory"}
            if k == "gone":
                wiki["_superseded_by"] = "1"
            n = m.create_node(name=k, node_type=1, plane_flags=1, text=t, summary=t, wiki=wiki, vector=pack(vec(t)))
            ids[k] = int(getattr(n, "node_id", n))
        link = lambda a, b, lab: m.create_edge(ids[a], ids[b], 2, lab)
        link("seed", "old", "supersedes")
        link("seed", "far", "DERIVED_FROM")
        link("seed", "gone", "IS_A")
        link("seed", "foreign", "IS_A")
        link("old", "hop2", "DERIVED_FROM")
        link("seed", "other", "coref:Tesla")          # label not in the walk set
    node_ns = {i: (m.get_wiki(i) or {}).get("_ns", "memory") for i in ids.values()}
    return m, ids, node_ns


def hits(ids, *names):
    return [{"rank": i + 1, "node_id": ids[n], "text": n, "score": 1.0} for i, n in enumerate(names)]


def test_defaults_off():
    assert paths.DEFAULTS["typed_walk"] is False
    assert paths.DEFAULTS["typed_walk_k"] == 2


def test_walk_adds_one_hop_neighbour_in_last_slot_same_length(world):
    m, ids, node_ns = world
    res = hits(ids, "seed", "other", "far")
    out = TW.expand(m, res, vec("green tesla lease hyundai accent"), ["memory"], node_ns)
    assert len(out) == len(res)
    assert out[0]["node_id"] == ids["seed"]                       # seed kept
    names = [o["node_id"] for o in out]
    assert ids["old"] in names and out[-1]["node_id"] == ids["old"]
    assert out[-1]["via"] == "supersedes" and out[-1]["via_seed"] == ids["seed"]
    assert [o["rank"] for o in out] == [1, 2, 3]


def test_not_injected_superseded_foreign_unlabelled_or_two_hops(world):
    m, ids, node_ns = world
    res = hits(ids, "seed", "other", "far")
    out = TW.expand(m, res, vec("green tesla lease hyundai accent weekend"), ["memory"], node_ns)
    got = {o["node_id"] for o in out}
    assert ids["gone"] not in got           # _superseded_by
    assert ids["foreign"] not in got        # namespace
    assert ids["hop2"] not in got           # 2 hops (seed -> old -> hop2), and `old` is not a seed
    assert ids["other"] in got              # was already a hit; coref edge is not walked


def test_min_cos_and_empty(world):
    m, ids, node_ns = world
    res = hits(ids, "seed", "other", "hop2")
    out = TW.expand(m, res, vec("zzz qqq"), ["memory"], node_ns)   # nothing similar enough
    assert [o["node_id"] for o in out] == [r["node_id"] for r in res]
    assert TW.expand(m, [], vec("a"), ["memory"], node_ns) == []


def test_seeds_never_evicted_when_list_is_short(world):
    m, ids, node_ns = world
    res = hits(ids, "seed", "other")                                # 2 hits, 2 seeds -> no room
    out = TW.expand(m, res, vec("green tesla lease hyundai accent"), ["memory"], node_ns)
    assert [o["node_id"] for o in out] == [ids["seed"], ids["other"]]


def test_flag_off_returns_input_untouched(world, tmp_path, monkeypatch):
    m, ids, node_ns = world
    (tmp_path / "recall.config.json").write_text('{"typed_walk": false}')
    res = hits(ids, "seed", "other", "far")
    assert TW.expand(m, res, vec("green tesla lease hyundai accent"), ["memory"], node_ns) is res


def test_failure_never_raises(world):
    m, ids, node_ns = world
    res = hits(ids, "seed", "other", "far")

    class Broken:
        _edges = [object()]          # no .edge_id -> AttributeError inside the walk
        _edge_labels = {}

    assert TW.expand(Broken(), res, vec("x"), ["memory"], node_ns) == res
    assert "error" in TW.LAST


def test_provider_flag_off_does_not_import_walk():
    import inspect
    from mpmf_plus import provider as P
    src = inspect.getsource(P._recall_d20)
    gate = src.index('.get("typed_walk", False)')
    assert gate < src.index("import typed_walk")
