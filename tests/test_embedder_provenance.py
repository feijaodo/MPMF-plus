"""r61: a store records which embedder built it, and refuses to be mixed.

Three cases, as specified:
  * no record (v0.1.1 store)  -> accepted, stamped on the first write
  * matching record           -> accepted
  * different model or dim    -> ConfigError naming both

Offline: the embedder is stubbed, the store is real.
"""
import pytest

from mpmf_plus import memory, store
from mpmf_plus.errors import ConfigError

DIM = memory.VECTOR_DIM


@pytest.fixture(autouse=True)
def _stub_embedder(monkeypatch, tmp_path):
    monkeypatch.setenv("MPMF_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(memory, "_embed_floats_local",
                        lambda t, dim=DIM, cap=None: [0.01] * DIM)
    yield


def _make_store(path, embed_model=None, dim=DIM):
    settings = store.StoreSettings(path=str(path), mode="create", origins={})
    store.create_empty(settings, vector_dimensions=dim, embed_model=embed_model)
    return str(path)


def test_absent_record_is_accepted_and_stamped_on_first_write(tmp_path, monkeypatch):
    p = _make_store(tmp_path / "old.mpmf")          # no embed_model -> no record
    assert store.read_embedder(_load(p)) is None
    memory.configure_store(path=p, mode="attach")
    memory.store("the furnace was inspected on tuesday by the harbour crew",
                 name="prov_1", auto_extract=False, auto_link=False,
                 namespace="memory")
    rec = store.read_embedder(_load(p))
    assert rec is not None
    assert rec["embed_model"] == memory.EMBED_MODEL
    assert rec["embed_dim"] == str(DIM)


def test_stamp_is_written_once_not_per_write(tmp_path):
    p = _make_store(tmp_path / "once.mpmf")
    memory.configure_store(path=p, mode="attach")
    for i in range(3):
        memory.store(f"fact number {i} about the cobalt turbine assembly line",
                     name=f"prov_once_{i}", auto_extract=False, auto_link=False,
                     namespace="memory")
    _mm = _load(p); names = [_mm.get_name(i) for i in _mm.all_node_ids() if _mm.get_name(i) == store.EMBEDDER_NODE]
    assert names == [store.EMBEDDER_NODE]
    rec = store.read_embedder(_load(p))
    assert rec["embed_model"] == memory.EMBED_MODEL


def test_matching_record_is_accepted(tmp_path):
    p = _make_store(tmp_path / "match.mpmf", embed_model=memory.EMBED_MODEL)
    memory.configure_store(path=p, mode="attach")
    memory.store("a fact that should be accepted by the provenance check",
                 name="prov_match", auto_extract=False, auto_link=False,
                 namespace="memory")
    assert store.read_embedder(_load(p))["embed_model"] == memory.EMBED_MODEL


def test_different_model_raises_config_error_naming_both(tmp_path):
    p = _make_store(tmp_path / "other.mpmf", embed_model="nomic-embed-text")
    memory.configure_store(path=p, mode="attach")
    with pytest.raises(ConfigError) as ei:
        memory.store("a fact that must be refused across embedding spaces",
                     name="prov_bad", auto_extract=False, auto_link=False,
                     namespace="memory")
    msg = str(ei.value)
    assert "nomic-embed-text" in msg and memory.EMBED_MODEL in msg
    assert "embed_model=" in msg


def test_different_dim_raises_config_error_naming_both(tmp_path):
    p = _make_store(tmp_path / "dim768.mpmf", embed_model=memory.EMBED_MODEL, dim=768)
    memory.configure_store(path=p, mode="attach")
    with pytest.raises(ConfigError) as ei:
        memory.store("a fact that must be refused for a dimension mismatch",
                     name="prov_dim", auto_extract=False, auto_link=False,
                     namespace="memory")
    msg = str(ei.value)
    assert "dim=768" in msg and f"dim={DIM}" in msg


def test_reads_also_refuse_mismatch(tmp_path):
    p = _make_store(tmp_path / "read.mpmf", embed_model="nomic-embed-text")
    memory.configure_store(path=p, mode="attach")
    memory._invalidate_open_cache()
    with pytest.raises(ConfigError):
        memory.stats()


def test_create_mode_stamps_at_creation(tmp_path):
    """A store we create knows its embedder before the first user write."""
    p = _make_store(tmp_path / "new.mpmf", embed_model=memory.EMBED_MODEL)
    rec = store.read_embedder(_load(p))
    assert rec and rec["embed_dim"] == str(DIM)


def _load(path):
    from mpmf.memory import MemoryManager
    m = MemoryManager(str(path), vector_dimensions=DIM)
    m.load()
    return m
