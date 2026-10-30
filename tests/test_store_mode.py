"""store_mode = create | attach: resolution, precedence, and refusal to guess.

r60. Several agents may share one store, so the two things that must never
happen silently are (a) attaching to a path that does not exist and quietly
starting an empty memory, and (b) "creating" over a store that already holds
someone else's nodes. These tests pin both, plus the resolution order
argument > env > config.
"""
import json

import pytest

from mpmf_plus import memory, store
from mpmf_plus.errors import ConfigError, StoreNotFoundError


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Isolate from the developer's real ~/.mpmf and environment."""
    for k in ("MPMF_STORE", "MPMF_STORE_PATH", "MPMF_STORE_MODE", "MPMF_HOME"):
        monkeypatch.delenv(k, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("MPMF_HOME", str(home))
    monkeypatch.setenv("MPMF_RECALL_CONFIG", str(home / "recall.config.json"))
    return home


def _settings(path):
    return store.StoreSettings(path=str(path), mode="create", origins={})


def _write_cfg(home, **cfg):
    (home / "recall.config.json").write_text(json.dumps(cfg))


# -- attach never creates ----------------------------------------------------

def test_attach_missing_store_raises(tmp_path):
    s = store.resolve_store(path=str(tmp_path / "nope.mpmf"), mode="attach")
    with pytest.raises(StoreNotFoundError) as ei:
        store.attach_guard(s)
    assert "attach" in str(ei.value) and "nope.mpmf" in str(ei.value)


def test_attach_missing_store_creates_nothing(tmp_path):
    p = tmp_path / "nope.mpmf"
    with pytest.raises(StoreNotFoundError):
        store.ensure_usable(store.resolve_store(path=str(p), mode="attach"))
    assert not p.exists()
    assert not (tmp_path / "nope.mpmf.lock").exists()


def test_attach_on_directory_raises(tmp_path):
    d = tmp_path / "store_dir"
    d.mkdir()
    with pytest.raises(StoreNotFoundError) as ei:
        store.attach_guard(store.resolve_store(path=str(d), mode="attach"))
    assert "not a regular file" in str(ei.value)


def test_attach_existing_store_is_allowed(tmp_path):
    p = tmp_path / "real.mpmf"
    store.create_empty(store.resolve_store(path=str(p)))
    store.attach_guard(store.resolve_store(path=str(p), mode="attach"))  # no raise
    assert p.is_file()


def test_memory_write_in_attach_mode_raises_and_creates_nothing(tmp_path, monkeypatch):
    """Integration: the package write path honours attach mode (r60)."""
    p = tmp_path / "absent.mpmf"
    monkeypatch.setattr(memory, "_SETTINGS", store.StoreSettings(
        path=str(p), mode="attach", origins={"path": "test", "mode": "test"}))
    monkeypatch.setattr(memory, "PATH", str(p))
    monkeypatch.setattr(memory, "STORE_MODE", "attach")
    with pytest.raises(StoreNotFoundError):
        memory._open_rw()
    assert not p.exists()


# -- create never overwrites -------------------------------------------------

def test_create_refuses_to_overwrite(tmp_path):
    p = tmp_path / "existing.mpmf"
    store.create_empty(store.resolve_store(path=str(p)))
    before = (p.stat().st_size, p.stat().st_mtime_ns)
    with pytest.raises(FileExistsError) as ei:
        store.create_empty(store.resolve_store(path=str(p)))
    assert "already exists" in str(ei.value)
    assert (p.stat().st_size, p.stat().st_mtime_ns) == before


def test_create_initialises_missing_store(tmp_path, capsys):
    p = tmp_path / "fresh.mpmf"
    out = store.ensure_usable(store.resolve_store(path=str(p), mode="create"))
    assert out == str(p) and p.is_file()
    assert "created new store" in capsys.readouterr().err  # never silent


def test_create_empty_store_is_readable_by_core(tmp_path):
    from mpmf.memory import MemoryManager
    p = tmp_path / "empty.mpmf"
    store.create_empty(store.resolve_store(path=str(p)), vector_dimensions=64)
    m = MemoryManager(str(p), vector_dimensions=64)
    m.load()
    assert m.header is not None and m.all_node_ids() == []


# -- unknown mode ------------------------------------------------------------

@pytest.mark.parametrize("bad", ["shared", "", "ATTACHED", "creat"])
def test_unknown_mode_raises_config_error(tmp_path, bad):
    with pytest.raises(ConfigError) as ei:
        store.resolve_store(path=str(tmp_path / "s.mpmf"), mode=bad)
    assert "store_mode must be one of" in str(ei.value) or "empty" in str(ei.value)


def test_blank_mode_is_an_error_not_unset(monkeypatch, tmp_path):
    """'' must never be read as 'unset' and silently become create."""
    with pytest.raises(ConfigError):
        store.resolve_store(path=str(tmp_path / "s.mpmf"), mode="")
    monkeypatch.setenv("MPMF_STORE_MODE", "  ")
    with pytest.raises(ConfigError):
        store.resolve_store()


def test_whitespace_around_valid_mode_is_tolerated(tmp_path):
    s = store.resolve_store(path=str(tmp_path / "s.mpmf"), mode=" attach ")
    assert s.mode == "attach"


def test_blank_path_is_an_error(tmp_path):
    with pytest.raises(ConfigError):
        store.resolve_store(path="")
    with pytest.raises(ConfigError):
        store.resolve_store(path="   ")


def test_unknown_mode_from_env_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MPMF_STORE_MODE", "sideways")
    with pytest.raises(ConfigError):
        store.resolve_store()


def test_invalid_config_json_raises(tmp_path):
    (tmp_path / "home" / "recall.config.json").write_text("{not json")
    with pytest.raises(ConfigError):
        store.resolve_store()


# -- env honoured ------------------------------------------------------------

def test_env_store_and_mode_honoured(monkeypatch, tmp_path):
    monkeypatch.setenv("MPMF_STORE", str(tmp_path / "env.mpmf"))
    monkeypatch.setenv("MPMF_STORE_MODE", "attach")
    s = store.resolve_store()
    assert s.path == str(tmp_path / "env.mpmf")
    assert s.mode == "attach"
    assert s.origins == {"path": "env MPMF_STORE", "mode": "env MPMF_STORE_MODE"}


def test_legacy_env_store_path_still_honoured(monkeypatch, tmp_path):
    monkeypatch.setenv("MPMF_STORE_PATH", str(tmp_path / "legacy.mpmf"))
    assert store.resolve_store().path == str(tmp_path / "legacy.mpmf")


def test_default_path_is_under_home(monkeypatch, tmp_path):
    s = store.resolve_store()
    assert s.path == str(tmp_path / "home" / "store.mpmf")
    assert s.mode == "create"


# -- config keys honoured ----------------------------------------------------

def test_config_keys_honoured(tmp_path, monkeypatch):
    _write_cfg(tmp_path / "home", store_path=str(tmp_path / "cfg.mpmf"),
               store_mode="attach")
    s = store.resolve_store()
    assert s.path == str(tmp_path / "cfg.mpmf")
    assert s.mode == "attach"
    assert "config" in s.origins["path"] and "config" in s.origins["mode"]


# -- precedence: argument > env > config ------------------------------------

def test_precedence_argument_beats_env_and_config(tmp_path, monkeypatch):
    _write_cfg(tmp_path / "home", store_path=str(tmp_path / "cfg.mpmf"),
               store_mode="attach")
    monkeypatch.setenv("MPMF_STORE", str(tmp_path / "env.mpmf"))
    monkeypatch.setenv("MPMF_STORE_MODE", "attach")
    s = store.resolve_store(path=str(tmp_path / "arg.mpmf"), mode="create")
    assert s.path == str(tmp_path / "arg.mpmf") and s.mode == "create"
    assert s.origins == {"path": "argument", "mode": "argument"}


def test_precedence_env_beats_config(tmp_path, monkeypatch):
    _write_cfg(tmp_path / "home", store_path=str(tmp_path / "cfg.mpmf"),
               store_mode="attach")
    monkeypatch.setenv("MPMF_STORE", str(tmp_path / "env.mpmf"))
    monkeypatch.setenv("MPMF_STORE_MODE", "create")
    s = store.resolve_store()
    assert s.path == str(tmp_path / "env.mpmf") and s.mode == "create"


def test_configure_store_repoints_and_validates(tmp_path):
    p = tmp_path / "arg.mpmf"
    s = memory.configure_store(path=str(p), mode="create")
    assert s.path == str(p) and memory.PATH == str(p)
    assert memory.store_settings().mode == "create"
    with pytest.raises(ConfigError):
        memory.configure_store(path=str(p), mode="mirror")
