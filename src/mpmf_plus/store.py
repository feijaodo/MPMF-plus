"""Store resolution and lifecycle — ``store_mode = create | attach``.

Several agents may share ONE store on disk. That is only safe if every agent
agrees on two things up front: *which file* the store is, and *whether this
process is allowed to bring a new one into existence*.

Two modes:

``create``
    This agent owns the store. If the file does not exist it is created (once,
    explicitly, with a line on stderr — never silently mid-recall).

``attach``
    This agent is joining a store that another agent already created. The file
    MUST already exist; if it does not, every entry point raises
    ``StoreNotFoundError``. Attach mode never creates, never "repairs" by
    starting empty, and never falls back to a different path. A typo in the path
    must be a loud error, not a brand-new empty memory that silently answers
    "nothing recalled".

Resolution order for the path (first hit wins):

1. explicit argument (``configure_store(path=...)``, ``--store``)
2. ``$MPMF_STORE``                      (preferred env name)
3. ``$MPMF_STORE_PATH``                 (legacy name, still honoured)
4. config key ``store_path`` in ``$MPMF_RECALL_CONFIG`` /
   ``$MPMF_HOME/recall.config.json``
5. ``$MPMF_HOME/store.mpmf``

Resolution order for the mode:

1. explicit argument
2. ``$MPMF_STORE_MODE``
3. config key ``store_mode``
4. ``create`` (back-compatible default)

Anything else — an unknown mode, an unreadable config file, a store that is a
directory instead of a file — raises. There is no silent fallback in here.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, NamedTuple, Optional

from .errors import ConfigError, StoreNotFoundError

STORE_MODES = ("create", "attach")
DEFAULT_MODE = "create"


class StoreSettings(NamedTuple):
    """The resolved store: what file, which mode, and where each came from."""

    path: str
    mode: str
    origins: Dict[str, str]

    def describe(self) -> str:
        return (f"store={self.path!r} (from {self.origins.get('path', '?')}), "
                f"mode={self.mode} (from {self.origins.get('mode', '?')})")


def home() -> Path:
    return Path(os.environ.get("MPMF_HOME") or "~/.mpmf").expanduser()


def config_path() -> Path:
    return Path(os.environ.get("MPMF_RECALL_CONFIG")
                or str(home() / "recall.config.json"))


def read_config() -> Dict:
    """The recall config as a dict. Missing file -> {}; unreadable -> ConfigError."""
    p = config_path()
    try:
        with open(p) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        raise ConfigError(f"recall config unreadable: {p}: {type(e).__name__}: {e}") from e
    if not isinstance(cfg, dict):
        raise ConfigError(f"recall config is not a JSON object: {p}")
    return cfg


def resolve_store(path: Optional[str] = None,
                  mode: Optional[str] = None) -> StoreSettings:
    """Resolve (path, mode) with the precedence documented in the module docstring."""
    origins: Dict[str, str] = {}

    if path is not None:
        if not str(path).strip():
            raise ConfigError(
                f"store path is empty (from argument). Pass a file path, or omit it "
                f"to fall back to $MPMF_STORE / config store_path.")
        store_path, origins["path"] = path, "argument"
    elif os.environ.get("MPMF_STORE"):
        store_path, origins["path"] = os.environ["MPMF_STORE"], "env MPMF_STORE"
    elif os.environ.get("MPMF_STORE_PATH"):
        store_path, origins["path"] = os.environ["MPMF_STORE_PATH"], "env MPMF_STORE_PATH"
    else:
        cfg = read_config()
        if "store_path" in cfg:
            if not str(cfg["store_path"]).strip():
                raise ConfigError(
                    f"store_path is empty in {config_path()}. Remove the key or set "
                    f"a real path.")
            store_path, origins["path"] = str(cfg["store_path"]), "config store_path"
        else:
            store_path, origins["path"] = str(home() / "store.mpmf"), "default $MPMF_HOME/store.mpmf"

    # An EXPLICIT blank mode is an error, not "unset": silently treating "" as
    # "create" is how an agent ends up with a second store next to the shared
    # one. Whitespace around a real value is tolerated ("create " -> create).
    if mode is not None:
        if not str(mode).strip():
            raise ConfigError("store_mode is empty (from argument); use 'create' or 'attach'")
        store_mode, origins["mode"] = mode, "argument"
    elif "MPMF_STORE_MODE" in os.environ:
        if not os.environ["MPMF_STORE_MODE"].strip():
            raise ConfigError("$MPMF_STORE_MODE is set but empty; use 'create' or 'attach'")
        store_mode, origins["mode"] = os.environ["MPMF_STORE_MODE"], "env MPMF_STORE_MODE"
    else:
        cfg = read_config()
        if "store_mode" in cfg:
            if not str(cfg["store_mode"]).strip():
                raise ConfigError(
                    f"store_mode is empty in {config_path()}; use 'create' or 'attach'")
            store_mode, origins["mode"] = str(cfg["store_mode"]), "config store_mode"
        else:
            store_mode, origins["mode"] = DEFAULT_MODE, "default"

    store_mode = str(store_mode).strip().lower()
    if store_mode not in STORE_MODES:
        raise ConfigError(
            f"store_mode must be one of {STORE_MODES}, got {store_mode!r} "
            f"(from {origins['mode']}). Fix $MPMF_STORE_MODE or the "
            f"'store_mode' key in {config_path()}.")

    store_path = str(Path(store_path).expanduser())
    return StoreSettings(path=store_path, mode=store_mode, origins=origins)


def exists(path: str) -> bool:
    return Path(path).is_file()


def attach_guard(settings: StoreSettings) -> None:
    """In attach mode the store MUST exist. Loud failure otherwise; never creates."""
    if settings.mode != "attach":
        return
    p = Path(settings.path)
    if p.is_file():
        return
    if p.exists():  # exists but is not a regular file (dir, socket, ...)
        raise StoreNotFoundError(
            f"attach: store path exists but is not a regular file: {settings.path!r} "
            f"({settings.describe()})")
    parent = p.parent
    hint = ""
    if not parent.is_dir():
        hint = f" Its directory {str(parent)!r} does not exist either — check the path."
    raise StoreNotFoundError(
        f"attach: store not found at {settings.path!r} ({settings.describe()}). "
        f"attach mode never creates a store: either fix the path, or run the "
        f"first agent with store_mode=create to initialise it.{hint}")


def create_empty(settings: StoreSettings, vector_dimensions: int = 1024,
                 embed_model: Optional[str] = None) -> str:
    """Explicitly create an empty store at ``settings.path``.

    Used by ``store_mode=create`` and by ``--create`` in the CLI/selftest. Prints
    one line to stderr so creation is never silent. Refuses to overwrite.
    """
    from mpmf.memory import MemoryManager  # local import: store.py must stay import-light

    p = Path(settings.path)
    if p.exists():
        raise FileExistsError(
            f"refusing to create: {settings.path!r} already exists "
            f"({settings.describe()}). Use store_mode=attach to join it.")
    p.parent.mkdir(parents=True, exist_ok=True)
    m = MemoryManager(str(p), vector_dimensions=vector_dimensions)
    m.load()               # path absent -> empty working set
    if embed_model:        # r61: a store we create records what built it
        stamp_embedder(m, embed_model, vector_dimensions)
    m.periodic_save()      # atomic write of the empty image
    print(f"[mpmf_plus.store] created new store: {settings.path} "
          f"(dim={vector_dimensions})", file=sys.stderr)
    return settings.path


# ── r61: embedder provenance ────────────────────────────────────────────────
# The v0.1.x binary header has no free field for a model name (agent_id is 16
# bytes of random id and there is no string slot), so the record lives in a
# reserved wiki NODE instead. That is additive: a v0.1.1 store simply lacks the
# node and is accepted, then stamped on its first write. Namespace "meta" keeps
# it out of recall(), whose default namespaces are ("memory",).
EMBEDDER_NODE = "_mpmf_embedder"
EMBEDDER_NS = "meta"


def read_embedder(mem) -> Optional[Dict[str, str]]:
    """The recorded embedder for this store, or None for a pre-r61 store."""
    try:
        nid = mem.get_node_id_by_name(EMBEDDER_NODE)
        if nid is None:
            return None
        wiki = mem.get_wiki(nid) or {}
    except Exception:
        return None
    if not wiki.get("embed_model"):
        return None
    return {"embed_model": str(wiki.get("embed_model")),
            "embed_dim": str(wiki.get("embed_dim", "")),
            "stamped_at": str(wiki.get("stamped_at", ""))}


def stamp_embedder(mem, embed_model: str, embed_dim: int) -> None:
    """Write the provenance node for ``embed_model``/``embed_dim`` (idempotent)."""
    import time as _time

    if read_embedder(mem) is not None:
        return
    if not getattr(mem, "_in_txn", False):
        # r69: create_node() now refuses in-memory-only edits, and _open_rw()
        # calls this before any transaction is open. Take our own.
        with mem.write_txn():
            stamp_embedder(mem, embed_model, embed_dim)
        return
    from mpmf.format import NodeType, Plane
    mem.create_node(
        name=EMBEDDER_NODE,
        node_type=int(NodeType.FACT),
        plane_flags=int(Plane.PRIORITY.value),
        text=(f"Reserved node: this store's vectors were produced by "
              f"{embed_model!r} ({embed_dim}d). Written by mpmf-plus >=0.2.0."),
        summary=f"embedder={embed_model} dim={embed_dim}",
        wiki={"_ns": EMBEDDER_NS,
              "embed_model": embed_model,
              "embed_dim": str(embed_dim),
              "stamped_at": str(int(_time.time()))},
        vector=None,
        importance=0.0,
    )


def assert_embedder(mem, embed_model: str, embed_dim: int) -> Optional[Dict[str, str]]:
    """Raise ConfigError if the store was built by a DIFFERENT embedder.

    Absent record -> accepted (pre-r61 store); the caller stamps it on first
    write. Present and matching -> fine. Present and different -> loud error
    naming both, because cosine over two different embedding spaces is garbage
    that no distance check downstream can detect.
    """
    rec = read_embedder(mem)
    if rec is None:
        return None
    got_model, got_dim = rec["embed_model"], rec["embed_dim"]
    if got_model != embed_model or str(got_dim) != str(embed_dim):
        raise ConfigError(
            f"embedder mismatch: this store was built with embed_model="
            f"{got_model!r} dim={got_dim}, but this process would write with "
            f"embed_model={embed_model!r} dim={embed_dim}. Mixing embedding "
            f"spaces silently corrupts recall. Use the recorded embedder, or "
            f"create a separate store for {embed_model!r}.")
    return rec


def ensure_usable(settings: StoreSettings,
                  vector_dimensions: int = 1024,
                  create: bool = True) -> str:
    """Make the store usable for a WRITE under the declared mode.

    * attach  -> must already exist (``attach_guard``), never creates.
    * create  -> created once if absent, with a stderr line.
    """
    attach_guard(settings)
    if exists(settings.path):
        return settings.path
    if settings.mode == "attach":  # pragma: no cover - attach_guard already raised
        raise StoreNotFoundError(f"attach: store not found: {settings.path!r}")
    if not create:
        raise StoreNotFoundError(
            f"store not found: {settings.path!r} ({settings.describe()}) "
            f"and creation was not allowed here")
    return create_empty(settings, vector_dimensions=vector_dimensions)
