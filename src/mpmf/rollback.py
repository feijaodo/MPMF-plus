"""g3: the store half of ``./deploy.sh rollback <tag>``.

    python -m mpmf.rollback --config recall.config.json [--store S] [--dry-run | --check]

Code from before P17 cannot read a segmented store (it refuses with ``Checksum
mismatch``). So before the live tree is moved to an older tag this tool

  1. detects whether the store is segment-capable (header flag),
  2. if so takes the store's write lock, loads it, serialises a PLAIN legacy image to a
     temp file next to it, re-opens that temp file and compares node/edge counts, the
     name index and the full node/edge/vector signature with what was loaded,
     and only then renames it over the store (atomic). Any difference aborts and leaves
     the store untouched,
  3. removes THIS store from ``append_segments_stores`` in the config (a1: per-store flag; every
     other key and list entry kept, temp file + rename, valid JSON re-read afterwards, timestamped
     copy first; the legacy global ``append_segments`` is written false explicitly too, because the
     older tag we roll back to honours it) and appends a PAUSED row.

``--dry-run`` prints the plan and writes nothing. ``--check`` exits 0 when the store is a
plain image, 3 when it is segment-capable (a still-running writer with the flag on will
re-segment the store on its next save; restart it).
Exit codes: 0 ok, 2 aborted (nothing moved), 3 --check found a segmented store.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from . import segments as _seg
from .format import HEADER_SIZE, FileHeader
from .memory import MemoryManager


class RollbackAbort(Exception):
    pass


def resolve_store(config_path: str, store: str | None = None) -> str:
    if store:
        return store
    if os.environ.get("MPMF_STORE_PATH"):
        return os.environ["MPMF_STORE_PATH"]
    try:
        with open(config_path) as fh:
            sp = (json.load(fh) or {}).get("store_path")
        if sp:
            return str(sp)
    except Exception:
        pass
    home = os.environ.get("MPMF_HOME") or os.path.join(os.path.expanduser("~"), ".mpmf")
    return os.path.join(home, "athena.mpmf")


def detect(store: str) -> dict:
    """Header-only look at the store: no lock, no write."""
    size = os.path.getsize(store)
    with open(store, "rb") as fh:
        head = fh.read(HEADER_SIZE)
    if len(head) < HEADER_SIZE:
        raise RollbackAbort("store %s is too short to be an image (%d bytes)" % (store, len(head)))
    h = FileHeader.unpack(head)
    base_end = _seg.read_base_end(h)
    return {"path": store, "bytes": size, "segmented": base_end is not None,
            "base_end": base_end, "dim": int(h.vector_dimensions or 0),
            "bytes_after_base": (size - base_end) if base_end is not None else 0}


def _signature(m: MemoryManager) -> dict:
    ids = sorted(m._nodes)
    return {
        "nodes": len(m._nodes), "edges": len(m._edges), "names": len(m._name_index),
        "name_index": dict(m._name_index),
        "node_sigs": [m._node_sig(i) for i in ids],
        "edge_sigs": sorted((e.edge_id, m._edge_sig(e)) for e in m._edges),
        "vectors": {i: bytes(m.get_vector(i)) for i in ids if m.has_vector(i)},
        "ids": ids,
    }


def _diff(a: dict, b: dict) -> list:
    bad = []
    for k in ("nodes", "edges", "names"):
        if a[k] != b[k]:
            bad.append("%s %d != %d" % (k, a[k], b[k]))
    for k in ("name_index", "ids", "node_sigs", "edge_sigs", "vectors"):
        if a[k] != b[k]:
            bad.append("%s differ" % k)
    return bad


def compact_to_legacy(store: str, dim: int, lock_timeout: float = 120.0) -> dict:
    """Fold segments into a plain image, verified before the rename. Raises RollbackAbort."""
    info = detect(store)
    m = MemoryManager(store, vector_dimensions=dim or info["dim"] or 1024, append_segments=False)
    fd = m._acquire_lock(lock_timeout)            # LockTimeoutError aborts the rollback
    tmp = None
    try:
        m.load()
        m._require_writable()  # prefix-vs-prefix parity cannot detect a lost suffix
        before = _signature(m)
        folded = m._seg_count
        m._segments_on = False
        data = m._serialize()
        dir_path = os.path.dirname(os.path.abspath(store))
        free = shutil.disk_usage(dir_path).free
        if free < len(data) * 2:
            raise RollbackAbort("not enough disk: need %.0f MB, have %.0f MB free"
                                % (len(data) * 2 / 1e6, free / 1e6))
        tfd, tmp = tempfile.mkstemp(dir=dir_path, prefix=".mpmf_rollback_")
        with os.fdopen(tfd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        chk = MemoryManager(tmp, vector_dimensions=m.vector_dimensions, append_segments=False)
        chk.load()
        after = _signature(chk)
        problems = _diff(before, after)
        if chk._seg_base_end is not None or chk._seg_count != 0:
            problems.append("rewritten image is still segment-capable")
        if problems:
            raise RollbackAbort("verification FAILED, store untouched: " + "; ".join(problems))
        os.rename(tmp, store)
        tmp = None
        return {"segments_folded": folded, "bytes_before": info["bytes"], "bytes_after": len(data),
                "nodes": before["nodes"], "edges": before["edges"], "names": before["names"],
                "verified": "counts, name index, node/edge/vector signature identical"}
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)                        # our own half-written temp file only
        m._release_lock(fd)


def _norm(p) -> str:
    return os.path.realpath(os.path.expanduser(str(p)))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def set_flag_false(config_path: str, backup_dir: str | None, tag: str, store: str | None = None) -> dict:
    """a1: take ``store`` OUT of ``append_segments_stores`` (every other key and list entry kept).

    The OLD global ``append_segments`` is ALWAYS written ``false`` explicitly as well (g3 behaviour,
    kept): the new code ignores that key, but the older tag we are rolling back to honours it.
    """
    p = Path(config_path)
    try:
        cfg = json.loads(p.read_text()) if p.exists() else {}
    except Exception as e:
        raise RollbackAbort("config %s is not valid JSON, left alone: %s" % (p, e))
    if not isinstance(cfg, dict):
        raise RollbackAbort("config %s is not a JSON object, left alone" % p)
    old_list = cfg.get("append_segments_stores", "(absent)")
    old_global = cfg.get("append_segments", "(absent)")
    backup = None
    if p.exists() and backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        backup = os.path.join(backup_dir, "%s.pre_rollback_%s" % (
            p.name, time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())))
        shutil.copy2(str(p), backup)
    want = _norm(store) if store else None
    lst = cfg.get("append_segments_stores")
    if isinstance(lst, list):
        cfg["append_segments_stores"] = [x for x in lst if not (want and isinstance(x, str) and _norm(x) == want)]
    cfg["append_segments"] = False        # always explicit: new code ignores it, the older target tag honours it
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".cfg_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        json.loads(Path(tmp).read_text())
        os.replace(tmp, str(p))
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    back = json.loads(p.read_text())
    if want and any(isinstance(x, str) and _norm(x) == want for x in (back.get("append_segments_stores") or [])):
        raise RollbackAbort("config re-read still lists %s in append_segments_stores" % store)
    if back.get("append_segments") is not False:
        raise RollbackAbort("config re-read does not show append_segments false")
    left = [x for x in (back.get("append_segments_stores") or []) if isinstance(x, str)]
    try:
        health = Path(os.environ.get("MPMF_HEALTH_DIR") or (p.parent / "health"))
        health.mkdir(parents=True, exist_ok=True)
        with open(health / "PAUSED.md", "a") as fh:
            fh.write("| %s | append_segments off (rollback to %s) | %s | store %s removed from "
                     "append_segments_stores (was %s; legacy key was %s); store compacted to plain "
                     "image first; backup %s | deploy.sh rollback |\n"
                     % (_now(), tag, p.name, store, old_list, old_global, backup))
    except Exception:
        pass
    return {"was": old_list, "was_global": old_global, "backup": backup, "still_listed": left}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mpmf.rollback")
    ap.add_argument("--config", required=True)
    ap.add_argument("--store")
    ap.add_argument("--dim", type=int, default=0)
    ap.add_argument("--tag", default="?")
    ap.add_argument("--backup-dir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    try:
        store = resolve_store(a.config, a.store)
        if not os.path.isfile(store):
            raise RollbackAbort("store %s does not exist; refusing to roll back blind" % store)
        info = detect(store)
        print("store: %s  %d bytes  %s%s" % (store, info["bytes"],
              "SEGMENT-CAPABLE (base_end=%d, %d bytes after base)" % (info["base_end"], info["bytes_after_base"])
              if info["segmented"] else "plain legacy image", ""))
        if a.check:
            return 3 if info["segmented"] else 0
        if a.dry_run:
            print("plan: %s" % ("compact to plain legacy image under the store lock (verify counts + name index + "
                                "signature before the rename)" if info["segmented"]
                                else "store is already plain: no compaction"))
            print("plan: remove %s from append_segments_stores in %s (legacy append_segments set false if it "
                  "was true; other keys kept)" % (store, a.config))
            return 0
        if info["segmented"]:
            res = compact_to_legacy(store, a.dim or info["dim"])
            print("compacted: %s" % json.dumps(res))
        else:
            print("store already plain: no compaction needed")
        res = set_flag_false(a.config, a.backup_dir, a.tag, store)
        print("config: append_segments false for %s: removed from append_segments_stores (was %s), "
              "backup %s" % (store, res["was"], res["backup"]))
        if res["still_listed"]:
            print("NOTE: other stores are still listed in append_segments_stores: %s -- they are NOT "
                  "compacted by this command" % res["still_listed"])
        return 0
    except RollbackAbort as e:
        print("ROLLBACK ABORTED: %s" % e)
        return 2
    except Exception as e:      # lock timeout, unreadable store, ...: nothing was moved
        print("ROLLBACK ABORTED: %s: %s" % (type(e).__name__, e))
        return 2


if __name__ == "__main__":
    sys.exit(main())
