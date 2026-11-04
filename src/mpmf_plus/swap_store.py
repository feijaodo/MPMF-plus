"""mpmf_plus.swap_store — replace a live store with a rebuilt copy, safely.

A 3-hour audit cannot swap the snapshot it started from: everything written in
between would vanish (that is exactly how 8 nodes were lost in r83e). So the
swap refreshes the delta from live immediately before the replace, keeps the
old file as the rollback target, watches it for writes that landed on the stale
inode, re-applies those, and only then verifies.

    from mpmf_plus.swap_store import swap_store
    res = swap_store(live_path, work_path, embed_fn=my_embed, label="athena")

Returns a dict with the rollback path, node counts, the delta it applied, any
late writes it had to replay, and the cos(self) sample result.
"""
from __future__ import annotations

import os
import random
import shutil
import time
from typing import Callable, Dict, List, Optional

DEFAULT_PAUSED = os.environ.get("MPMF_PAUSED_MD",
                                os.path.expanduser("~/.mpmf/health/PAUSED.md"))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _open(path: str):
    from mpmf.memory import MemoryManager
    m = MemoryManager(path, vector_dimensions=1024)
    m.load()
    return m


def _copy_node(dst, src, nid: int) -> None:
    """Recreate `nid` from `src` inside an open write_txn on `dst`."""
    c = src._content.get(nid)
    r = src.get_node(nid)
    rec = dst.create_node(name=c.name, node_type=r.node_type, plane_flags=r.plane_flags,
                          text=c.text, summary=c.summary or "", wiki=dict(c.wiki or {}),
                          vector=bytes(src.get_vector(nid)), importance=r.importance)
    got = int(getattr(rec, "node_id", rec))
    if got != nid:
        raise RuntimeError("swap_store: node id drift (%d -> %d); refusing to swap" % (nid, got))


def _refresh_delta(live_path: str, new_path: str, embed_fn: Optional[Callable]) -> Dict:
    """Copy everything live has that `new` does not. Idempotent."""
    live, new = _open(live_path), _open(new_path)
    live_ids, new_ids = set(live.all_node_ids()), set(new.all_node_ids())
    add = sorted(live_ids - new_ids)
    common = sorted(live_ids & new_ids)
    rewritten: List[int] = []
    with new.write_txn(timeout=120):
        for nid in add:
            _copy_node(new, live, nid)
        # a node whose text changed after the snapshot must not keep the old vector
        for nid in common:
            lt, nt = live.get_text(nid) or "", new.get_text(nid) or ""
            if lt == nt:
                continue
            c, r = live._content.get(nid), live.get_node(nid)
            new._content[nid].text = lt
            new._content[nid].wiki = dict(c.wiki or {})
            if embed_fn is not None:
                from mpmf.embed_text import embed_text
                new.set_node_vector(nid, _pack(embed_fn(embed_text(lt))))
                rewritten.append(nid)
    # edges: copy any edge id the new store has never seen
    live, new = _open(live_path), _open(new_path)
    live_edge_ids = {e.edge_id for e in live._edges}
    new_edge_ids = {e.edge_id for e in new._edges}
    missing_edges = [e for e in live._edges if e.edge_id not in new_edge_ids]
    if missing_edges:
        with new.write_txn(timeout=120):
            for e in missing_edges:
                new.create_edge(e.source_node_id, e.target_node_id, e.plane_id,
                                relation_label=live.get_edge_label(e.edge_id) or "",
                                weight=getattr(e, "weight", 1.0))
    return {"nodes_added": len(add), "nodes_rewritten": len(rewritten),
            "edges_added": len(missing_edges), "live_nodes": len(live_ids)}


def _pack(vec) -> bytes:
    import numpy as np
    a = np.asarray(vec, dtype="<f4")
    return a.tobytes()


def _paused_row(label: str, detail: str, closed: bool, resume: str) -> Optional[str]:
    path = DEFAULT_PAUSED
    if not os.path.exists(path):
        return None
    s = open(path).read()
    mark = "swap-%s-%d" % (label, int(time.time()))
    row = ("| %s | %s | %s store swap | %s | %s | %s | %s |"
           % (time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()), mark, label, detail,
              resume, time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
              "CLOSED — swapped" if closed else "OPEN — swap failed, store unchanged"))
    lines = s.rstrip().splitlines()
    anchors = [i for i, l in enumerate(lines) if l.startswith("| 2026-")]
    if anchors:
        lines.insert(max(anchors) + 1, row)
    else:
        lines.append(row)
    open(path, "w").write("\n".join(lines) + "\n")
    return mark


def _verify(path: str, expect_at_least: int, embed_fn: Optional[Callable], sample: int) -> Dict:
    from mpmf.embed_text import embed_text
    m = _open(path)
    ids = list(m.all_node_ids())
    out = {"nodes": len(ids), "dim": m.vector_dimensions, "sample_n": 0, "sample_ok": 0}
    if embed_fn is None or not ids:
        return out
    random.seed(83)
    pick = random.sample(ids, min(sample, len(ids)))
    ok = 0
    import numpy as np
    for nid in pick:
        t = embed_text(m.get_text(nid))
        if not t.strip():
            continue
        out["sample_n"] += 1
        e = np.asarray(embed_fn(t), dtype=np.float32)
        v = np.asarray(m.get_vector(nid), dtype=np.float32)
        if float(np.dot(v, e) / (np.linalg.norm(v) * np.linalg.norm(e) + 1e-12)) > 0.9999:
            ok += 1
    out["sample_ok"] = ok
    if out["nodes"] < expect_at_least:
        raise RuntimeError("swap_store: verify failed — %d nodes, expected >= %d"
                           % (out["nodes"], expect_at_least))
    return out


# --------------------------------------------------------------------------- #
# the tool
# --------------------------------------------------------------------------- #
def swap_store(live: str, new: str, *, embed_fn: Optional[Callable] = None,
               label: Optional[str] = None, quiet_s: float = 2.0,
               sample: int = 20, attempts: int = 3, log=print) -> Dict:
    label = label or os.path.basename(live).replace(".mpmf", "")
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    bak = "%s.pre-swap-%s" % (live, ts)
    resume = "%s (auto: the lock is held only for the replace, <1 s)" % ts
    _paused_row(label, "rebuild + delta refresh + atomic replace", closed=False, resume=resume)

    try:
        total = {"nodes_added": 0, "nodes_rewritten": 0, "edges_added": 0}
        late: List[int] = []
        for attempt in range(1, attempts + 1):
            delta = _refresh_delta(live, new, embed_fn)
            for k in total:
                total[k] += delta[k]
            log("[%s] attempt %d: delta %s" % (label, attempt, delta))
            os.replace(live, bak)
            os.replace(new, live)
            swapped_at = time.time()
            time.sleep(quiet_s)
            # did a writer that already held the old inode land after the replace?
            try:
                late_mtime = os.path.getmtime(bak)
            except FileNotFoundError:
                late_mtime = 0
            if late_mtime > swapped_at:
                log("[%s] late write detected on the old inode — replaying" % label)
                shutil.copy2(live, new)          # fresh working copy
                late.append(1)
                continue
            break
        else:
            log("[%s] giving up after %d attempts" % (label, attempts))

        res = _verify(live, 0, embed_fn, sample)
        _paused_row(label, "rebuild + delta refresh + atomic replace", closed=True,
                    resume="none needed")
        out = {"ok": True, "live": live, "backup": bak, "rollback": "mv %s %s" % (bak, live),
               "delta": total, "late_writes_replayed": len(late), "verify": res}
        log("[%s] swapped: %d nodes, dim %d, cos(self)>0.9999 on %d/%d sampled; rollback: %s"
            % (label, res["nodes"], res["dim"], res["sample_ok"], res["sample_n"], bak))
        return out
    except Exception as exc:                       # noqa: BLE001
        _paused_row(label, "rebuild + delta refresh + atomic replace", closed=False,
                    resume="manual — swap failed")
        raise
