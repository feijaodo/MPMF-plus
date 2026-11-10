"""P17 compaction tool: fold append-only segments back into one base image.

    python -m mpmf.compact STORE [--dim 1024] [--legacy] [--dry-run]

Takes the store's cross-process lock, replays every segment, writes a fresh
image with a tempfile + fsync + rename (the same atomic path as a normal full
save), then re-opens the result and checks node/edge/vector parity with what it
folded. --legacy writes a plain pre-P17 image (no segment flag), which is the
way back for older code. --dry-run reports sizes without touching the file.
"""
from __future__ import annotations

import argparse
import sys

from .memory import MemoryManager


def _signature(m: MemoryManager):
    ids = sorted(m._nodes)
    return (ids,
            [m._node_sig(i) for i in ids],
            sorted((e.edge_id, m._edge_sig(e)) for e in m._edges),
            {i: bytes(m.get_vector(i)) for i in ids if m.has_vector(i)})


def compact(path: str, dim: int = 1024, legacy: bool = False, dry_run: bool = False) -> dict:
    m = MemoryManager(path, vector_dimensions=dim, append_segments=False)
    m.load()
    if m.header is not None and m._seg_base_end is None and not legacy:
        pass  # a plain image compacts to itself (rewrite); fine
    folded = m._seg_count
    report = {"path": path, "segments": folded, "nodes": len(m._nodes),
              "edges": len(m._edges)}
    if dry_run:
        report["dry_run"] = True
        return report
    sig = _signature(m)
    keep = not legacy and m._seg_base_end is not None
    res = m.compact(keep_segments=keep)
    check = MemoryManager(path, vector_dimensions=dim, append_segments=False)
    check.load()
    if _signature(check) != sig or check._seg_count != 0:
        raise SystemExit("compact: parity check FAILED after rewrite of %s" % path)
    report.update(res)
    report["parity"] = "ok"
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mpmf.compact")
    ap.add_argument("store")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--legacy", action="store_true",
                    help="write a plain pre-P17 image (no segment flag)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    print(compact(a.store, a.dim, a.legacy, a.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
