"""Preflight for text-changing crons: is the text -> vector guard ARMED?

A job that rewrites a node's text must re-embed it in the same write transaction, or
the stored vector stops matching the text (the S_* drift the r103 audit found at
cos 0.78-0.97). ``mpmf.memory.set_node_text`` enforces that: if the text changed and
the store has no embedder, the write_txn REFUSES to save.

That refusal is only useful if it happens in a DRY RUN, not at 04:00 in a cron. This
module answers the question on scratch stores, so it never touches the live data:

  1. the store refuses a text change with no embedder            (the guard exists)
  2. MPMFMemory installs the canonical embed_text() hook         (so it can auto-fix)
  3. an auto-embedding store really changes the vector           (the hook works)

Exit code 0 = armed, 1 = not armed. ``python -m mpmf_plus.drycheck``.
"""
from __future__ import annotations

import os
import sys
import tempfile
from typing import List

_DIM = 8


def guard_errors() -> List[str]:
    """Return [] when the guard is armed, else one line per failure."""
    errs: List[str] = []
    try:
        import numpy as np

        from mpmf.format import NodeType
        from mpmf.query import MPMFMemory
    except Exception as e:                                  # pragma: no cover
        return ["cannot import the store layer: %s" % (e,)]

    tmp = tempfile.mkdtemp(prefix="mpmf-drycheck-")

    def _new(name: str) -> MPMFMemory:
        return MPMFMemory(os.path.join(tmp, name), vector_dimensions=_DIM)

    def _seed(m: MPMFMemory) -> int:
        with m.mem.write_txn():
            rec = m.mem.create_node("probe", NodeType.FACT, 1)
            nid = getattr(rec, "node_id", getattr(rec, "id", rec))
            m.mem.set_node_text(nid, "original text")
            m.mem.set_node_vector(nid, np.ones(_DIM, dtype=np.float32))
        return int(nid)

    # 1 — a writer with NO embedder must be refused, not silently drifted.
    try:
        bare = _new("bare.mpmf")
        bare.mem.embedder = None
        nid = _seed(bare)
        try:
            with bare.mem.write_txn():
                bare.mem.set_node_text(nid, "changed without a vector")
        except RuntimeError as e:
            if "without set_node_vector" not in str(e):
                errs.append("guard raised the wrong error: %s" % (e,))
        else:
            errs.append("a text change with no embedder was ACCEPTED — the guard is not armed")
    except Exception as e:
        errs.append("guard probe failed: %s" % (e,))

    # 2 — the shipped writer installs the hook.
    try:
        m = _new("hooked.mpmf")
        if getattr(m.mem, "embedder", None) is None:
            errs.append("MPMFMemory did not install a text->vector hook")
        else:
            nid = _seed(m)
            before = list(m.mem.get_vector_list(nid) or [])
            with m.mem.write_txn():
                m.mem.set_node_text(nid, "completely different text")
            after = list(m.mem.get_vector_list(nid) or [])
            # 3 — and the hook actually re-embeds.
            if before == after:
                errs.append("the hook is installed but the vector did not change")
            if m.mem.get_text(nid) != "completely different text":
                errs.append("the text change did not persist")
    except Exception as e:
        errs.append("auto-embed probe failed: %s" % (e,))

    return errs


def main(argv=None) -> int:
    errs = guard_errors()
    if errs:
        print("drycheck: text/vector guard NOT armed", file=sys.stderr)
        for e in errs:
            print("  - %s" % e, file=sys.stderr)
        print("drycheck: a cron that changes text would fail (or drift) at run time", file=sys.stderr)
        return 1
    print("drycheck: text/vector guard armed (refuses text-without-vector; "
          "MPMFMemory re-embeds via embed_text = first 512 chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
