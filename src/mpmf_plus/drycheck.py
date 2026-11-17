"""Preflight for source-changing crons: is the embed-source -> vector guard ARMED?

A node's vector represents ONE string (r105: ``embed_from or summary or text``,
truncated by ``embed_text``). A job that rewrites the winning string must re-embed
in the same write transaction, or the stored vector stops matching its own node --
which is how 1,783 summary-embedded vectors were mistaken for drift in r103.

``mpmf.memory`` enforces it: if the embed SOURCE changed and the store has no
embedder, the write_txn REFUSES to save. That refusal is only useful in a DRY RUN,
not at 04:00 in a cron. This module answers the question on scratch stores, so it
never touches live data:

  1. the store refuses a source change with no embedder         (the guard exists)
  2. MPMFMemory installs the canonical embed_source() hook      (so it can self-fix)
  3. a text-as-source node re-embeds on a text edit             (the hook works)
  4. a summary-as-source node does NOT re-embed on a text edit  (the rule is the
     source, not the text) but DOES on a summary edit

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

    def _seed(m: MPMFMemory, summary: str = "") -> int:
        with m.mem.write_txn():
            rec = m.mem.create_node("probe", NodeType.FACT, 1)
            nid = getattr(rec, "node_id", getattr(rec, "id", rec))
            m.mem.set_node_text(nid, "original text")
            m.mem.set_node_summary(nid, summary)
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
            errs.append("a source change with no embedder was ACCEPTED — the guard is not armed")
    except Exception as e:
        errs.append("guard probe failed: %s" % (e,))

    # 2/3 — the shipped writer installs the hook, and text-as-source re-embeds.
    try:
        m = _new("hooked.mpmf")
        if getattr(m.mem, "embedder", None) is None:
            errs.append("MPMFMemory did not install a source->vector hook")
        else:
            nid = _seed(m)
            before = list(m.mem.get_vector_list(nid) or [])
            with m.mem.write_txn():
                m.mem.set_node_text(nid, "completely different text")
            after = list(m.mem.get_vector_list(nid) or [])
            if before == after:
                errs.append("the hook is installed but the vector did not change")
            if m.mem.get_text(nid) != "completely different text":
                errs.append("the text change did not persist")
    except Exception as e:
        errs.append("auto-embed probe failed: %s" % (e,))

    # 4 — summary-as-source: text edits are inert, summary edits re-embed.
    try:
        s = _new("summary.mpmf")
        nid = _seed(s, summary="a curated summary")
        v0 = list(s.mem.get_vector_list(nid) or [])
        with s.mem.write_txn():
            s.mem.set_node_text(nid, "the text moved, the source did not")
        v1 = list(s.mem.get_vector_list(nid) or [])
        if v0 != v1:
            errs.append("editing the text of a summary-sourced node re-embedded it "
                        "(the rule must key on the source, not the text)")
        with s.mem.write_txn():
            s.mem.set_node_summary(nid, "a different curated summary")
        v2 = list(s.mem.get_vector_list(nid) or [])
        if v1 == v2:
            errs.append("editing the summary of a summary-sourced node did not re-embed")
    except Exception as e:
        errs.append("summary-source probe failed: %s" % (e,))

    return errs


def main(argv=None) -> int:
    errs = guard_errors()
    if errs:
        print("drycheck: embed-source/vector guard NOT armed", file=sys.stderr)
        for e in errs:
            print("  - %s" % e, file=sys.stderr)
        print("drycheck: a cron that changes a source would fail (or drift) at run time",
              file=sys.stderr)
        return 1
    print("drycheck: embed-source/vector guard armed (refuses source-without-vector; "
          "MPMFMemory re-embeds via embed_source = embed_from|summary|text, first 512 chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
