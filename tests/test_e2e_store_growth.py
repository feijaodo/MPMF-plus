#!/usr/bin/env python3
"""End-to-end effect test: a turn must leave a fact in the store.

The contract test proves the methods exist; this proves they *do* something.
Round 90b's failure passed every shape check and still wrote nothing, so the
deploy gate now requires:

  sync_turn(real turn) -> node count grows -> new fact found by prefetch

Exit 0 = the write path works. Any other exit = refuse the deploy.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import traceback

LIVE_SRC = os.environ.get("MPMF_LIB_PATH", "/root/mpmf")
SMALL_STORE = os.environ.get("MPMF_TEST_STORE", "/root/.mpmf/pre.mpmf")


def main() -> int:
    if LIVE_SRC not in sys.path:
        sys.path.insert(0, LIVE_SRC)
    tmp = tempfile.mkdtemp(prefix="mpmf-e2e-")
    store = os.path.join(tmp, "store.mpmf")
    shutil.copy2(SMALL_STORE, store)
    os.environ["MPMF_HOME"] = tmp

    marker = "zq%d" % int(time.time())
    user = ("Round e2e %s: my locker code is %s and the spare key is under the "
            "blue pot on the top shelf." % (marker, marker))
    assistant = ("Noted for %s: locker code %s, spare key under the blue pot on "
                 "the top shelf." % (marker, marker))

    try:
        from mpmf_plus import provider
        provider.configure(PATH=store)
        log_path = os.path.join(tmp, "store.log")
        provider.STORE_LOG = log_path
        from mpmf_plus import hermes_adapter

        prov = hermes_adapter.MPMFPlusProvider()
        prov.initialize("e2e-gate")
        before = provider._open().mem.header.node_count
        prov.sync_turn(user, assistant, session_id="e2e-gate")
        after = provider._open().mem.header.node_count

        if after <= before:
            print("E2E FAILED: store did not grow (%d -> %d)" % (before, after))
            print("  sync_turn returned without persisting a fact")
            return 1

        # r90d: the turn must ingest under the Hermes session id, not a
        # throwaway turn_<epoch> conversation.
        conv = ""
        try:
            with open(log_path) as f:
                for line in f:
                    if "conv=" in line:
                        conv = line.split("conv=", 1)[1].split()[0]
        except OSError:
            pass
        print("E2E ingest conv=%r (want %r)" % (conv, "e2e-gate"))
        if conv != "e2e-gate":
            print("E2E FAILED: ingest conv id is not the session id")
            return 1

        block = prov.prefetch("where is the spare key and what is the locker code") or ""
        found = marker in block or "blue pot" in block
        print("E2E store growth: %d -> %d (+%d); prefetch found fact: %s"
              % (before, after, after - before, found))
        if not found:
            print("E2E FAILED: fact not retrievable by prefetch")
            return 1
    except Exception:
        print("E2E FAILED: exception\n" + traceback.format_exc())
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("E2E OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
