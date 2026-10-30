"""Round 60 / B3 — freshness across processes, one shared store.

Two SEPARATE OS processes:

  writer  : waits for the reader to say it is warm, then writes a unique marker
  reader  : warms its read snapshot, announces readiness, waits, then recalls
            the marker WITHOUT restarting or re-importing anything

Usage:
  python freshness.py writer --store P --handshake DIR
  python freshness.py reader --store P --handshake DIR

Reader exits 0 and prints "FRESHNESS PASS" only if recall() returns the marker
node it never wrote, using the read path the agent actually calls.
"""
import argparse
import os
import sys
import time
import uuid

READY = "ready"
WROTE = "wrote.json"
RESULT = "reader.json"


def writer(store_path, hs, delay=0.0):
    from mpmf_plus import memory
    memory.configure_store(path=store_path, mode="attach")
    marker = f"fresh60_{uuid.uuid4().hex[:10]}"
    text = f"Round 60 freshness marker {marker}: the vault code is {marker[-4:].upper()}."
    # wait until the reader has warmed its snapshot
    t0 = time.time()
    while not os.path.exists(os.path.join(hs, READY)):
        if time.time() - t0 > 120:
            raise SystemExit("writer: reader never announced readiness")
        time.sleep(0.05)
    time.sleep(delay)
    before = memory.stats()["nodes"]
    memory.store(text, name=marker, auto_extract=False, auto_link=False,
                 namespace="memory")
    after = memory.stats()["nodes"]
    with open(os.path.join(hs, WROTE), "w") as f:
        f.write(f"{marker}\n{text}\n{before}\n{after}\n")
    print(f"writer: wrote {marker} ({before} -> {after} nodes)")


def reader(store_path, hs, poll=120.0):
    from mpmf_plus import memory
    memory.configure_store(path=store_path, mode="attach")
    st = memory.stats()          # warms the memoised snapshot in THIS process
    print(f"reader: snapshot warm at {st['nodes']} nodes, path={st['path']}")
    open(os.path.join(hs, READY), "w").close()
    t0 = time.time()
    while not os.path.exists(os.path.join(hs, WROTE)):
        if time.time() - t0 > poll:
            print("FRESHNESS FAIL: writer never wrote")
            return 1
        time.sleep(0.05)
    with open(os.path.join(hs, WROTE)) as f:
        marker, text = f.read().split("\n")[:2]

    # THE test: same process, same module object, no restart, no reconfigure.
    rows = memory.recall(text, top_n=5, use_rerank=False, namespaces=["memory"])
    ids = [r.get("node_id") for r in rows]
    got = memory.get(marker)
    hit = bool(got) and got.get("id") in ids
    elapsed = time.time() - t0
    print(f"reader: recall({text[:40]!r}...) -> ids={ids} after {elapsed:.2f}s")
    print(f"reader: get({marker!r}) -> {'found' if got else 'MISSING'}")
    if hit:
        print(f"FRESHNESS PASS: reader saw {marker} written by another process "
              f"(rank {ids.index(got['id']) + 1}/{len(ids)}), no restart")
        return 0
    print("FRESHNESS FAIL: the reader could not see the marker written by the other process")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["writer", "reader"])
    ap.add_argument("--store", required=True)
    ap.add_argument("--handshake", required=True)
    ap.add_argument("--delay", type=float, default=0.0)
    a = ap.parse_args()
    os.makedirs(a.handshake, exist_ok=True)
    if a.role == "writer":
        writer(a.store, a.handshake, a.delay)
        return 0
    return reader(a.store, a.handshake)


if __name__ == "__main__":
    sys.exit(main())
