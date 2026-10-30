"""Round 60 / B4 — two processes x 25 writes into ONE store, simultaneous.

Each writer calls the package's public write path (``mpmf_plus.memory.store``)
25 times: 50 nodes must appear, none lost to the other writer's whole-file
image.

Two things this script is careful about:

* **Distinct facts.** ``store()`` runs A2.1/A2.3 dedup: an exact-text hit, or a
  cosine >= 0.95 to an existing node, is MERGED instead of created (by design).
  A templated probe ("probe 01/02/03...") is ~0.99 similar, so it would report a
  fake "lost writes" failure. Each probe therefore carries distinct content
  words AND a random token.
* **Per-write verification.** After every ``store()`` the writer calls
  ``memory.get(name)``; a missing name means the write was merged or lost, and
  is counted and reported loudly instead of being inferred from the total.

Usage:
  python concurrency.py writer --store P --tag A --n 25 --go FILE
  python concurrency.py count  --store P --checkpoint FILE
"""
import argparse
import json
import os
import random
import sys
import time
import uuid

NOUNS = ("harbour turbine alloy vaccine glacier ledger anvil compass orchard "
         "lantern furnace marrow cipher quarry saddle thicket beacon cauldron "
         "mosaic pelican ravine tundra bramble cinder dune falcon girder juniper "
         "kettle lagoon mesa nectar obelisk prairie quartz reef sable thistle "
         "umber vellum willow zephyr").split()
ADJS = ("cobalt brittle silent porous salty lunar dusty braided molten jagged "
        "velvet hollow restless copper spectral narrow humid crimson glacial "
        "polished ragged wolfish tranquil").split()
PLACES = ("Reykjavik Trieste Oaxaca Kyoto Tallinn Recife Perth Nagoya Cusco "
          "Bergen Muscat Hobart Galway Kandy Durban Leeds Odense").split()
VERBS = ("repaired sold painted borrowed stole buried measured shipped "
         "invented cancelled decoded").split()
UNITS = ("litres grams volts metres calories sheets").split()

# Six unrelated sentence frames. Dedup merges at cosine >= 0.95 (A2.3), so two
# probes must not share a template: sharing one is what produced a fake
# "lost writes" number (0.954-0.9575) on the first attempt at this test.
FRAMES = (
    "Please fix the {adj} {noun} in {place} tomorrow; ticket {hx} says it takes {n} {unit}.",
    "Why did the {noun} vanish from {place}? {hx} claims {n} {unit} of {adj} debris were found.",
    "Recipe: beat {n} {unit} of sugar with {adj} {noun}, then chill for the {place} fair.",
    "{hx} replaced the {adj} {noun} at the {place} depot with {n} {unit} of rope.",
    "Interview notes {hx}: the {place} council {verb} the {adj} {noun} after {n} complaints.",
    "Do not {verb} a {adj} {noun} without {n} {unit} of water and a {hx} permit.",
)


def make_fact(tag, i):
    rnd = random.Random(uuid.uuid4().hex)
    return FRAMES[i % len(FRAMES)].format(
        adj=rnd.choice(ADJS), noun=rnd.choice(NOUNS), place=rnd.choice(PLACES),
        verb=rnd.choice(VERBS), unit=rnd.choice(UNITS),
        n=rnd.randint(11, 999), hx=uuid.uuid4().hex[:10])


def writer(store_path, tag, n, go_file):
    from mpmf_plus import memory
    memory.configure_store(path=store_path, mode="attach")
    checks = 0
    while True:                      # rendezvous: both writers start together
        if os.path.exists(go_file):
            break
        if checks > 24000:
            raise SystemExit("writer: GO file never appeared")
        checks += 1
        time.sleep(0.005)

    verified, merged, errors = 0, [], []
    for i in range(n):
        name = f"conc60_{tag}_{i:02d}"
        text = make_fact(tag, i)
        try:
            memory.store(text, name=name, auto_extract=False, auto_link=False,
                         namespace="memory")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue
        if memory.get(name) is None:
            merged.append(name)      # dedup merged it, or the node was lost
        else:
            verified += 1
    print(f"writer {tag}: wrote {n}, verified_present={verified}, "
          f"merged_or_lost={len(merged)}, errors={len(errors)}")
    if merged:
        print(f"writer {tag}: NOT created: {merged}")
    for e in errors:
        print(f"writer {tag}: ERROR {e}")
    if verified != n or errors:
        sys.exit(1)


def count(store_path, checkpoint):
    from mpmf_plus import memory
    memory.configure_store(path=store_path, mode="attach")
    st = memory.stats()
    mm = memory._open().mem
    names = {mm.get_name(nid) for nid in mm.all_node_ids()}
    a = sum(1 for k in names if k.startswith("conc60_A_"))
    b = sum(1 for k in names if k.startswith("conc60_B_"))
    with open(checkpoint) as f:
        cp = json.load(f)
    delta = st["nodes"] - cp["nodes_before"]
    print(f"count: nodes_before={cp['nodes_before']} nodes_after={st['nodes']} delta={delta}")
    print(f"count: tag A present={a}/25  tag B present={b}/25  total conc60={a + b}")
    ok = (delta == 50 and a == 25 and b == 25)
    print(("CONCURRENCY PASS" if ok else "CONCURRENCY FAIL") +
          f": expected exactly +50 with 0 missing, got +{delta} (A={a}, B={b})")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["writer", "count"])
    ap.add_argument("--store", required=True)
    ap.add_argument("--tag", default="A")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--go", default="")
    ap.add_argument("--checkpoint", default="")
    a = ap.parse_args()
    if a.role == "writer":
        writer(a.store, a.tag, a.n, a.go)
        return 0
    return count(a.store, a.checkpoint)


if __name__ == "__main__":
    sys.exit(main())
