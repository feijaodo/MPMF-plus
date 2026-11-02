"""r69 regression tests for the two silent failures.

A. store() must never near-dup merge kind=state/category/catstate; a fact merge
   returns {"merged_into": id} and is logged.
B. create_node / create_edge / update_node_wiki outside write_txn() must raise;
   inside it must persist across a reopen.
"""
import json, os, shutil, sys, tempfile

SRC = "/root/mpmf-release/src"
sys.path.insert(0, SRC)
from mpmf_plus import memory                      # noqa: E402
from mpmf_plus.memory import _pack                # noqa: E402

TMP = tempfile.mkdtemp(prefix="r69pkg_")
STORE = os.path.join(TMP, "s.mpmf")
HOME = os.path.join(TMP, "home")
os.makedirs(HOME, exist_ok=True)
os.environ.update(MPMF_HOME=HOME, MPMF_RECALL_CONFIG=os.path.join(HOME, "recall.config.json"))
memory.configure_store(path=STORE, mode="create" if "create" in str(memory.store_settings()[1]).lower() else "attach")
print("store settings:", memory.store_settings())

# bootstrap an empty store image
from mpmf import MemoryManager                      # noqa: E402
MemoryManager(STORE, vector_dimensions=1024).periodic_save()
memory.configure_store(path=STORE, mode="attach")

# ---- A: structural kinds never merge ----
FACT = "The user owns a Volvo V60 that they lease for about 250 dollars a month."
n1 = memory.store(FACT, name="f_seed", auto_extract=False, auto_link=False,
                  namespace="memory", wiki={"kind": "fact", "entity": "volvo v60"})
assert isinstance(n1, str), f"fact create should return a name, got {n1!r}"
print("A0 fact created:", n1)

CS1 = "The user owns a Volvo V60 that they lease for about 250 dollars a month, mainly."
r1 = memory.store(CS1, name="CS_car", auto_extract=False, auto_link=False, namespace="memory",
                  wiki={"kind": "catstate", "entity": "car", "as_of": "2026-02-21"})
assert isinstance(r1, str), f"A FAIL: catstate was merged, store() returned {r1!r}"
r2 = memory.store(CS1 + " Update.", name="CS_car2", auto_extract=False, auto_link=False,
                  namespace="memory", wiki={"kind": "catstate", "entity": "car"})
assert isinstance(r2, str), f"A FAIL: second near-identical catstate merged: {r2!r}"
mm = memory._open().mem
kinds = {}
for i in mm.all_node_ids():
    w = mm.get_wiki(i) or {}
    if w.get("kind") == "catstate": kinds[mm.get_name(i)] = (i, w.get("entity"))
print("A1 catstates on disk after reopen:", kinds)
assert len(kinds) == 2, f"A FAIL: expected 2 catstates on disk, got {kinds}"
ids = [v[0] for v in kinds.values()]
assert len(set(ids)) == 2, f"A FAIL: catstate ids collided: {ids}"
print("A2 distinct ids, both persisted:", ids)

# fact merge returns a dict and creates nothing
r3 = memory.store(FACT, name="f_dup", auto_extract=False, auto_link=False, namespace="memory",
                  wiki={"kind": "fact", "entity": "volvo v60"})
print("A3 fact near-dup return:", r3)
assert isinstance(r3, dict) and "merged_into" in r3, f"A FAIL: merge did not return dict: {r3!r}"
assert mm.get_node_id_by_name("f_dup") is None, "A FAIL: merged fact was also created as a new node"
print("A4 merge reported, no new node:", r3)

# ---- B: writes outside write_txn must raise ----
mm2 = memory._open().mem
victim = mm2.get_node_id_by_name(mm2.get_name(mm2.all_node_ids()[0]))
for fn, label in [
    (lambda: mm2.update_node_wiki(victim, {"_probe": "1"}), "update_node_wiki"),
    (lambda: mm2.create_node(name="nope", node_type=1, plane_flags=1, text="x",
                             summary="x", wiki={}, vector=_pack("x")), "create_node"),
    (lambda: mm2.create_edge(victim, victim, memory.Plane.CAUSAL, "X"), "create_edge"),
]:
    try:
        fn()
    except RuntimeError as e:
        print(f"B {label}: raised OK -> {str(e)[:78]}...")
    else:
        raise AssertionError(f"B FAIL: {label} did NOT raise outside write_txn")

with mm2.write_txn():
    mm2.update_node_wiki(victim, {"_probe": "inside-txn"})
    mm2.create_node(name="r69_probe", node_type=mm2.get_node(victim).node_type,
                    plane_flags=mm2.get_node(victim).plane_flags, text="probe node",
                    summary="probe node", wiki={"kind": "fact", "entity": "probe"},
                    vector=_pack("probe node"))
print("B4 inside txn: no raise")
mm3 = memory._open().mem
pid = mm3.get_node_id_by_name("r69_probe")
probe = (mm3.get_wiki(victim) or {}).get("_probe")
print("B5 after reopen: probe=", probe, "| r69_probe id=", pid)
assert probe == "inside-txn", f"B FAIL: wiki edit did not persist ({probe!r})"
assert pid is not None, "B FAIL: created node did not persist"
print("B6 both persisted after reopen")
print("ALL r69 PACKAGE TESTS PASSED")
