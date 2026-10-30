"""``python -m mpmf_plus.selftest --store PATH [--mode create|attach]``

The mandatory post-install check for an agent (see AGENT_INSTALL.md, step 5).
It answers one question with real I/O, no self-report: *will a memory written
now come back through the same path the model's prompt is built from?*

Steps, each printed as it runs:

  1. resolve the store (path + mode) and validate the recall config
  2. apply ``store_mode``: create initialises a missing store, attach REFUSES
  3. write a unique marker through ``mpmf_plus.memory.store()``
  4. read it back by name through ``mpmf_plus.memory.get()``
  5. recall it by its own text through ``mpmf_plus.memory.recall()``
  6. render the memory block for that text and assert the marker is IN it —
     this is the same block ``hermes_adapter.prefetch()`` hands to the model

Exit code 0 = PASS, 1 = FAIL. Every failure prints ``FAIL: <step>: <type>:
<message>`` so the agent can look the error up in AGENT_INSTALL.md §6.

The marker is left in the store on purpose: step 5 of AGENT_INSTALL.md asks the
agent to send one real turn and find that exact text in the prompt the model
received. Each run adds exactly one node named ``selftest-<hex>``; the core
package has no delete API, so markers accumulate (they are inert, namespaced
``memory``, and easy to grep).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import uuid
from typing import List, Optional

PASS, FAIL = "PASS", "FAIL"

# Marker wording is randomised per run. A fixed template ("MPMF+ self-test
# marker selftest-...: the gate code is ...") is ~0.99 cosine to the previous
# run's marker, so store()'s A2.3 dedup merges it and the store legitimately
# fails to grow by one — which made the SECOND agent's selftest fail on a shared
# store (r60c cold walkthrough: nodes 1 -> 1). Distinct content per run keeps the
# node-count assertion meaningful, which is the whole point of the check.
_SUBJ = ("observatory ferry vaccine orchard turntable ledger kiln lighthouse "
         "aqueduct bookstore windmill").split()
_VERB = ("surveyed repaired catalogued rewired sampled unloaded").split()
_PLACE = ("Riga Valparaiso Yerevan Busan Ljubljana Galway").split()
_UNITS = ("crates ballots panels litres").split()
_FRAMES = (
    "The {place} {subj} was {verb} on {d} and logged {n} {unit}; token {tok}.",
    "Ticket {tok}: the {subj} near the {place} needed {n} {unit} on {d}.",
    "Experiment {tok} ({d}): {verb} a {subj} in {place}, measured {n} {unit}.",
)


def marker_text(marker: str) -> str:
    """A unique, semantically distinct marker fact (dedup-safe across runs)."""
    r = random.Random(uuid.uuid4().hex)
    return _FRAMES[r.randrange(len(_FRAMES))].format(
        subj=r.choice(_SUBJ), verb=r.choice(_VERB), place=r.choice(_PLACE),
        unit=r.choice(_UNITS), n=r.randint(7, 999),
        d=f"2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}", tok=marker)


class SelfTestFailure(RuntimeError):
    """A step failed; carries the step name for the FAIL line."""

    def __init__(self, step: str, detail: str, exit_code: int = 1):
        super().__init__(detail)
        self.step = step
        self.detail = detail
        self.exit_code = exit_code


def _emit(ok: bool, step: str, detail: str = "") -> None:
    tag = "ok  " if ok else "FAIL"
    print(f"[{tag}] {step}" + (f" — {detail}" if detail else ""), flush=True)


def run(store_path: Optional[str] = None, mode: Optional[str] = None,
        as_json: bool = False) -> int:
    """Run every step. Returns the process exit code (0 = PASS)."""
    marker = f"selftest-{uuid.uuid4().hex[:12]}"
    mtext = marker_text(marker)
    steps: List[dict] = []
    step = "1/6 resolve store"

    def record(ok: bool, detail: str) -> None:
        steps.append({"step": step, "ok": ok, "detail": detail})

    try:
        # ---- 1: resolve path + mode, validate the recall config -----------
        from . import memory as mem
        from . import store as store_mod
        from . import block as block_mod

        settings = store_mod.resolve_store(path=store_path, mode=mode)
        _emit(True, step, settings.describe())
        record(True, settings.describe())

        cfg_path = block_mod.RECALL_CFG_PATH
        if not os.path.isfile(str(cfg_path)):
            raise SelfTestFailure(
                step,
                f"recall config missing: {cfg_path}. The renderer needs it — "
                f"write {{\"memory_render\": \"full\"}} there (AGENT_INSTALL.md §3).")
        block_mod.render_cfg()  # raises ConfigError if missing/invalid
        _emit(True, "1/6 recall config", str(cfg_path))

        # ---- 2: store_mode -----------------------------------------------
        step = "2/6 store_mode"
        if settings.mode == "attach":
            store_mod.attach_guard(settings)   # StoreNotFoundError if absent
            _emit(True, step, "attach: store exists, nothing created")
            record(True, "attach ok")
        else:
            created = not store_mod.exists(settings.path)
            store_mod.ensure_usable(settings, vector_dimensions=mem.VECTOR_DIM)
            _emit(True, step, "create: " + ("initialised a new store" if created
                                            else "existing store reused"))
            record(True, "create ok")

        # ---- 3: write -----------------------------------------------------
        step = "3/6 write marker"
        mem.configure_store(path=settings.path, mode=settings.mode)
        before = mem.stats()
        existing = mem.get(marker)
        if existing is not None:  # pragma: no cover - uuid collision
            raise SelfTestFailure(step, f"marker name already present: {marker}")
        mem.store(mtext, name=marker, auto_extract=False, auto_link=False,
                  namespace="memory")
        after = mem.stats()
        if after["nodes"] != before["nodes"] + 1:
            raise SelfTestFailure(
                step,
                f"node count went {before['nodes']} -> {after['nodes']} "
                f"(expected +1). A dedup merge or a stale writer swapped this store.")
        _emit(True, step, f"{marker} (nodes {before['nodes']} -> {after['nodes']})")
        record(True, f"nodes {before['nodes']} -> {after['nodes']}")

        # ---- 4: read back by name ----------------------------------------
        step = "4/6 get() by name"
        got = mem.get(marker)
        if not got:
            raise SelfTestFailure(step, f"get({marker!r}) returned nothing after store()")
        if mtext not in (got.get("text") or ""):
            raise SelfTestFailure(
                step, f"get({marker!r}) text mismatch: {got.get('text')!r}")
        _emit(True, step, "text matches")
        record(True, "text matches")

        # ---- 5: recall by text -------------------------------------------
        step = "5/6 recall()"
        rows = mem.recall(mtext, top_n=5, use_rerank=False,
                          namespaces=["memory"])
        ids = [r.get("node_id") for r in rows]
        nid = got.get("id")
        if nid not in ids:
            raise SelfTestFailure(
                step,
                f"recall() did not return node {nid} for its own text; got ids "
                f"{ids}. Store has the node (step 4 passed) but retrieval lost it.")
        rank = ids.index(nid) + 1
        _emit(True, step, f"marker is rank {rank} of {len(ids)}")
        record(True, f"rank {rank}/{len(ids)}")

        # ---- 6: render the block the model actually receives --------------
        step = "6/6 render prompt block"
        shim = block_mod.wrapper_recall(mtext, 5, mem)
        if shim is None:
            raise SelfTestFailure(step, "wrapper_recall() returned None (recall raised)")
        body, items = block_mod.render_memory_block(
            mtext, shim, mem._open().mem, 5, max_chars=3500)
        if marker not in (body or ""):
            raise SelfTestFailure(
                step,
                f"marker {marker!r} is absent from the rendered memory block. The "
                f"store retrieves it (step 5) but the renderer drops it — the model "
                f"would never see it. Rendered {len(body)} chars, {len(items)} items.")
        _emit(True, step, f"marker present in a {len(body)}-char block")
        record(True, f"block {len(body)} chars")

        print(f"\n{PASS}: store {settings.path!r} mode={settings.mode} "
              f"node={nid} marker={marker}", flush=True)
        print(f"marker text: {mtext}", flush=True)
        print(f"Next (AGENT_INSTALL.md step 5): send one real turn to the model and "
              f"confirm {marker!r} appears in the prompt it received.", flush=True)
        if as_json:
            print(json.dumps({"result": PASS, "store": settings.path,
                              "mode": settings.mode, "marker": marker,
                              "node_id": nid, "steps": steps}, indent=2))
        return 0

    except SelfTestFailure as e:
        _emit(False, e.step, e.detail)
        print(f"\nFAIL: {e.step}: {e.detail}", file=sys.stderr, flush=True)
        if as_json:
            print(json.dumps({"result": FAIL, "step": e.step,
                              "detail": e.detail, "steps": steps}, indent=2))
        return e.exit_code
    except Exception as e:  # any unexpected failure is still a loud failure
        _emit(False, step, f"{type(e).__name__}: {e}")
        print(f"\nFAIL: {step}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        if as_json:
            print(json.dumps({"result": FAIL, "step": step,
                              "detail": f"{type(e).__name__}: {e}",
                              "steps": steps}, indent=2))
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m mpmf_plus.selftest",
        description="Write a unique marker, read it back, recall it, and confirm "
                    "it reaches the rendered memory block. Exit 0 = PASS.")
    ap.add_argument("--store", default=None,
                    help="store path (default: $MPMF_STORE, $MPMF_STORE_PATH, "
                         "config store_path, then $MPMF_HOME/store.mpmf)")
    ap.add_argument("--mode", default=None, choices=["create", "attach"],
                    help="store_mode override (default: $MPMF_STORE_MODE / "
                         "config store_mode / create)")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="also emit a JSON result line")
    a = ap.parse_args(argv)
    return run(store_path=a.store, mode=a.mode, as_json=a.as_json)


if __name__ == "__main__":
    raise SystemExit(main())
