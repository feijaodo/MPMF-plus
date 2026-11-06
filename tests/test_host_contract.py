#!/usr/bin/env python3
"""Host contract test — the check that would have caught round 89d.

Hermes calls a fixed set of methods on a memory provider. The r89 package port
dropped two of them (get_tool_schemas, handle_tool_call) and nothing noticed
until a live restart: "Memory provider plugin init failed". This test loads the
plugin exactly the way the host does (spec_from_file_location + exec_module +
register), derives the contract from the pre89 class, and then calls every
method once against a throwaway store copy.

Run:  python3 tests/test_host_contract.py            (exit 0 = contract holds)
      python3 tests/test_host_contract.py --verbose
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import traceback

LIVE_SRC = os.environ.get("MPMF_LIB_PATH", "/root/mpmf")
PLUGIN = os.environ.get(
    "MPMF_PLUGIN", "/root/.hermes/plugins/mpmf_plus_user/__init__.py")
LEGACY = os.environ.get(
    "MPMF_LEGACY_PLUGIN",
    "/root/.hermes/plugins/mpmf_plus_user/__init__.pre89.keep.py")
SMALL_STORE = os.environ.get("MPMF_TEST_STORE", "/root/.mpmf/pre.mpmf")

# Every method Hermes is known to call on a memory provider, in call order.
HOST_CALLS = [
    "name", "is_available", "initialize", "system_prompt_block",
    "get_tool_schemas", "handle_tool_call", "prefetch", "sync_turn",
    "on_pre_tool_call", "on_session_end", "shutdown",
]


def load_plugin(path: str, modname: str):
    """Load a plugin module the way Hermes does (file path, not package)."""
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def public_methods(cls) -> set:
    return {n for n in dir(cls)
            if not n.startswith("_") and callable(getattr(cls, n, None))}


def main(argv=None) -> int:
    verbose = "--verbose" in (argv or sys.argv)
    failures: list[str] = []

    if LIVE_SRC not in sys.path:
        sys.path.insert(0, LIVE_SRC)

    tmp = tempfile.mkdtemp(prefix="mpmf-contract-")
    store = os.path.join(tmp, "store.mpmf")
    shutil.copy2(SMALL_STORE, store)
    os.environ["MPMF_HOME"] = tmp
    os.environ["MPMF_STORE_PATH"] = store
    os.environ.setdefault("MPMF_THINKING_STORE", store)

    try:
        plugin = load_plugin(PLUGIN, "mpmf_plus_user_contract")
        cls = getattr(plugin, "MPMFPlusProvider", None)
        if cls is None:
            return _fail(["plugin exposes no MPMFPlusProvider"])

        # 1. contract from the pre89 class (what the host has always called)
        expected = set(HOST_CALLS)
        pre = public_methods(cls)
        if os.path.exists(LEGACY):
            legacy = load_plugin(LEGACY, "mpmf_plus_user_legacy")
            pre = public_methods(legacy.MPMFPlusProvider)
            expected |= {n for n in pre if n in HOST_CALLS}
        missing = sorted(n for n in expected if not hasattr(cls, n))
        if missing:
            failures.append("missing host methods: %s" % ", ".join(missing))

        # 2. call every one of them once, on the copy
        prov = cls()
        prov.initialize(session_id="contract-test")
        if verbose:
            print("initialize ok")

        if len(prov.get_tool_schemas() or []) < 2:
            failures.append("get_tool_schemas returned < 2 tools")

        rec = prov.handle_tool_call("mpmf_recall", {"query": "contract test", "top_k": 2})
        if not rec or "error" in rec[:40]:
            failures.append("handle_tool_call(mpmf_recall) -> %s" % str(rec)[:160])

        st = prov.handle_tool_call(
            "mpmf_store",
            {"text": "Host contract test fact for round 89e.", "name": "contract_89e"})
        if "error" in str(st)[:40]:
            failures.append("handle_tool_call(mpmf_store) -> %s" % str(st)[:160])

        block = prov.prefetch("host contract test query")
        if not isinstance(block, str):
            failures.append("prefetch did not return str")

        prov.sync_turn("user said this", "assistant answered that", session_id="contract-test")
        hook = prov.on_pre_tool_call(tool_name="terminal", args={"command": "ls"})
        if hook is not None and not isinstance(hook, dict):
            failures.append("on_pre_tool_call returned %r" % type(hook))
        if hasattr(prov, "system_prompt_block"):
            prov.system_prompt_block()
        if hasattr(prov, "on_session_end"):
            prov.on_session_end(session_id="contract-test")
        prov.shutdown()
        if verbose:
            print("all calls ok")
    except Exception:
        failures.append("exception:\n" + traceback.format_exc())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return _fail(failures, verbose)


def _fail(failures, verbose=False) -> int:
    if failures:
        print("HOST CONTRACT FAILED (%d)" % len(failures))
        for f in failures:
            print(" -", f)
        return 1
    if verbose:
        print("HOST CONTRACT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
