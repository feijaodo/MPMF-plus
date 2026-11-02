"""``mpmf-rebuild`` CLI: export -> extraction -> stage 2 -> retag -> manifest."""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

from .extraction import (BudgetExceeded, FAILED, load_export, persist_facts,
                         run_extraction)
from .retag import retag_modes
from .stage2 import build_states, classify_facts


def _default_client(model: str):
    """OpenRouter chat client. Imported lazily so tests need no network."""
    import os

    import requests

    key = os.environ.get("OPENROUTER_API_KEY") or ""

    def call(prompt: str) -> str:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "temperature": 0, "max_tokens": 1200,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=120,
        ).json()
        return ((r.get("choices") or [{}])[0].get("message", {}) or {}).get("content") or ""

    return call


def build_store(export_path: str, out_path: str, cap: float = 0.0,
                workers: int = 8, model: str = "deepseek/deepseek-v4.1-flash",
                call_model=None, mem_factory=None, skip_retag: bool = False) -> Dict[str, Any]:
    """Run all three stages and return the manifest (also written beside the store)."""
    call_model = call_model or _default_client(model)
    convs = load_export(export_path)
    stage1 = run_extraction(convs, call_model, cap_usd=cap, workers=workers)

    if mem_factory is None:
        from mpmf_plus.memory import _open_rw as mem_factory      # noqa: WPS433
    mem = mem_factory(out_path)

    rows = {r["conv"]: r for r in stage1["rows"]}
    facts: List[Dict[str, Any]] = []

    with mem.write_txn():
        def make_node(name, text, summary, wiki):
            vec = mem.embed(text) if hasattr(mem, "embed") else [0.0] * 1024
            node = mem.create_node(name=name, node_type=1, plane_flags=1, text=text,
                                   summary=summary, wiki=wiki, vector=vec, importance=0.6)
            return int(getattr(node, "node_id", node))

        for conv in convs:
            rec = rows.get(conv.get("id"))
            if not rec or rec["status"] == FAILED:
                continue                       # a failed conversation is never zeroed
            ids = persist_facts(mem, conv, rec["facts"], make_node)
            for f, nid in zip(rec["facts"], ids):
                facts.append(dict(f, _conv_id=conv.get("id"),
                                  _said_at=(conv.get("start") or "")[:10], name=nid))

        facts = classify_facts(facts, workers=workers)

        def make_state(name, text, summary, wiki):
            return make_node(name, text, summary, wiki)

        def link(src, dst, label):
            mem.create_edge(int(src), int(dst), 1, label)

        # States + their IS_A edges are created inside THIS transaction.
        stage2 = build_states(mem, facts, create_node=make_state, link=link, workers=workers)

    msgs_by_conv = {c.get("id"): [(m.get("text") or "") for m in (c.get("messages") or [])]
                    for c in convs}
    if skip_retag:
        stage3 = {"facts": 0, "flips": {}}
    else:
        stage3 = retag_modes(facts, lambda cid: msgs_by_conv.get(cid, []), call_model,
                             workers=workers)

    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "export": export_path, "store": out_path, "model": model,
        "cap_usd": cap, "workers": workers,
        "stage1": {k: stage1[k] for k in ("conversations", "ok", "empty", "failed",
                                         "facts", "spent_usd")},
        "stage2": stage2,
        "stage3": {"facts": stage3["facts"], "flips": stage3["flips"]},
        "schema_fields": ["entity", "_msg_index", "_evidence", "_fact_date", "_said_at", "_conv_id"],
    }
    with open(out_path + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="mpmf-rebuild", description=__doc__)
    ap.add_argument("--input", required=True, help="conversation export (JSONL)")
    ap.add_argument("--out", required=True, help="output store path")
    ap.add_argument("--cap", type=float, default=0.0, help="spend cap in USD (0 = no cap)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    ap.add_argument("--no-retag", action="store_true", help="skip the context-aware mode re-tag")
    args = ap.parse_args(argv)
    try:
        manifest = build_store(args.input, args.out, args.cap, args.workers, args.model,
                               skip_retag=args.no_retag)
    except BudgetExceeded as exc:
        print(f"mpmf-rebuild: {exc}", file=sys.stderr)
        return 3
    s1 = manifest["stage1"]
    print(f"mpmf-rebuild: {s1['conversations']} conversations, {s1['facts']} facts, "
          f"failed={s1['failed']} states={manifest['stage2']['states']} "
          f"${s1['spent_usd']:.4f} -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
