"""f1: re-ingest turns from the ingest dead-letter file.

  python -m mpmf_plus.reingest --list
  python -m mpmf_plus.reingest --from-state-db SESSION_ID MSG_ROW_ID --msg-index N
        # dead-letter a turn that was lost BEFORE the dead-letter file existed
        # (user message + the session's reply, rebuilt from Hermes state.db), then:
  python -m mpmf_plus.reingest --run          # ingest every unresolved row

Nothing is deleted: a row stays in ingest_dead_letter.jsonl; success appends
{conv, msg, ts, facts} to ingest_dead_letter.resolved.jsonl and the row stops counting.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import ingest as _ing


def _read(path: str) -> List[Dict[str, Any]]:
    try:
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    except FileNotFoundError:
        return []


def resolved_file(dead: str) -> str:
    return dead[:-6] + ".resolved.jsonl" if dead.endswith(".jsonl") else dead + ".resolved"


def unresolved(dead_letter_path: Optional[str] = None) -> List[Dict[str, Any]]:
    dead = _ing.dead_letter_file(dead_letter_path)
    done = {(r.get("conv"), r.get("msg")) for r in _read(resolved_file(dead))}
    seen, out = set(), []
    for r in _read(dead):
        k = (r.get("conv"), r.get("msg"))
        if k in done or k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def reingest(store: Any, dead_letter_path: Optional[str] = None, log_path: Optional[str] = None,
             **ingest_kw) -> List[Dict[str, Any]]:
    """Ingest every unresolved dead-letter row; returns [{conv, msg, status, facts|error}]."""
    dead = _ing.dead_letter_file(dead_letter_path)
    results = []
    for r in unresolved(dead):
        try:
            res = _ing.ingest(r["text"], conv_id=r["conv"], msg_index=int(r["msg"]), store=store,
                              log_path=log_path, dead_letter_path=dead, **ingest_kw)
        except Exception as e:                      # stays unresolved; ingest() re-alerted if JSON
            results.append({"conv": r["conv"], "msg": r["msg"], "status": "failed",
                            "error": "%s: %s" % (type(e).__name__, e)})
            continue
        n = len(res["written"])
        results.append({"conv": r["conv"], "msg": r["msg"], "status": res["status"], "facts": n,
                        "skipped": len(res["skipped"])})
        if n > 0 or res["skipped"]:                 # facts written, or all already present
            with open(resolved_file(dead), "a", encoding="utf-8") as f:
                f.write(json.dumps({"conv": r["conv"], "msg": r["msg"], "facts": n,
                                    "ts": _ing._utc()}) + "\n")
    return results


def dead_letter_from_state_db(session_id: str, row_id: int, msg_index: int,
                              db: Optional[str] = None, dead_letter_path: Optional[str] = None
                              ) -> Dict[str, Any]:
    """Add a pre-existing lost turn to the dead-letter file, text rebuilt like sync_turn's digest."""
    import sqlite3
    db = db or os.environ.get("HERMES_STATE_DB") or str(Path.home() / ".hermes" / "state.db")
    con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    try:
        u = con.execute("select content, timestamp from messages where id=? and session_id=? "
                        "and role='user'", (row_id, session_id)).fetchone()
        if not u:
            raise SystemExit("no user message %s in session %s" % (row_id, session_id))
        a = con.execute("select content from messages where session_id=? and id>? and role='assistant' "
                        "and coalesce(content,'')!='' order by id limit 1", (session_id, row_id)).fetchone()
    finally:
        con.close()
    digest = "[session=%s t=%d]\nUSER: %s\nASSISTANT: %s" % (
        session_id, int(u[1]), (u[0] or "")[:600], ((a[0] if a else "") or "")[:600])
    row = {"conv": session_id, "msg": msg_index, "text": digest,
           "error": "JSONDecodeError: lost before dead-letter existed (failures.log 2026-11-10T04:56:29Z)",
           "ts": _ing._utc()}
    p = _ing.dead_letter_file(dead_letter_path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def _cli(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="mpmf-reingest")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--from-state-db", nargs=2, metavar=("SESSION", "ROW_ID"))
    ap.add_argument("--msg-index", type=int, default=0)
    ap.add_argument("--dead-letter", default=None)
    a = ap.parse_args(argv)
    if a.from_state_db:
        r = dead_letter_from_state_db(a.from_state_db[0], int(a.from_state_db[1]), a.msg_index,
                                      dead_letter_path=a.dead_letter)
        print("dead-lettered conv=%s msg=%s (%d chars)" % (r["conv"], r["msg"], len(r["text"])))
    if a.list:
        for r in unresolved(a.dead_letter):
            print("UNRESOLVED conv=%s msg=%s ts=%s %s" % (r["conv"], r["msg"], r["ts"], r["error"][:80]))
    if a.run:
        from . import provider as _prov
        import mpmf_plus.memory as _ppm
        _ppm.configure_store(str(_prov.PATH))
        out = reingest(_prov._open().mem, a.dead_letter, log_path=_prov.STORE_LOG,
                       model=_prov.LLM_MODEL)
        for o in out:
            print(json.dumps(o))
        return 0 if all(o["status"] != "failed" for o in out) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
