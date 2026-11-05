#!/usr/bin/env python3
"""intake_gate.py — Round 48 intake gate for staged rumination lessons.

Pipeline position:
    historical_rumination.py  ->  cog_stage_* in ns "cognition_staging"  (NOT injectable)
                                          |
                                     intake_gate.py
                                          |
                    merged into an existing pool lesson  |  promoted to cog_rum_*
                                          |
                              injectable pool (cognition)

Rules (Round 48 spec):
  a. NEAR-COPY: embed the staged lesson's situation clause exactly the way the
     sit sidecar does (athena_memory._sit_text) and compare against the pool
     sit vectors and against other staged lessons. cosine >= 0.90  ->  merge:
     append the session id to the existing node's _source_sessions, bump
     _occurrences, retire the staged copy.
  b. PROMOTE  ->  only if vet verdict == keep AND >= 2 distinct source sessions.
  c. Every decision is appended to $MPMF_HOME/health/intake_gate.jsonl.

Deviation, recorded deliberately: the MPMF storage layer exposes no node-delete
and no rename API (MemoryManager public surface: create_node/create_edge/get_*/
set_*/update_node_wiki — see Round 47 probe). "Delete the staged copy" is
therefore implemented as RETIRE-IN-PLACE: _status is set to merged/rejected,
_merged_into points at the survivor, and the node keeps a cog_stage_ name, which
_is_cognition_node() and _sit_pool_ids() both reject. Promotion is likewise a
COPY + RETIRE, not a rename.

READ-ONLY on the live store unless --store points at a copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from . import paths as _paths


from . import provider as A  # package provider replaces the legacy wrapper

GATE_LOG = _paths.log_path("intake_gate.jsonl")
VET_PATH = _paths.home() / "state" / "lesson_vet.json"
LOCK_PATH = _paths.home() / "state" / "intake.lock"
STAGE_PREFIX = "cog_stage_"
STAGE_NS = "cognition_staging"
POOL_PREFIX = "cog_rum_"
NEAR_COPY = 0.90
MIN_SESSIONS = 2
PANEL_MODELS = ["anthropic/claude-sonnet-5.5", "openai/gpt-6.1-sol",
                "google/gemini-4-argon"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ---------------------------------------------------------------- logging
def _now():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- lock
def _acquire_intake_lock(nonblocking: bool = True):
    """Shared writer/gate lock (flock on state/intake.lock).

    Returns the fd, or None when the lock is already held (or cannot be taken).
    The writer holds this for its whole run and passes --lock-held to the gate,
    so the two can never interleave.
    """
    try:
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o644)
    except Exception as e:
        print(f"[gate] lock open failed: {e!r}", file=sys.stderr)
        return None
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        return fd
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        return None


def _release_intake_lock(fd) -> None:
    if fd is None:
        return
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        os.close(fd)
    except Exception:
        pass


def log(rec: dict) -> None:
    rec.setdefault("ts", _now())
    try:
        GATE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(GATE_LOG, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception as e:
        print(f"[gate] log failed: {e!r}", file=sys.stderr)


# ---------------------------------------------------------------- helpers
def _sha(text: str) -> str:
    return hashlib.sha1((text or "").encode()).hexdigest()


def _sessions(wiki: dict) -> list:
    raw = wiki.get("_source_sessions")
    out = []
    if raw:
        try:
            v = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(v, list):
                out = [str(x) for x in v if x]
        except Exception:
            out = [s.strip() for s in str(raw).strip("[]").split(",") if s.strip()]
    one = wiki.get("_source_session")
    if one:
        out.append(str(one))
    return sorted({s for s in out if s})


def _parse_ts(raw):
    """Parse a stored ISO timestamp; None when absent or unparseable."""
    if not raw:
        return None
    try:
        d = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _occurrences(wiki: dict) -> int:
    try:
        return int(wiki.get("_occurrences", 0) or 0)
    except Exception:
        return 0


def _clause(text: str) -> str:
    """Same clause extraction the sit sidecar uses (athena_memory._sit_text)."""
    return A._sit_text(text)


def _vec(text: str):
    return A._embed_floats_local(_clause(text), cap=None)


def _cos(a, b) -> float:
    import numpy as np
    x = np.asarray(a, dtype="float32")
    y = np.asarray(b, dtype="float32")
    nx = float(np.linalg.norm(x)) or 1.0
    ny = float(np.linalg.norm(y)) or 1.0
    return float(x.dot(y) / (nx * ny))


def _rw_update(name: str, mutate) -> bool:
    """Read-modify-write one node by NAME through a fresh writer, with verify.

    The shared read snapshot must never be mutated (writers use a private
    MPMFMemory), and back-to-back saves can lose the last write, so every
    mutation is verified from a fresh load and retried once.
    """
    for _ in range(2):
        m = A._open_rw()
        nid = m.mem.get_node_id_by_name(name)
        if nid is None:
            A._invalidate_open_cache()
            return False
        wiki = m.mem.get_wiki(nid) or {}
        try:
            mutate(wiki)
        except Exception:
            return False
        with m.mem.write_txn():      # Round 78: r69 guard is enforced now
            m.mem.update_node_wiki(nid, wiki)
        try:
            m.save()
        except Exception:
            pass
        A._invalidate_open_cache()
        mm2 = A._open().mem
        nid2 = mm2.get_node_id_by_name(name)
        if nid2 is not None and (mm2.get_wiki(nid2) or {}).get("_status") == wiki.get("_status"):
            return True
    return False


def _retire(mm, nid: int, status: str, target: str = "", reason: str = "") -> None:
    """Retire a staged node in place (no delete API exists).

    'held' is NOT terminal: a held lesson must stay visible so a later batch can
    add a second session (promotion) or so expiry can reach it. Only merged /
    rejected / promoted / expired set _retired_at, which the scan skips.
    """
    name = mm.get_name(nid) or ""
    terminal = status in ("merged", "rejected", "promoted", "expired")

    def _m(wiki):
        wiki["_status"] = status
        if terminal:
            wiki["_retired_at"] = _now()
        else:
            wiki["_held_at"] = _now()
        if target:
            wiki["_merged_into"] = target
        if reason:
            wiki["_gate_reason"] = reason[:300]

    if not _rw_update(name, _m):
        log({"name": name, "decision": "retire_failed", "want": status})


def _merge_into(mm, nid: int, session_ids: list, reason: str) -> dict:
    """Append sessions to an existing node and bump _occurrences."""
    name = mm.get_name(nid) or ""
    out = {}

    def _m(wiki):
        before = _sessions(wiki)
        merged = sorted(set(before) | set(session_ids))
        occ = max(_occurrences(wiki), 1)
        wiki["_source_sessions"] = json.dumps(merged)
        if merged:
            wiki["_source_session"] = merged[0]
        wiki["_occurrences"] = str(occ + len(set(merged) - set(before)))
        wiki["_last_seen"] = _now()
        out.update({"sessions_before": len(before), "sessions_after": len(merged),
                    "occurrences": int(wiki["_occurrences"]), "reason": reason})

    _rw_update(name, _m)
    return out


# ---------------------------------------------------------------- vet panel
def _load_vet() -> dict:
    try:
        return json.loads(VET_PATH.read_text())
    except Exception:
        return {"schema_version": 1, "generated_at": _now(),
                "gate": {}, "entries": {}}


def _vet_index(doc: dict) -> dict:
    idx = {}
    for name, e in (doc.get("entries") or {}).items():
        if isinstance(e, dict) and e.get("sha1"):
            idx[e["sha1"]] = e
    return idx


def _vet_write(doc: dict) -> None:
    tmp = str(VET_PATH) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    os.replace(tmp, VET_PATH)


def _panel_prompt() -> str:
    try:
        import vet40  # gate_eval/vet40.py — the exact judge the sidecar used
        return vet40.SYS
    except Exception:
        return ("Decide whether this lesson is a KEEP or REJECT for reuse as a "
                "prior-reasoning lesson. Answer with exactly one word: KEEP or REJECT.")


def _call_panel(model: str, sys_prompt: str, text: str, key: str,
                timeout=90) -> tuple:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": text[:2000]}],
        "temperature": 0, "max_tokens": 8,
    }).encode()
    req = urllib.request.Request(OPENROUTER_URL, data=body, headers={
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
    })
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    out = (d.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    usage = d.get("usage") or {}
    low = out.strip().lower()
    keep = True if "keep" in low and "reject" not in low else (
        False if "reject" in low else None)
    return keep, usage, round(time.time() - t0, 2), out.strip()[:40]


def _est_cost(usage: dict) -> float:
    """Deliberately conservative ceiling (assumes ~$3/Mtok in, ~$15/Mtok out)."""
    p = float(usage.get("prompt_tokens") or 0)
    c = float(usage.get("completion_tokens") or 0)
    return (p * 3.0 + c * 15.0) / 1_000_000.0


def vet_lesson(text: str, doc: dict, idx: dict, allow_panel: bool,
               spend: dict, max_vet: int) -> tuple:
    """Return (verdict, source, cost_usd, detail). verdict in keep/reject/unvetted."""
    sha = _sha(text)
    e = idx.get(sha)
    if e:
        return (e.get("v") or "unvetted"), "sidecar", 0.0, e

    if not allow_panel or spend["panel_calls"] >= max_vet:
        return "unvetted", "skipped", 0.0, {}
    if spend["usd"] >= spend["cap"]:
        return "unvetted", "spend_cap", 0.0, {}

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        return "unvetted", "no_api_key", 0.0, {}

    sys_prompt = _panel_prompt()
    votes, cost = [], 0.0
    detail = {"votes": [], "models": []}
    for model in PANEL_MODELS:
        try:
            keep, usage, secs, raw = _call_panel(model, sys_prompt, text, key)
            c = _est_cost(usage)
            cost += c
            spend["usd"] += c
            spend["panel_calls"] += 1
            detail["votes"].append(keep)
            detail["models"].append({"model": model, "keep": keep,
                                     "secs": secs, "raw": raw,
                                     "usd_est": round(c, 6)})
            if keep is not None:
                votes.append(keep)
        except Exception as ex:
            detail["models"].append({"model": model, "error": repr(ex)[:200]})
        if spend["usd"] >= spend["cap"]:
            break

    if not votes:
        return "unvetted", "panel_error", cost, detail
    keep = sum(1 for v in votes if v) * 2 > len(votes)
    verdict = "keep" if keep else "reject"
    doc.setdefault("entries", {})[None] = None
    doc["entries"].pop(None, None)
    doc["entries"][sha] = {
        "sha1": sha, "v": verdict, "audited": False,
        "panel": {"yes": sum(1 for v in votes if v), "n": len(votes)},
        "cheap": [], "form": "principle", "round": 48,
        "text": text[:300], "ts": _now(),
    }
    idx[sha] = doc["entries"][sha]
    try:
        doc["generated_at"] = _now()
        _vet_write(doc)
    except Exception as ex:
        detail["vet_write_error"] = repr(ex)[:200]
    return verdict, "panel", cost, detail


# ---------------------------------------------------------------- promotion
def _promote(mm, nid, name, text, wiki, sessions, occ) -> str:
    pool_name = POOL_PREFIX + name[len(STAGE_PREFIX):]
    from athena_memory import store
    store(text, name=pool_name, namespace="cognition", auto_extract=True,
          auto_link=True, summary=(wiki.get("_summary") or text)[:300],
          wiki={
              "_status": "hypothesis",
              "_origin": wiki.get("_origin") or "historical_rumination",
              "_created_at": wiki.get("_created_at") or _now(),
              "_promoted_at": _now(),
              "_source_session": sessions[0] if sessions else "",
              "_source_sessions": json.dumps(sorted(set(sessions))),
              "_occurrences": str(occ),
              "_intake_gate": "round48",
          })
    return pool_name


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Round 48 intake gate")
    ap.add_argument("--store", default=None,
                    help="store path to operate on (default: athena_memory.PATH)")
    ap.add_argument("--near-copy", type=float, default=NEAR_COPY)
    ap.add_argument("--min-sessions", type=int, default=MIN_SESSIONS)
    ap.add_argument("--no-panel", action="store_true",
                    help="never call the vet panel; unvettered lessons are held")
    ap.add_argument("--max-vet", type=int, default=4, help="max panel calls")
    ap.add_argument("--spend-cap", type=float, default=0.50, help="USD ceiling")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--lock-held", action="store_true",
                    help="the caller already holds state/intake.lock")
    ap.add_argument("--expire-days", type=int, default=30,
                    help="expire held single-session lessons older than N days")
    args = ap.parse_args(argv)

    lock_fd = None
    if not args.lock_held:
        lock_fd = _acquire_intake_lock()
        if lock_fd is None:
            print("intake_gate: skipped: locked")
            log({"decision": "skipped", "reason": "locked"})
            return 0

    if args.store:
        A.PATH = str(args.store)
        A._invalidate_open_cache()

    mm = A._open().mem
    ids = A._sit_vectors()[0]
    mat = A._sit_vectors()[1]
    pool_vecs = {}
    if ids and mat is not None:
        for i, nid in enumerate(ids):
            pool_vecs[int(nid)] = mat[i]

    vet_doc = _load_vet()
    vet_idx = _vet_index(vet_doc)
    spend = {"usd": 0.0, "panel_calls": 0, "cap": float(args.spend_cap)}

    staged = []
    n = mm.header.node_count
    # NB: node ids are 1-BASED — the highest id equals header.node_count, so
    # range(node_count) silently skips the newest node (the one just written).
    id_iter = mm.all_node_ids() if hasattr(mm, "all_node_ids") else range(1, n + 1)
    for nid in id_iter:
        nm = mm.get_name(nid) or ""
        if not nm.startswith(STAGE_PREFIX):
            continue
        w = mm.get_wiki(nid) or {}
        if w.get("_retired_at"):
            continue
        staged.append((nid, nm, w))
    staged.sort(key=lambda x: str(x[2].get("_created_at") or ""))

    summary = {"staged": len(staged), "merged": 0, "promoted": 0,
               "held": 0, "rejected": 0, "retired": 0, "expired": 0,
               "spend_usd": 0.0, "decisions": []}

    # ---- held-expiry (48e step 4): a lesson that has been held for N days
    # without ever earning a second session is dead weight. Retire it loudly.
    if args.expire_days > 0:
        cut = datetime.now(timezone.utc) - timedelta(days=args.expire_days)
        keep = []
        for nid, nm, wiki in staged:
            created = _parse_ts(wiki.get("_created_at"))
            if (str(wiki.get("_status") or "") == "held"
                    and len(set(_sessions(wiki))) < args.min_sessions
                    and created is not None and created < cut):
                age = (datetime.now(timezone.utc) - created).days
                if not args.dry_run:
                    _retire(mm, nid, "expired",
                            reason="held_%dd_single_session" % age)
                summary["expired"] += 1
                log({"name": nm, "decision": "expired", "age_days": age,
                     "sessions": _sessions(wiki),
                     "text": (mm.get_text(nid) or "")[:200]})
                continue
            keep.append((nid, nm, wiki))
        staged = keep
        summary["staged"] = len(staged)

    # staged-vs-staged dedupe: track kept staged nodes by vector
    kept_staged = []

    for nid, nm, wiki in staged:
        text = mm.get_text(nid) or mm.get_summary(nid) or ""
        if not text.strip():
            _retire(mm, nid, "rejected", reason="empty_text")
            summary["rejected"] += 1
            log({"name": nm, "decision": "rejected", "reason": "empty_text"})
            continue
        try:
            v = _vec(text)
        except Exception as e:
            _retire(mm, nid, "held", reason="embed_failed")
            summary["held"] += 1
            log({"name": nm, "decision": "held", "reason": "embed_failed",
                 "error": repr(e)[:200]})
            continue

        sessions = _sessions(wiki)
        occ = _occurrences(wiki) or 1

        # ---- (a) near-copy against the pool
        best_pool, best_sim = None, 0.0
        for pid, pv in pool_vecs.items():
            try:
                s = _cos(v, pv)
            except Exception:
                continue
            if s > best_sim:
                best_pool, best_sim = pid, s
        # ---- (a2) near-copy against another staged lesson kept this run
        best_stage, best_stage_sim = None, 0.0
        for sid, sv, sname in kept_staged:
            try:
                s = _cos(v, sv)
            except Exception:
                continue
            if s > best_stage_sim:
                best_stage, best_stage_sim = sid, s

        target, sim, why = None, 0.0, ""
        if best_sim >= args.near_copy:
            target, sim, why = best_pool, best_sim, "near_copy_pool"
        elif best_stage_sim >= args.near_copy:
            target, sim, why = best_stage, best_stage_sim, "near_copy_staged"

        if target is None:
            kept_staged.append((nid, v, nm))

        if target is not None:
            info = {"sim": round(sim, 4), "why": why,
                    "target": mm.get_name(target) or str(target)}
            if not args.dry_run:
                info.update(_merge_into(mm, target, sessions, why))
                _retire(mm, nid, "merged", target=info["target"], reason=why)
                summary["merged"] += 1
                summary["retired"] += 1
            log({"name": nm, "decision": "merged", **info,
                 "sessions": sessions, "text": text[:200]})
            summary["decisions"].append({"name": nm, "decision": "merged",
                                         "sim": round(sim, 4),
                                         "target": info["target"],
                                         "text": text[:200]})
            continue

        # ---- (b) promotion gate
        uniq = sorted(set(sessions))
        if len(uniq) < args.min_sessions:
            if not args.dry_run:
                _retire(mm, nid, "held",
                        reason="sessions=%d<%d" % (len(uniq), args.min_sessions))
            log({"name": nm, "decision": "held",
                 "reason": "single_session", "sessions": uniq,
                 "text": text[:200]})
            summary["held"] += 1
            summary["decisions"].append({"name": nm, "decision": "held",
                                         "reason": "sessions=%d" % len(uniq),
                                         "text": text[:200]})
            continue

        verdict, vsrc, vcost, vdetail = vet_lesson(
            text, vet_doc, vet_idx, not args.no_panel, spend, args.max_vet)
        summary["spend_usd"] = round(spend["usd"], 6)

        if verdict == "reject":
            if not args.dry_run:
                _retire(mm, nid, "rejected", reason="vet_reject")
            log({"name": nm, "decision": "rejected", "reason": "vet_reject",
                 "vet_source": vsrc, "vet": vdetail, "sessions": uniq,
                 "text": text[:200]})
            summary["rejected"] += 1
            summary["decisions"].append({"name": nm, "decision": "rejected",
                                         "reason": "vet_reject",
                                         "text": text[:200]})
            continue

        if verdict != "keep":
            if not args.dry_run:
                _retire(mm, nid, "held", reason="vet_%s" % verdict)
            log({"name": nm, "decision": "held", "reason": "vet_%s" % verdict,
                 "vet_source": vsrc, "sessions": uniq, "text": text[:200]})
            summary["held"] += 1
            summary["decisions"].append({"name": nm, "decision": "held",
                                         "reason": "vet_%s" % verdict,
                                         "text": text[:200]})
            continue

        # ---- promote
        pool_name = POOL_PREFIX + nm[len(STAGE_PREFIX):]
        if not args.dry_run:
            _promote(mm, nid, nm, text, wiki, uniq, occ)
            _retire(mm, nid, "promoted", target=pool_name, reason="vet_keep+2sess")
            summary["promoted"] += 1
            summary["retired"] += 1
        log({"name": nm, "decision": "promoted", "as": pool_name,
             "vet": vdetail, "vet_source": vsrc, "sessions": uniq,
             "occurrences": occ, "text": text[:200]})
        summary["decisions"].append({"name": nm, "decision": "promoted",
                                     "as": pool_name, "text": text[:200]})

    summary["spend_usd"] = round(spend["usd"], 6)
    summary["panel_calls"] = spend["panel_calls"]

    if args.json:
        print(json.dumps(summary, indent=1, default=str))
    else:
        print("intake_gate: staged=%(staged)d merged=%(merged)d promoted=%(promoted)d "
              "held=%(held)d rejected=%(rejected)d expired=%(expired)d "
              "retired=%(retired)d panel_calls=%(panel_calls)d "
              "spend=$%(spend_usd).4f" % summary)
        for d in summary["decisions"][:10]:
            print("  - %(decision)-9s %(name)s %(reason)s" % d
                  if "reason" in d else
                  "  - %(decision)-9s %(name)s -> %(target)s" % d)
    _release_intake_lock(lock_fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
