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
  a. MATCH (g1): embed the staged lesson's situation clause exactly the way the
     sit sidecar does (athena_memory._sit_text) and compare against the pool
     sit vectors and against other staged lessons. Same rule/question id ->
     merge at cosine >= 0.80; either side without an id -> cosine >= 0.85;
     different ids never merge. A merge appends the session to the survivor's
     _source_sessions (a set), adds the absorbed occurrences to _occurrences
     (EVERY merge counts) and retires the staged copy.
  b. PROMOTE  ->  only if vet verdict == keep AND >= 2 DISTINCT source sessions
     (occurrences never promote: one long chat cannot promote alone).
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
import re
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
# g1: the match rule. Same rule/question id -> cosine >= COS_SAME_ID; either
# side without an id -> cosine >= COS_NO_ID; different ids never merge. This
# replaces the flat 0.90 near-copy rule: 0.90 is above both thresholds, so for
# same-id and no-id pairs it is subsumed, and for different-id pairs it
# contradicts "never merge". NEAR_COPY stays only so importers do not break.
COS_SAME_ID = 0.80
COS_NO_ID = 0.85
NEAR_COPY = 0.90            # retired in g1, unused
MIN_SESSIONS = 2
MAX_MERGED_FROM = 200       # source names remembered on a survivor (idempotent merges)
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
    # r93: a re-judged lesson that is ALREADY held for the same reason needs no
    # write. Each _rw_update reopens + re-saves the whole store (~4 s at 131 MB),
    # so re-writing ~270 unchanged held nodes blew the cron's 300 s window.
    if status == "held":
        try:
            cur = mm.get_wiki(nid) or {}
            if (cur.get("_status") == "held"
                    and (cur.get("_gate_reason") or "") == (reason or "")[:300]):
                return
        except Exception:
            pass

    if not _rw_update(name, _retire_mut(status, target, reason)):
        log({"name": name, "decision": "retire_failed", "want": status})


def _retire_mut(status: str, target: str = "", reason: str = ""):
    """The wiki mutation _retire applies (shared with the atomic merge write)."""
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
    return _m


def _merge_into(mm, nid: int, session_ids: list, reason: str) -> dict:
    """Record one merge on an existing node: sessions (a set) + occurrences (+1 per merge)."""
    name = mm.get_name(nid) or ""
    out = {}
    _rw_update(name, _merge_mut(list(session_ids), reason, "", 1, "", out))
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
    # r91c: package write path (no athena_memory).
    from . import memory as _mem
    store = _mem.store
    rid = _rule_id(wiki, text)      # g1: the rule id survives promotion
    store(text, name=pool_name, namespace="cognition", auto_extract=True,
          auto_link=True, summary=(wiki.get("_summary") or text)[:300],
          wiki={
              **({"_rule_id": rid} if rid else {}),
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
# ---------------------------------------------------------------- g1: atomic writes + match rule
class _AbortTxn(Exception):
    pass


def _rw_update_many(ops) -> bool:
    """Apply [(name, mutate), ...] in ONE write txn and ONE save, then verify.

    A merge touches two nodes (the survivor gets the sessions, the source is
    retired); doing both in one save means a crash can never leave a counted
    merge with a live source (which would count the same merge twice).
    """
    for _ in range(2):
        m = A._open_rw()
        expect = {}
        try:
            with m.mem.write_txn():
                for name, mutate in ops:
                    nid = m.mem.get_node_id_by_name(name)
                    if nid is None:
                        raise _AbortTxn(name)
                    wiki = m.mem.get_wiki(nid) or {}
                    before = dict(wiki)
                    mutate(wiki)
                    m.mem.update_node_wiki(nid, wiki)
                    expect.setdefault(name, {}).update(
                        {k: str(v) for k, v in wiki.items()
                         if str(before.get(k)) != str(v)})
        except Exception:
            # raised INSIDE write_txn: the context exits without saving, so a
            # failed op can never leave a half-applied batch on disk
            A._invalidate_open_cache()
            return False
        try:
            m.save()
        except Exception:
            pass
        A._invalidate_open_cache()
        mm2 = A._open().mem
        ok = True
        for name, want in expect.items():
            nid2 = mm2.get_node_id_by_name(name)
            got = (mm2.get_wiki(nid2) or {}) if nid2 is not None else {}
            if nid2 is None or any(str(got.get(k)) != v for k, v in want.items()):
                ok = False
                break
        if ok:
            return True
    return False


_RULE_RX = re.compile(r"Rule fired:\s*([A-Za-z0-9][A-Za-z0-9_\-]*)\s*(?:[.:|,;]|$)")


def _rule_id(wiki: dict, text: str = "") -> str:
    """The rule / question id a lesson carries, or '' when it has none.

    Looked up in staged metadata (_rule_id, _question_id), then in the text
    (`Rule fired: <id>` as deterministic corrections write it). A prose rule
    name with spaces is not an id. NOT _qid: on the pool that is 'live'.
    """
    for k in ("_rule_id", "_question_id"):
        v = str((wiki or {}).get(k) or "").strip().lower()
        if v:
            return v
    m = _RULE_RX.search(text or "")
    return m.group(1).lower() if m else ""


def match_threshold(rule_a: str, rule_b: str):
    """Cosine a pair must reach to merge; None = never (different rule ids)."""
    if rule_a and rule_b:
        return COS_SAME_ID if rule_a == rule_b else None
    return COS_NO_ID


def _merged_from(wiki: dict) -> list:
    raw = (wiki or {}).get("_merged_from")
    try:
        v = json.loads(raw) if isinstance(raw, str) else (raw or [])
        return [str(x) for x in v if x] if isinstance(v, list) else []
    except Exception:
        return []


def _merge_state(sessions_a, occ_a, sessions_b, occ_b):
    """Merged (distinct sorted sessions, occurrences).

    Sessions are a SET (a session already recorded adds none); occurrences are
    a plain sum, so every merge counts: absorbing a lesson seen once adds one.
    Promotion looks at len(sessions) only, never at occurrences.
    """
    return sorted(set(sessions_a) | set(sessions_b)), int(occ_a) + int(occ_b)


def _merge_mut(session_ids, reason: str, src_name: str, src_occ: int,
               rule_id: str, out: dict):
    """Wiki mutation that records one merge on the survivor.

    Idempotent per source name: a replayed merge (crashed/retried run) finds the
    name in _merged_from and changes nothing, so it is never counted twice.
    """
    def _m(wiki):
        before = _sessions(wiki)
        done = _merged_from(wiki)
        if src_name and src_name in done:
            out.update({"already": True, "sessions_after": len(before),
                        "occurrences": max(_occurrences(wiki), 1)})
            return
        occ0 = max(_occurrences(wiki), len(before), 1)
        merged, occ = _merge_state(before, occ0, session_ids, max(int(src_occ), 1))
        wiki["_source_sessions"] = json.dumps(merged)
        if merged:
            wiki["_source_session"] = merged[0]
        wiki["_occurrences"] = str(occ)
        wiki["_last_seen"] = _now()
        if src_name:
            wiki["_merged_from"] = json.dumps((done + [src_name])[-MAX_MERGED_FROM:])
        if rule_id and not (wiki.get("_rule_id") or wiki.get("_question_id")):
            wiki["_rule_id"] = rule_id
        out.update({"sessions_before": len(before), "sessions_after": len(merged),
                    "occurrences": occ, "reason": reason})
    return _m


# ---------------------------------------------------------------- g1: collect / plan / rebuild
def _norm_vec(v):
    import numpy as np
    a = np.asarray(v, dtype="float64")
    n = float(np.linalg.norm(a))
    return a / (n or 1.0)


def _item(name, nid, kind, wiki, text, vec=None):
    sess = _sessions(wiki)
    return {"name": name, "nid": nid, "kind": kind, "vec": vec,
            "rule": _rule_id(wiki, text), "sessions": sess,
            "occ": max(_occurrences(wiki), len(sess), 1),
            "created": str(wiki.get("_created_at") or ""), "text": text,
            "wiki": wiki, "absorbed": []}


def _collect_pool(mm, pool_vecs: dict) -> list:
    """Pool lessons with a sit vector: name, rule id, recorded sessions, occurrences."""
    out = []
    for nid, vec in pool_vecs.items():
        try:
            name = mm.get_name(nid) or ""
            if not name or name.startswith(STAGE_PREFIX):
                continue
            w = mm.get_wiki(nid) or {}
            text = mm.get_text(nid) or mm.get_summary(nid) or ""
            out.append(_item(name, nid, "pool", w, text, _norm_vec(vec)))
        except Exception:
            continue
    return out


def plan_matches(staged: list, pool: list) -> dict:
    """Pure planner: which staged lessons merge into which survivor.

    staged: items (see _item) in creation order, each with a unit `vec`.
    pool:   items with a unit `vec`.
    A staged lesson merges into the BEST survivor whose pair threshold it
    reaches: same rule id -> COS_SAME_ID, either id missing -> COS_NO_ID,
    two different ids -> never. Survivors are pool lessons or EARLIER staged
    lessons that were themselves kept. Every merge adds the absorbed lesson's
    occurrences to the survivor; sessions are a set, so a session already
    recorded adds an occurrence but no session.

    Returns {"merges": [...], "survivors": [staged items kept, final state],
             "pool": {name: pool item with final state}}.
    """
    import numpy as np
    pool = [dict(p, sessions=list(p["sessions"]), absorbed=[]) for p in pool]
    pool_by = {p["name"]: p for p in pool}
    P = np.vstack([p["vec"] for p in pool]) if pool else None
    pool_rules = np.array([p["rule"] for p in pool], dtype=object) if pool else None
    kept, merges = [], []
    for s0 in staged:
        s = dict(s0, sessions=list(s0["sessions"]), absorbed=[])
        best = None                                   # (sim, kind, item, thr)
        if P is not None and s["vec"] is not None:
            sims = P @ s["vec"]
            thr = np.full(len(pool), COS_NO_ID, dtype="float64")
            if s["rule"]:
                same = pool_rules == s["rule"]
                other = (pool_rules != "") & ~same
                thr[same] = COS_SAME_ID
                thr[other] = np.inf
            ok = sims >= thr - 1e-9
            if ok.any():
                i = int(np.argmax(np.where(ok, sims, -np.inf)))
                best = (float(sims[i]), "pool", pool[i], float(thr[i]))
        for k in kept:
            if s["vec"] is None or k["vec"] is None:
                continue
            t = match_threshold(s["rule"], k["rule"])
            if t is None:
                continue
            sim = float(s["vec"] @ k["vec"])
            if sim >= t - 1e-9 and (best is None or sim > best[0]):
                best = (sim, "staged", k, t)
        if best is None:
            kept.append(s)
            continue
        sim, kind, tgt, thr = best
        before = set(tgt["sessions"])
        tgt["sessions"], tgt["occ"] = _merge_state(tgt["sessions"], tgt["occ"],
                                                   s["sessions"], s["occ"])
        if s["rule"] and not tgt["rule"]:
            tgt["rule"] = s["rule"]
        tgt["absorbed"].append(s["name"])
        merges.append({"src": s["name"], "target": tgt["name"], "kind": kind,
                       "sim": round(sim, 4), "thr": thr,
                       "rule_src": s["rule"], "rule_tgt": tgt["rule"],
                       "src_sessions": list(s["sessions"]), "src_occ": s["occ"],
                       "new_sessions": len(set(s["sessions"]) - before),
                       "tgt_sessions_after": len(tgt["sessions"]),
                       "tgt_occ_after": tgt["occ"], "src_text": s["text"][:200]})
    return {"merges": merges, "survivors": kept, "pool": pool_by}


def read_gate_log(path=None) -> dict:
    """Distinct lesson names in intake_gate.jsonl -> what the log knows of them.

    The same held lesson is logged again on every run, so everything is keyed
    by DISTINCT name: sessions (union), last decision, and the merges whose
    target is that name (distinct source names, their sessions, the highest
    occurrence count the log recorded).
    """
    path = path or GATE_LOG
    names, merged_in = {}, {}
    try:
        fh = open(path)
    except Exception:
        return {"names": names, "merged_in": merged_in}
    with fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            n = r.get("name")
            if not n or r.get("dry_run"):      # a dry run decided nothing
                continue
            e = names.setdefault(n, {"sessions": set(), "decision": None})
            e["sessions"].update(str(s) for s in (r.get("sessions") or []) if s)
            e["decision"] = r.get("decision")
            if r.get("decision") == "merged" and r.get("target"):
                m = merged_in.setdefault(r["target"], {"sources": {}, "occ": 0})
                m["sources"].setdefault(n, set()).update(
                    str(s) for s in (r.get("sessions") or []) if s)
                try:
                    m["occ"] = max(m["occ"], int(r.get("occurrences") or 0))
                except Exception:
                    pass
    return {"names": names, "merged_in": merged_in}


def rebuild_held_state(staged: list, log_path=None, log_info=None) -> dict:
    """Reconstruct each held lesson's distinct sessions and occurrence count.

    Sources, per DISTINCT lesson name: the node's own metadata
    (_source_sessions, _occurrences) and intake_gate.jsonl (the sessions logged
    for that name, and the merges logged INTO it: those sources' sessions and
    the highest occurrence count the log recorded). Sessions = union; occurrences
    = the largest of the node's count, the log's count, 1 + distinct merged-in
    sources, and the session count. Mutates the items in `staged` and returns
    {name: {"changed": bool, "sessions": [...], "occ": n, "merged_from": [...]}}.
    """
    info = log_info if log_info is not None else read_gate_log(log_path)
    out = {}
    for it in staged:
        n = it["name"]
        own = (info["names"].get(n) or {}).get("sessions", set())
        mi = info["merged_in"].get(n) or {"sources": {}, "occ": 0}
        sess = set(it["sessions"]) | set(own)
        for ss in mi["sources"].values():
            sess |= ss
        sess = sorted(sess)
        occ = max(it["occ"], int(mi["occ"]), len(sess), 1 + len(mi["sources"]), 1)
        changed = (sess != sorted(it["sessions"])) or occ != it["occ"]
        out[n] = {"changed": changed, "sessions": sess, "occ": occ,
                  "merged_from": sorted(mi["sources"]),
                  "was_sessions": len(it["sessions"]), "was_occ": it["occ"]}
        it["sessions"], it["occ"] = sess, occ
    return out


def _rebuild_mut(sessions, occ, merged_from):
    def _m(wiki):
        wiki["_source_sessions"] = json.dumps(sorted(sessions))
        if sessions:
            wiki["_source_session"] = sorted(sessions)[0]
        wiki["_occurrences"] = str(occ)
        if merged_from:
            mf = sorted(set(_merged_from(wiki)) | set(merged_from))
            wiki["_merged_from"] = json.dumps(mf[-MAX_MERGED_FROM:])
    return _m


def _plan_ops(plan: dict, reason_by_src: str = "") -> list:
    """Write ops for a whole plan: each target once (all its sources), each
    absorbed source retired as merged. One txn, one save."""
    by_target = {}
    for m in plan["merges"]:
        by_target.setdefault(m["target"], []).append(m)
    ops = []
    for tname, ms in by_target.items():
        def _mut(wiki, ms=ms):
            for m in ms:
                _merge_mut(m["src_sessions"], "merge_" + m["kind"], m["src"],
                           m["src_occ"], m["rule_src"], {})(wiki)
        ops.append((tname, _mut))
    for m in plan["merges"]:
        ops.append((m["src"], _retire_mut(
            "merged", m["target"], "near_copy_" + ("pool" if m["kind"] == "pool" else "staged"))))
    return ops


# ---------------------------------------------------------------- g1: staging writer
def _absorb_by_name(target: str, session_ids, rule_id: str = "") -> dict:
    out = {}
    if not _rw_update_many([(target, _merge_mut(list(session_ids), "restaged", "", 1,
                                                rule_id, out))]):
        return {"action": "absorb_failed", "target": target}
    out.update({"action": "absorbed", "target": target})
    return out


def stage_lesson(text: str, session: str = "", rule_id: str = "",
                 origin: str = "rumination", source: str = "",
                 auto_extract: bool = True, dry_run: bool = False) -> dict:
    """Stage one lesson as cog_stage_* in cognition_staging (never injectable).

    The one writer for pipelines that are not historical_rumination (which has
    its own, same shape). Identical text from another session, or any text the
    store's own near-dup check folds into an existing node, is NOT lost: the
    existing node absorbs the session and its occurrence count goes up by one.
    dry_run: look only (no write, no model call); returns would_stage /
    would_absorb / skipped.
    Returns {"action": staged|absorbed|skipped|failed, "name": ..., ...}.
    """
    from . import memory as _mem
    text = (text or "").strip()
    if not text:
        return {"action": "skipped", "reason": "empty_text"}
    name = STAGE_PREFIX + _sha(text)[:12]
    sess = [session] if session else []
    rid = (rule_id or "").strip().lower()
    mm = _mem._open().mem
    nid = mm.get_node_id_by_name(name)
    if nid is not None:
        w = mm.get_wiki(nid) or {}
        target = name
        if w.get("_retired_at"):
            target = w.get("_merged_into") or ""
            if not target or mm.get_node_id_by_name(target) is None:
                return {"action": "skipped", "name": name,
                        "reason": "retired_%s" % (w.get("_status") or "?")}
        if dry_run:
            return {"action": "would_absorb", "name": name, "target": target}
        return dict(_absorb_by_name(target, sess, rid), name=name)
    if dry_run:
        return {"action": "would_stage", "name": name, "rule_id": rid, "session": session}
    wiki = {"_source_session": session, "_source_sessions": json.dumps(sess),
            "_occurrences": "1", "_origin": origin, "_created_at": _now(),
            "_status": "hypothesis", "_reanalysis": "true"}
    if rid:
        wiki["_rule_id"] = rid
    if source:
        wiki["_rule_source"] = source
    got = None
    for attempt in range(3):
        try:
            got = _mem.store(text, name=name, namespace=STAGE_NS,
                             auto_extract=auto_extract, auto_link=True,
                             summary=text[:300], wiki=wiki)
        except Exception as e:
            log({"name": name, "decision": "stage_failed", "error": repr(e)[:200]})
            continue
        if isinstance(got, dict):              # the store folded it into a node
            tgt = got.get("merged_into_name") or ""
            if tgt:
                return dict(_absorb_by_name(tgt, sess, rid), name=name,
                            via="store_near_dup")
            return {"action": "skipped", "name": name, "reason": "store_dedup"}
        _mem._invalidate_open_cache()
        if _mem._open().mem.get_node_id_by_name(name) is not None:
            return {"action": "staged", "name": name, "rule_id": rid,
                    "session": session}
    return {"action": "failed", "name": name}



def recall_cognition(query: str, top_k: int = 3, mode: str = "lexical",
                     bump_counts: bool = True, query_vec=None, arm=None,
                     return_meta: bool = False, log_inject: bool = True,
                     vet: bool = False):
    """Package-side <prior-reasoning> recall (r91c).

    Reads the `cognition` namespace through memory.recall and formats the block
    historical_rumination feeds its prompt. The legacy athena_memory signature
    is accepted (bump_counts / log_inject / vet / arm / query_vec) so callers
    port without rewriting call sites; usage bumping and panel vetting are NOT
    performed here — vetting is cognition.vet_lesson's job at intake time.
    """
    from . import memory

    try:
        hits = memory.recall(query, top_n=top_k, namespaces=["cognition"],
                             use_rerank=(mode or "").lower() not in ("lexical", "none"))
    except Exception:
        hits = []
    lines = []
    for h in hits or []:
        t = (h.get("text") or "").strip().replace("\n", " ")
        if t:
            lines.append("- " + t[:280])
    block = ""
    if lines:
        block = ("<prior-reasoning>\nThe following lessons were distilled from prior "
                 "sessions. Apply them where relevant, but do not treat them as "
                 "inviolable rules.\n" + "\n".join(lines) + "\n</prior-reasoning>")
    if return_meta:
        return block, {"n": len(lines), "mode": mode, "port": "r91c"}
    return block


def _held_op(mm, nid: int, name: str, reason: str):
    """Op to mark a lesson held, or None when it is already held for that reason."""
    try:
        cur = mm.get_wiki(nid) or {}
        if cur.get("_status") == "held" and (cur.get("_gate_reason") or "") == reason[:300]:
            return None
    except Exception:
        pass
    return (name, _retire_mut("held", "", reason))


LAST_SUMMARY: dict = {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Round 48 intake gate (g1: match rule, every merge counts)")
    ap.add_argument("--store", default=None,
                    help="store path to operate on (default: athena_memory.PATH)")
    ap.add_argument("--near-copy", type=float, default=None,
                    help="ignored since g1 (see COS_SAME_ID / COS_NO_ID); accepted for old callers")
    ap.add_argument("--min-sessions", type=int, default=MIN_SESSIONS)
    ap.add_argument("--no-panel", action="store_true",
                    help="never call the vet panel; unvettered lessons are held")
    ap.add_argument("--max-vet", type=int, default=4, help="max panel calls")
    ap.add_argument("--spend-cap", type=float, default=0.50, help="USD ceiling")
    ap.add_argument("--dry-run", action="store_true",
                    help="write nothing and call no panel (sidecar verdicts are still read)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--lock-held", action="store_true",
                    help="the caller already holds state/intake.lock")
    ap.add_argument("--expire-days", type=int, default=30,
                    help="expire held single-session lessons older than N days")
    ap.add_argument("--as-of", default=None,
                    help="ISO date to judge expiry at (default now); for 'what expires by X'")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild each held lesson's sessions/occurrences from "
                         "intake_gate.jsonl + node metadata before matching")
    args = ap.parse_args(argv)
    allow_panel = not (args.no_panel or args.dry_run)

    lock_fd = None
    if not args.lock_held:
        lock_fd = _acquire_intake_lock()
        if lock_fd is None:
            print("intake_gate: skipped: locked")
            log({"decision": "skipped", "reason": "locked"})
            return 0
    try:
        return _run_gate(args, allow_panel)
    finally:
        _release_intake_lock(lock_fd)


def _run_gate(args, allow_panel: bool) -> int:
    def L(rec):
        if args.dry_run:
            rec["dry_run"] = True
        log(rec)

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
               "spend_usd": 0.0, "decisions": [], "merges": [],
               "vet_candidates": 0, "vet_budget_held": 0, "rebuilt": 0,
               "would_expire": 0, "dry_run": bool(args.dry_run)}
    now = datetime.now(timezone.utc)
    as_of = now
    if args.as_of:
        as_of = _parse_ts(args.as_of) or now

    # ---- embed every staged lesson once (empty / un-embeddable ones leave here)
    items = []
    for nid, nm, wiki in staged:
        text = mm.get_text(nid) or mm.get_summary(nid) or ""
        if not text.strip():
            if not args.dry_run:
                _retire(mm, nid, "rejected", reason="empty_text")
            summary["rejected"] += 1
            L({"name": nm, "decision": "rejected", "reason": "empty_text"})
            continue
        try:
            v = _vec(text)
        except Exception as e:
            if not args.dry_run:
                _retire(mm, nid, "held", reason="embed_failed")
            summary["held"] += 1
            L({"name": nm, "decision": "held", "reason": "embed_failed",
                 "error": repr(e)[:200]})
            continue
        items.append(_item(nm, nid, "staged", wiki, text, _norm_vec(v)))

    # ---- optional rebuild of session sets / occurrence counts
    pool_items = _collect_pool(mm, pool_vecs)
    rb = {}
    if args.rebuild:
        info = read_gate_log()
        hit = [p for p in pool_items if p["name"] in info["merged_in"]]
        rb = rebuild_held_state(items + hit, log_info=info)
        summary["rebuilt"] = sum(1 for r in rb.values() if r["changed"])

    # ---- the match rule over staged + pool (g1)
    plan = plan_matches(items, pool_items)
    merged_src = {m["src"] for m in plan["merges"]}
    survivors = plan["survivors"]
    summary["merges"] = plan["merges"]
    summary["merged"] = len(plan["merges"])

    if not args.dry_run:
        ops = []
        for it in survivors + pool_items:
            r = rb.get(it["name"])
            if r and r["changed"]:
                ops.append((it["name"], _rebuild_mut(r["sessions"], r["occ"], r["merged_from"])))
        ops.extend(_plan_ops(plan))
        if ops and not _rw_update_many(ops):
            L({"decision": "merge_batch_failed", "ops": len(ops)})
            print("intake_gate: merge batch failed, nothing counted", file=sys.stderr)
            summary["merged"] = 0
            summary["merge_failed"] = True
            survivors = [it for it in items]          # nothing was merged
            merged_src = set()
        else:
            summary["retired"] += len(plan["merges"])
    for m in plan["merges"]:
        why = "near_copy_pool" if m["kind"] == "pool" else "near_copy_staged"
        L({"name": m["src"], "decision": "merged", "sim": m["sim"], "why": why,
             "target": m["target"], "threshold": m["thr"],
             "rule_id": m["rule_src"] or m["rule_tgt"],
             "sessions": m["src_sessions"], "occurrences": m["tgt_occ_after"],
             "sessions_after": m["tgt_sessions_after"],
             "new_sessions": m["new_sessions"], "text": m["src_text"]})
        summary["decisions"].append({"name": m["src"], "decision": "merged",
                                     "sim": m["sim"], "target": m["target"],
                                     "text": m["src_text"]})

    # ---- held-expiry (48e step 4): a lesson that has been held for N days
    # without ever earning a second session is dead weight. Retire it loudly.
    # g1: judged on the post-merge session sets, so a lesson that merged away
    # or reached 2 sessions never expires.
    keep = []
    if args.expire_days > 0:
        cut = as_of - timedelta(days=args.expire_days)
        for it in survivors:
            created = _parse_ts(it["created"])
            if (str(it["wiki"].get("_status") or "") == "held"
                    and len(set(it["sessions"])) < args.min_sessions
                    and created is not None and created < cut):
                age = (as_of - created).days
                if not args.dry_run:
                    _retire(mm, it["nid"], "expired",
                            reason="held_%dd_single_session" % age)
                    summary["expired"] += 1
                else:
                    summary["would_expire"] += 1
                L({"name": it["name"], "decision": "expired", "age_days": age,
                     "sessions": it["sessions"], "dry_run": bool(args.dry_run),
                     "as_of": as_of.isoformat(), "text": it["text"][:200]})
                continue
            keep.append(it)
        survivors = keep
    summary["staged"] = len(survivors)

    # ---- (b) promotion gate on the survivors
    held_ops = []
    candidates = []
    for it in survivors:
        uniq = sorted(set(it["sessions"]))
        if len(uniq) < args.min_sessions:
            reason = "sessions=%d<%d" % (len(uniq), args.min_sessions)
            if not args.dry_run:
                op = _held_op(mm, it["nid"], it["name"], reason)
                if op:
                    held_ops.append(op)
            L({"name": it["name"], "decision": "held",
                 "reason": "single_session", "sessions": uniq,
                 "occurrences": it["occ"], "text": it["text"][:200]})
            summary["held"] += 1
            summary["decisions"].append({"name": it["name"], "decision": "held",
                                         "reason": "sessions=%d" % len(uniq),
                                         "text": it["text"][:200]})
        else:
            candidates.append(it)
    if held_ops and not _rw_update_many(held_ops):
        L({"decision": "held_batch_failed", "ops": len(held_ops)})
    summary["vet_candidates"] = len(candidates)
    # most evidence first: distinct sessions, then occurrences, then oldest
    candidates.sort(key=lambda c: (-len(set(c["sessions"])), -c["occ"], c["created"]))

    for it in candidates:
        nid, nm, text, wiki = it["nid"], it["name"], it["text"], it["wiki"]
        uniq = sorted(set(it["sessions"]))
        occ = it["occ"]
        verdict, vsrc, vcost, vdetail = vet_lesson(
            text, vet_doc, vet_idx, allow_panel, spend, args.max_vet)
        summary["spend_usd"] = round(spend["usd"], 6)

        if verdict == "reject":
            if not args.dry_run:
                _retire(mm, nid, "rejected", reason="vet_reject")
            L({"name": nm, "decision": "rejected", "reason": "vet_reject",
                 "vet_source": vsrc, "vet": vdetail, "sessions": uniq,
                 "text": text[:200]})
            summary["rejected"] += 1
            summary["decisions"].append({"name": nm, "decision": "rejected",
                                         "reason": "vet_reject",
                                         "text": text[:200]})
            continue

        if verdict != "keep":
            # out of vet budget (calls / spend cap) is not a verdict: held, never rejected
            budget = allow_panel and vsrc in ("skipped", "spend_cap")
            reason = "vet_budget" if budget else "vet_%s" % verdict
            if budget:
                summary["vet_budget_held"] += 1
            if not args.dry_run:
                _retire(mm, nid, "held", reason=reason)
            L({"name": nm, "decision": "held", "reason": reason,
                 "vet_source": vsrc, "sessions": uniq, "occurrences": occ,
                 "text": text[:200]})
            summary["held"] += 1
            summary["decisions"].append({"name": nm, "decision": "held",
                                         "reason": reason,
                                         "text": text[:200]})
            continue

        # ---- promote (>= min_sessions DISTINCT sessions, vet keep)
        pool_name = POOL_PREFIX + nm[len(STAGE_PREFIX):]
        if not args.dry_run:
            _promote(mm, nid, nm, text, wiki, uniq, occ)
            _retire(mm, nid, "promoted", target=pool_name, reason="vet_keep+2sess")
            summary["promoted"] += 1
            summary["retired"] += 1
        L({"name": nm, "decision": "promoted", "as": pool_name,
             "vet": vdetail, "vet_source": vsrc, "sessions": uniq,
             "occurrences": occ, "text": text[:200]})
        summary["decisions"].append({"name": nm, "decision": "promoted",
                                     "as": pool_name, "text": text[:200]})

    summary["spend_usd"] = round(spend["usd"], 6)
    summary["panel_calls"] = spend["panel_calls"]
    LAST_SUMMARY.clear()
    LAST_SUMMARY.update(summary)

    if args.json:
        print(json.dumps(summary, indent=1, default=str))
    else:
        print("intake_gate: staged=%(staged)d merged=%(merged)d promoted=%(promoted)d "
              "held=%(held)d rejected=%(rejected)d expired=%(expired)d "
              "retired=%(retired)d panel_calls=%(panel_calls)d "
              "spend=$%(spend_usd).4f" % summary)
        for d in summary["decisions"][:10]:
            print("  - %-9s %s %s" % (d["decision"], d["name"],
                                      d.get("reason") or ("-> " + str(d.get("target") or d.get("as") or ""))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
