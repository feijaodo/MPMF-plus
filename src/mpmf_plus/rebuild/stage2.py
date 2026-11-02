"""Stage 2: kind/mode classification, hub states, categories.

Rules carried over from the r69/r74 rounds:

* ``kind`` is decided by an ANCHORED verb rule first (subject-verb at the start of
  the sentence, then anywhere), and only then by the model. Matching "are" inside
  "User asks whether motors are better" must not make a fact a ``have``.
* ``mode`` is ``actual`` / ``temporary`` (rented, borrowed, trial) / ``hypothetical``
  (if, would, suppose, imagine, a character's world).
* Only an **ACTUAL change/have** fact can set the current state, and the NEWEST
  such fact wins. Intent and interest never override it; older intents are
  superseded; hypothetical facts never render as possessions.
* A state node and its ``IS_A`` edge are created in the SAME transaction - an
  orphan state is invisible to categories and is the bug that cost round 69f.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple

HAVE_RULES = [
    (re.compile(r"^(the )?(user|i)\s+(has|have|owns|holds)\b", re.I), "have"),
    (re.compile(r"^(the )?user'?s\b", re.I), "have"),
    (re.compile(r"\b(owns|leased|leases|drives|bought|owns a)\b", re.I), "have"),
]
CHANGE_RULES = [
    (re.compile(r"^(the )?(user|i)\s+(moved|switched|changed|started|stopped|got|left|quit)\b", re.I), "change"),
    (re.compile(r"\b(since|as of|no longer|used to|previously)\b", re.I), "change"),
]
INTENT_RULES = [
    (re.compile(r"^(the )?(user|i)\s+(wants?|plans?|intends?|hopes?|is going to|will)\b", re.I), "intent"),
]
INTEREST_RULES = [
    (re.compile(r"^(the )?(user|i)\s+(likes?|loves?|enjoys?|is interested in|prefers?)\b", re.I), "interest"),
]
HYPO_RULES = re.compile(r"\b(if i|if the user|would|could have|suppose|imagine|hypothetical|"
                        r"in (his|her|their|my) (story|novel|book|fiction)|character)\b", re.I)
TEMP_RULES = re.compile(r"\b(rented|renting|borrowed|trial|temporarily|for now|on loan)\b", re.I)


def classify_text(text: str) -> Dict[str, str]:
    """kind + mode for one fact sentence (deterministic, no model call)."""
    t = (text or "").strip()
    kind = "have" if HAVE_RULES[0][0].search(t) or HAVE_RULES[1][0].search(t) else None
    if kind is None:
        for rx, k in CHANGE_RULES + INTENT_RULES + INTEREST_RULES + HAVE_RULES[2:]:
            if rx.search(t):
                kind = k
                break
    kind = kind or "interest"
    if HYPO_RULES.search(t):
        mode = "hypothetical"
    elif TEMP_RULES.search(t):
        mode = "temporary"
    else:
        mode = "actual"
    return {"kind": kind, "mode": mode}


def classify_facts(facts: Iterable[Dict[str, Any]], workers: int = 8) -> List[Dict[str, Any]]:
    def one(f):
        out = dict(f)
        out.update(classify_text(f.get("sentence") or f.get("text") or ""))
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, facts))


def current_state(facts: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Newest ACTUAL change/have fact, or None. Ordering = (fact_date, msg_index)."""
    cands = [f for f in facts
             if f.get("kind") in ("change", "have") and f.get("mode") == "actual"]
    if not cands:
        return None

    def key(f):
        return (str(f.get("fact_date") or f.get("_said_at") or ""), int(f.get("msg_index") or 0))

    return max(cands, key=key)


def build_states(mem, facts: List[Dict[str, Any]], create_node, link,
                 workers: int = 8) -> Dict[str, int]:
    """One hub state per entity that owns an ACTUAL change/have fact.

    ``link`` is called INSIDE the caller's ``write_txn`` so the state and its
    IS_A edge are committed together.
    """
    by_entity: Dict[str, List[Dict[str, Any]]] = {}
    for f in facts:
        by_entity.setdefault((f.get("entity") or "other").lower(), []).append(f)

    def build(item):
        entity, rows = item
        cur = current_state(rows)
        if cur is None:
            return 0
        state = create_node(
            name=f"S_{entity}",
            text=f"As of {cur.get('fact_date') or cur.get('_said_at')}: {cur['sentence']}",
            summary=cur["sentence"][:120],
            wiki={"kind": "state", "entity": entity, "_state_mode": "actual",
                  "_from_fact": cur.get("name") or "", "_msg_index": cur.get("msg_index")},
        )
        # IS_A in the same txn as the state node (r69f orphan-state lesson).
        link(state, entity, "IS_A")
        for r in rows:
            if r.get("name"):
                link(r["name"], state, "DERIVED_FROM")
        return 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        built = sum(ex.map(build, list(by_entity.items())))
    return {"entities": len(by_entity), "states": built}
