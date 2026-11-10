"""mpmf_plus.keep_details — p9: keep the specifics when a turn is extracted.

The 67e prompt already says "keep every name, number, year", but the extractor
sometimes writes "User bought a laptop." for the evidence "I bought a Lenovo
ThinkPad X1 for 1,450 euros in March". This module is the check the prompt
cannot be:

* ``missing_specifics(evidence, sentence, fact_date)`` lists every proper noun,
  number and date in the EVIDENCE sentence that is absent from the FACT sentence.
* ``repair(facts, body, call_model, parse)`` — when any fact is missing something,
  ask the extractor ONCE more, naming exactly what was dropped. A retried fact only
  replaces its original when it is the same fact (same evidence) AND misses fewer
  specifics. Facts are never added or removed, so a retry cannot invent a memory.

Flag ``keep_details`` (recall.config.json / ``MPMF_KEEP_DETAILS``), default OFF.
Off: ``ingest`` never calls this module (tests/test_keep_details.py pins it).
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Tuple

# Words that start a sentence capitalised without being proper nouns.
_STOP = frozenset("""
a about after again all also although am an and any are as at be because been before
but by can could did do does don't during each even every first for from had has have
he her here his how i i'd i'll i'm i've if in into is it it's its just let let's like
maybe me more most my no not now of oh ok okay on one only or other our please so some
sorry sure thanks thank that the their then there these they this those though to too
user was we well were what when where which while who why will with would yeah yes you
your still since once today tonight yesterday tomorrow lately recently
""".split())

# numbers glued to letters (V60, X1, mp3) belong to a name token, not a number of their own
_NUM = re.compile(r"(?<![^\W\d_])(?<![\w])\d[\d,.:/\-]*\d(?![^\W\d_])|(?<![\w])\d(?![^\W\d_\d])")
_CAP = re.compile(r"[^\W\d_][\w&'’\-]*")
_WORD = re.compile(r"[\w&'’\-]+")


def _norm_num(tok: str) -> str:
    return tok.replace(",", "").strip(".:/-")


def _stem(tok: str) -> str:
    t = tok.lower().replace("’", "'")
    for suf in ("'s", "'"):
        if t.endswith(suf):
            t = t[: -len(suf)]
    return t


def _sentence_start(text: str, pos: int) -> bool:
    before = text[:pos].rstrip(" \"'“‘([")
    return not before or before[-1] in ".!?:;\n"


def specifics(evidence: str) -> Dict[str, List[str]]:
    """Proper nouns / numbers in an evidence sentence (order kept, de-duplicated)."""
    ev = evidence or ""
    nums, seen = [], set()
    for m in _NUM.finditer(ev):
        t = _norm_num(m.group(0))
        if t and t not in seen:
            seen.add(t)
            nums.append(t)
    letters = [c for c in ev if c.isalpha()]
    shouting = bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.7
    props, seen = [], set()
    if not shouting:
        caps = [m for m in _CAP.finditer(ev) if m.group(0)[:1].isupper()]
        inside = {_stem(m.group(0)) for m in caps if not _sentence_start(ev, m.start())}
        for m in caps:
            tok = m.group(0).rstrip("-'’")
            s = _stem(tok)
            if not s or s in _STOP or s in seen:
                continue
            # a capital that only starts a sentence is a name only if it looks like one
            # (ThinkPad, V60) or the same word is capitalised mid-sentence elsewhere
            if _sentence_start(ev, m.start()) and s not in inside and not re.search(r"[A-Z0-9]", tok[1:]):
                continue
            seen.add(s)
            props.append(tok)
    return {"proper": props, "numbers": nums}


def missing_specifics(evidence: str, sentence: str, fact_date: str = "") -> List[str]:
    """Specifics present in ``evidence`` but absent from ``sentence`` ([] = none)."""
    sp = specifics(evidence)
    sent = sentence or ""
    low = sent.lower().replace("’", "'")
    words = {_stem(w) for w in _WORD.findall(sent)}
    flat = sent.replace(",", "")
    runs = set(re.findall(r"\d+(?:\.\d+)?", flat + " " + (fact_date or "")))
    out: List[str] = []
    for tok in sp["proper"]:
        s = _stem(tok)
        if s in words:
            continue
        parts = [p for p in re.split(r"[-&]", s) if p]
        if len(parts) > 1 and all(p in words or p in low for p in parts):
            continue
        out.append(tok)
    for tok in sp["numbers"]:
        if re.search(r"[:/\-]", tok):          # date / time / range: every digit run must appear
            if all(r in runs for r in re.findall(r"\d+(?:\.\d+)?", tok)):
                continue
        elif tok in flat or tok in runs:
            continue
        out.append(tok)
    return out


def _overlap(a: str, b: str) -> float:
    wa = {w for w in re.findall(r"\w+", (a or "").lower()) if w not in _STOP}
    wb = {w for w in re.findall(r"\w+", (b or "").lower()) if w not in _STOP}
    return len(wa & wb) / len(wa) if wa else 0.0


def _key(evidence: Any) -> str:
    return re.sub(r"\W+", " ", (evidence or "").lower()).strip()


def audit(facts: List[Dict[str, Any]]) -> List[Tuple[int, List[str]]]:
    """[(index, missing)] for every fact that dropped a specific.

    Only a fact that is the SOLE fact of its evidence sentence is held to it: when several
    facts share one evidence sentence its names/numbers are legitimately split between them,
    and demanding all of them in each fact makes the extractor merge facts (p9 run 3: 15/40
    turns changed, specificity -0.26 on those, facts lost to dedup).
    """
    count: Dict[str, int] = {}
    for f in facts:
        k = _key(f.get("evidence"))
        count[k] = count.get(k, 0) + 1
    bad = []
    for i, f in enumerate(facts):
        if not f.get("evidence") or count[_key(f.get("evidence"))] > 1:
            continue
        m = missing_specifics(f.get("evidence") or "", f.get("sentence") or "",
                              f.get("fact_date") or "")
        if m:
            bad.append((i, m))
    return bad


def retry_body(body: str, facts: List[Dict[str, Any]],
               bad: List[Tuple[int, List[str]]]) -> str:
    lines = []
    for i, m in bad:
        f = facts[i]
        lines.append('- evidence "%s": your fact_sentence "%s" dropped: %s'
                     % ((f.get("evidence") or "")[:200], f.get("sentence"), ", ".join(m)))
    return (body + "\n\nREVIEW: your previous answer lost details that the evidence states.\n"
            + "\n".join(lines)
            + "\nExtract again. Keep the SAME facts, one per fact (do not merge them, add none,"
              " do not copy the whole message); add to each fact_sentence the names, numbers"
              " and dates from its evidence_sentence that belong to that fact. Same JSON shape.")


def repair(facts: List[Dict[str, Any]], body: str, call_model: Callable[[str], str],
           parse: Callable[[str], List[Dict[str, Any]]]) -> Dict[str, Any]:
    """One retry. Returns {"facts","bad","fixed","retried","error",...}; never raises."""
    info: Dict[str, Any] = {"facts": facts, "bad": 0, "fixed": 0, "retried": False,
                            "error": None, "before": [], "after": []}
    bad = audit(facts)
    info["bad"] = len(bad)
    if not bad:
        return info
    info["retried"] = True
    try:
        again = parse(call_model(retry_body(body, facts, bad)))
    except Exception as exc:  # noqa: BLE001 — a failed retry keeps the original facts
        info["error"] = "%s: %s" % (type(exc).__name__, exc)
        return info
    by_ev: Dict[str, List[Dict[str, Any]]] = {}
    for g in again:
        by_ev.setdefault(_key(g.get("evidence")), []).append(g)
    out = list(facts)
    used = set()
    for i, m in bad:
        f = facts[i]
        ek = _key(f.get("evidence"))
        best, best_k, best_gi = None, None, -1
        for gi, g in enumerate(by_ev.get(ek, [])):
            if (ek, gi) in used:
                continue
            gm = missing_specifics(g.get("evidence") or "", g.get("sentence") or "",
                                   g.get("fact_date") or "")
            ov = _overlap(f["sentence"], g["sentence"])
            # same fact (not a sibling from the same sentence), strictly fewer drops,
            # and not a merge / copy-paste of the whole message
            if len(gm) >= len(m) or ov < 0.4 or len(g["sentence"]) > 2 * len(f["sentence"]) + 120:
                continue
            k = (len(gm), -ov)
            if best_k is None or k < best_k:
                best, best_k, best_gi = g, k, gi
        if best is not None:
            used.add((ek, best_gi))
            info["before"].append(f["sentence"])
            info["after"].append(best["sentence"])
            out[i] = dict(f, sentence=best["sentence"])
            info["fixed"] += 1
    info["facts"] = out
    return info


def enabled() -> bool:
    try:
        from . import paths as _paths
        return bool(_paths.load_config().get("keep_details", False))
    except Exception:
        return False
