"""b0 dataset loaders: LoCoMo, DMR (MSC-Self-Instruct), LongMemEval, BEAM -> one common shape.

A *corpus* is the conversation history that gets ingested into ONE fresh store; a *question* points at
one corpus. Nothing here touches the network; the data directory is an argument.

Units: the thing ingested per call. ``pair`` benchmarks (LongMemEval, BEAM: user + assistant) follow the
production ``sync_turn`` digest (user turn + assistant turn together); ``single`` benchmarks (LoCoMo,
DMR: two humans) ingest one turn per call with the speaker name in front.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

BENCHMARKS = ("locomo", "dmr", "longmemeval", "beam")
ANCHOR = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)   # DMR has relative times only


@dataclass
class Turn:
    session: int
    speaker: str
    text: str
    date: str            # ISO 8601 UTC, "YYYY-MM-DDTHH:MM:SSZ"
    role: str = "user"   # "user" | "assistant" (pair benchmarks)


@dataclass
class Corpus:
    cid: str
    turns: List[Turn]
    pairing: str = "single"      # "single" | "pair"


@dataclass
class Question:
    qid: str
    bench: str
    category: str
    question: str
    gold: str
    cid: str
    extra: Dict[str, Any] = field(default_factory=dict)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _lit(v: Any) -> Any:
    """Fields may be real lists/dicts or python-literal / JSON strings."""
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in "[{(":
            try:
                return ast.literal_eval(s)
            except Exception:  # noqa: BLE001
                try:
                    return json.loads(s)
                except Exception:  # noqa: BLE001
                    return v
    return v


# ------------------------------------------------------------------------------------------ LoCoMo
LOCOMO_CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
_LOCOMO_DT = re.compile(r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", re.I)


def parse_locomo_date(s: str) -> datetime:
    m = _LOCOMO_DT.search(s or "")
    if not m:
        raise ValueError("unparseable LoCoMo session date: %r" % (s,))
    hh, mm, ap, dd, mon, yy = m.groups()
    h = int(hh) % 12 + (12 if ap.lower() == "pm" else 0)
    d = datetime.strptime("%s %s %s" % (dd, mon[:3].title(), yy), "%d %b %Y")
    return d.replace(hour=h, minute=int(mm), tzinfo=timezone.utc)


def load_locomo(data_dir: str, include_cat5: bool = False) -> Tuple[Dict[str, Corpus], List[Question]]:
    with open(os.path.join(data_dir, "locomo10.json")) as fh:
        data = json.load(fh)
    corpora: Dict[str, Corpus] = {}
    qs: List[Question] = []
    for conv in data:
        cid = conv["sample_id"]
        c = conv["conversation"]
        turns: List[Turn] = []
        snums = sorted(int(k.split("_")[1]) for k in c if re.fullmatch(r"session_\d+", k))
        for n in snums:
            base = parse_locomo_date(c["session_%d_date_time" % n])
            for i, t in enumerate(c["session_%d" % n]):
                text = t.get("text") or ""
                if t.get("blip_caption"):
                    text += " [shares a photo: %s]" % t["blip_caption"]     # LoCoMo's own caption field
                turns.append(Turn(n, t["speaker"], text, _iso(base + timedelta(seconds=i))))
        corpora[cid] = Corpus(cid, turns, "single")
        for i, q in enumerate(conv["qa"]):
            cat = int(q["category"])
            if cat == 5 and not include_cat5:
                continue
            gold = q.get("answer", q.get("adversarial_answer", ""))
            qs.append(Question("%s#%d" % (cid, i), "locomo", LOCOMO_CATS[cat], q["question"], str(gold), cid,
                               {"cat": cat, "evidence": q.get("evidence", [])}))
    return corpora, qs


# ------------------------------------------------------------------------------------------ DMR
def _time_back(s: str) -> timedelta:
    d = timedelta()
    for n, unit in re.findall(r"(\d+)\s*(week|day|hour|minute)s?", s or ""):
        d += timedelta(**{unit + "s": int(n)})
    return d


def load_dmr(data_dir: str) -> Tuple[Dict[str, Corpus], List[Question]]:
    corpora: Dict[str, Corpus] = {}
    qs: List[Question] = []
    with open(os.path.join(data_dir, "msc_self_instruct.jsonl")) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            meta = _lit(r.get("metadata")) or {}
            cid = str(meta.get("initial_data_id") or "")
            if not cid or cid in corpora:
                raise ValueError("DMR row without a unique metadata.initial_data_id: %r" % (cid,))
            si = _lit(r["self_instruct"]) or {}
            prev = _lit(r.get("previous_dialogs")) or []
            turns: List[Turn] = []
            for n, p in enumerate(prev, 1):
                when = ANCHOR - _time_back(p.get("time_back", ""))
                for i, t in enumerate(_lit(p["dialog"])):
                    turns.append(Turn(n, t.get("id") or ("Speaker %d" % (1 + i % 2)), t["text"], _iso(when + timedelta(seconds=i))))
            n = len(prev) + 1
            for i, t in enumerate(_lit(r["dialog"])):
                turns.append(Turn(n, t.get("id") or ("Speaker %d" % (1 + i % 2)), t["text"], _iso(ANCHOR + timedelta(seconds=i))))
            corpora[cid] = Corpus(cid, turns, "single")
            qs.append(Question(cid, "dmr", "dmr", str(si.get("B", "")), str(si.get("A", "")), cid, {}))
    return corpora, qs


# ------------------------------------------------------------------------------------------ LongMemEval
_LME_DT = re.compile(r"(\d{4})/(\d{2})/(\d{2})(?:\s*\([A-Za-z]+\))?\s*(\d{1,2}):(\d{2})")


def parse_lme_date(s: str) -> datetime:
    m = _LME_DT.search(s or "")
    if not m:
        raise ValueError("unparseable LongMemEval date: %r" % (s,))
    y, mo, d, h, mi = map(int, m.groups())
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


def load_longmemeval(path: str) -> Tuple[Dict[str, Corpus], List[Question]]:
    """One corpus per question (each question has its own haystack). ``path`` = oracle / S / M json."""
    with open(path) as fh:
        data = json.load(fh)
    corpora: Dict[str, Corpus] = {}
    qs: List[Question] = []
    for e in data:
        cid = str(e["question_id"])
        turns: List[Turn] = []
        for n, (sess, dt) in enumerate(zip(e["haystack_sessions"], e["haystack_dates"]), 1):
            base = parse_lme_date(dt)
            for i, t in enumerate(sess):
                turns.append(Turn(n, "User" if t["role"] == "user" else "Assistant", t["content"],
                                  _iso(base + timedelta(seconds=i)), t["role"]))
        corpora[cid] = Corpus(cid, turns, "pair")
        qs.append(Question(cid, "longmemeval", e["question_type"], e["question"], str(e["answer"]), cid,
                           {"question_date": e.get("question_date", ""), "abstention": "_abs" in cid}))
    return corpora, qs


# ------------------------------------------------------------------------------------------ BEAM
_BEAM_DT = re.compile(r"([A-Za-z]+)-(\d{1,2})-(\d{4})")


def parse_beam_date(s: Optional[str]) -> Optional[datetime]:
    m = _BEAM_DT.fullmatch((s or "").strip())
    if not m:
        return None
    try:
        return datetime.strptime("%s %s %s" % (m.group(2), m.group(1)[:3].title(), m.group(3)), "%d %b %Y").replace(
            hour=12, tzinfo=timezone.utc)
    except ValueError:
        return None


def load_beam(path: str) -> Tuple[Dict[str, Corpus], List[Question]]:
    import pandas as pd
    df = pd.read_parquet(path)
    corpora: Dict[str, Corpus] = {}
    qs: List[Question] = []
    for _, r in df.iterrows():
        cid = str(r["conversation_id"])
        turns: List[Turn] = []
        last = ANCHOR
        for n, batch in enumerate(r["chat"], 1):
            batch = list(batch)
            when = parse_beam_date(batch[0].get("time_anchor")) or last
            last = when
            for i, m in enumerate(batch):
                role = m["role"]
                turns.append(Turn(n, "User" if role == "user" else "Assistant", str(m["content"]),
                                  _iso(when + timedelta(seconds=i)), role))
        corpora[cid] = Corpus(cid, turns, "pair")
        pq = _lit(r["probing_questions"])
        for ability in sorted(pq):
            for j, q in enumerate(pq[ability]):
                rub = q.get("rubric") or []
                qs.append(Question("%s/%s#%d" % (cid, ability, j), "beam", ability, q["question"],
                                   str(q.get("ideal_response") or q.get("answer") or ""), cid,
                                   {"rubric": [str(x) for x in rub]}))
    return corpora, qs


def load(bench: str, data_dir: str, *, include_cat5: bool = False, lme_file: str = "longmemeval_oracle.json",
         beam_file: str = "beam_100K.parquet") -> Tuple[Dict[str, Corpus], List[Question]]:
    if bench == "locomo":
        return load_locomo(data_dir, include_cat5)
    if bench == "dmr":
        return load_dmr(data_dir)
    if bench == "longmemeval":
        return load_longmemeval(os.path.join(data_dir, lme_file))
    if bench == "beam":
        return load_beam(os.path.join(data_dir, beam_file))
    raise ValueError("unknown benchmark %r (one of %s)" % (bench, ", ".join(BENCHMARKS)))


# ------------------------------------------------------------------------------------------ sampling
def seeded_order(questions: List[Question], seed: int) -> List[Question]:
    """Seeded shuffle of the whole pool. A prefix of it is a uniform random sample."""
    ids = sorted(questions, key=lambda q: q.qid)
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids


def sample(questions: List[Question], n: Optional[int], seed: int) -> List[Question]:
    order = seeded_order(questions, seed)
    return order if not n or n >= len(order) else order[:n]


def ids_md5(qs: List[Question]) -> str:
    return hashlib.md5("\n".join(q.qid for q in qs).encode()).hexdigest()


def units(corpus: Corpus) -> List[Dict[str, Any]]:
    """The ingest units in order: {session, idx (within session), user, assistant|None, date}."""
    out: List[Dict[str, Any]] = []
    seq: Dict[int, int] = {}
    ts = corpus.turns
    if corpus.pairing == "single":
        for t in ts:
            i = seq.get(t.session, 0)
            seq[t.session] = i + 1
            out.append({"session": t.session, "idx": i, "user": "%s: %s" % (t.speaker, t.text), "assistant": None,
                        "date": t.date})
        return out
    i = 0
    while i < len(ts):
        t = ts[i]
        if t.role == "user":
            a = ts[i + 1] if i + 1 < len(ts) and ts[i + 1].role == "assistant" and ts[i + 1].session == t.session else None
            k = seq.get(t.session, 0)
            seq[t.session] = k + 1
            out.append({"session": t.session, "idx": k, "user": t.text, "assistant": a.text if a else None, "date": t.date})
            i += 2 if a else 1
        else:       # an assistant turn with no user turn before it: not a user message, nothing to extract
            i += 1
    return out
