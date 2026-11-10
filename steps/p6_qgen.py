#!/usr/bin/env python3
"""P6: ~300 new sealed questions written from RAW user messages only.

The writer model sees one message (or up to 5 dated messages of one conversation)
plus the conversation title. No .mpmf store is opened or imported here; the only
package import is the LLM client + cost counter (scorecard.llm / COST).

Raw sources: ChatGPT export, Grok export (both read from /root/historica zips),
Hermes session db (real telegram/cli turns only, read-only).

Types: history, then, order, now, recent.  Split TUNE/HOLDOUT/LOCKED = 40/30/30 %,
stratified by type, assigned by CONVERSATION (no conversation, message or question
text appears in two splits). Messages/conversations already used by earlier sets
are excluded.

usage: p6_qgen.py <outdir>      env: MPMF_SRC (package source, for the LLM client)
Question files live OUTSIDE the repo (private chat content); the repo gets only the manifest.
"""
import glob, hashlib, json, os, random, re, sqlite3, sys, time, zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.environ["MPMF_SRC"])
from mpmf_plus.eval import scorecard as sc   # noqa: E402  (LLM client + COST only)

OUT = sys.argv[1]
SEED = 9606
WRITER = "deepseek/deepseek-v4.1-flash"
STATE_DB = os.path.expanduser("~/.hermes/state.db")
RECENT_FROM = "2026-10-01"
MIN_LEN, MAX_LEN, CLIP = 40, 1500, 500
TARGET = {"history": 120, "then": 70, "order": 60, "now": 50, "recent": 60}   # caps per type
OFFER = {"history": 330, "then": 160, "order": 110, "now": 200, "recent": 150}
SPLITS = {"TUNE": 0.4, "HOLDOUT": 0.3, "LOCKED": 0.3}
BUDGET_USD = 3.0
DEADLINE_S = 1500

SECRET = [re.compile(p) for p in (
    r"sk-or-v1-[A-Za-z0-9]{20,}", r"sk-[A-Za-z0-9_\-]{32,}", r"gh[pousr]_[A-Za-z0-9]{30,}", r"AKIA[0-9A-Z]{16}",
    r"xox[abp]-[A-Za-z0-9-]{10,}", r"\b\d{8,10}:[A-Za-z0-9_\-]{35}\b", r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"(?i)(api[_-]?key|secret|token|password|passwd)\s*[=:]\s*['\"]?[A-Za-z0-9_\-\.]{12,}",
    r"(?i)bearer\s+[A-Za-z0-9_\-\.]{24,}", r"hf_[A-Za-z0-9]{30,}",
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
    r"(?<!\d)\+?\d{1,2}[ .\-]?\(?\d{3}\)?[ .\-]\d{3}[ .\-]\d{4}(?!\d)")]
SYSTEM_PREFIX = ("[System", "[ASYNC", "[IMPORTANT", "[CONTEXT", "You've reached the maximum", "[Note:", "[SYSTEM")
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
day = lambda t: time.strftime("%Y-%m-%d", time.gmtime(t))


def clean_ok(t):
    return not any(p.search(t) for p in SECRET) and not t.startswith(SYSTEM_PREFIX)


def chatgpt_convs():
    out = []
    for p in sorted(glob.glob("/root/historica/*.zip")):
        if "3505c57f" in p:
            continue
        with zipfile.ZipFile(p) as z:
            for n in sorted(x for x in z.namelist() if re.fullmatch(r"conversations-\d+\.json", x)):
                for conv in json.load(z.open(n)):
                    if not (isinstance(conv, dict) and "mapping" in conv):
                        continue
                    msgs = []
                    for node in (conv.get("mapping") or {}).values():
                        m = (node or {}).get("message") or {}
                        if ((m.get("author") or {}).get("role")) != "user":
                            continue
                        parts = (m.get("content") or {}).get("parts") or []
                        text = " ".join(x for x in parts if isinstance(x, str)).strip()
                        if text and m.get("create_time"):
                            msgs.append((float(m["create_time"]), text))
                    out.append(("chatgpt", "cg:" + str(conv.get("conversation_id") or conv.get("id")),
                                str(conv.get("title") or "untitled")[:80], sorted(msgs)))
    return out


def grok_convs():
    out = []
    for p in sorted(glob.glob("/root/historica/3505c57f*.zip")):
        with zipfile.ZipFile(p) as z:
            n = [x for x in z.namelist() if x.endswith("prod-grok-backend.json")]
            if not n:
                continue
            d = json.load(z.open(n[0]))
            for c in d.get("conversations", []):
                info = c.get("conversation") or {}
                msgs = []
                for r in c.get("responses", []):
                    r = r.get("response") or {}
                    if r.get("sender") != "human" or not r.get("message"):
                        continue
                    ct = r.get("create_time")
                    try:
                        ms = int(ct["$date"]["$numberLong"]) if isinstance(ct, dict) else None
                    except Exception:
                        ms = None
                    if ms:
                        msgs.append((ms / 1000.0, str(r["message"]).strip()))
                out.append(("grok", "gk:" + str(info.get("id")), str(info.get("title") or "untitled")[:80], sorted(msgs)))
    return out


def session_convs():
    con = sqlite3.connect("file:%s?mode=ro" % STATE_DB, uri=True)
    by = {}
    q = ("select m.session_id, m.timestamp, m.content, s.title from messages m join sessions s on s.id = m.session_id "
         "where m.role = 'user' and coalesce(s.source, '') in ('telegram', 'cli') and m.content is not null order by m.timestamp")
    for sid, ts, content, title in con.execute(q):
        text = str(content).strip()
        if not text or ts is None or text.startswith(SYSTEM_PREFIX):
            continue
        e = by.setdefault(sid, {"title": str(title or "chat session")[:80], "seen": set(), "msgs": []})
        if text in e["seen"]:
            continue
        e["seen"].add(text)
        e["msgs"].append((float(ts), text))
    con.close()
    return [("session", "hs:" + sid, e["title"], sorted(e["msgs"])) for sid, e in sorted(by.items())]


def prior_exclusions():
    """message-text prefixes and conversation ids already used by earlier sets"""
    texts, convs, qtexts = set(), set(), set()
    for p in ("/root/r85b/questions.json", "/root/r70/questions.json"):
        for x in json.load(open(p)):
            for k in ("text", "msg"):
                if x.get(k):
                    texts.add(norm(str(x[k]))[:100])
            if x.get("conv"):
                c = str(x["conv"])
                convs.update(("cg:" + c, "gk:" + c, "hs:" + c.replace("live:", "")))
            qtexts.add(norm(x.get("q", "")))
    for p in ("/root/r88/questions_order.json", "/root/r92_heldout/questions_order_heldout.json"):
        for x in json.load(open(p)):
            qtexts.add(norm(x.get("q", "")))
            if x.get("conv"):
                convs.update(("cg:" + x["conv"], "hs:" + x["conv"]))
    return texts, convs, qtexts


P_SINGLE = """One message the user sent on {date}. Write ONE question his future memory assistant
should answer from this message ALONE, plus its answer key (<= 25 words).
Prefer durable personal facts (possessions, roles, places, decisions, numbers), not small talk.
If the message cannot support a self-contained question about the user, return {{"drop":true}}.
Return JSON {{"q":"...","key":"..."}} or {{"drop":true}}

MESSAGE:
{msg}"""
P_THEN = """One message the user sent on {date}. Write ONE date-scoped question about the user that this
message alone answers, phrased with the year (e.g. "In {year}, what/where/who ...?").
Answer key <= 25 words. Prefer durable personal facts, not small talk.
If the message cannot support a self-contained question about the user, return {{"drop":true}}.
Return JSON {{"q":"...","key":"..."}} or {{"drop":true}}

MESSAGE:
{msg}"""
P_ORDER = """You write questions for a memory benchmark. Below are dated messages a user wrote in one conversation titled "{e}".
Write ONE question that asks about the CHRONOLOGICAL ORDER of these events (e.g. "In what order did ...").
Rules: answerable from these messages alone; do NOT mention dates; do not mention memory or notes.
Refer to the writer of the messages as "the user"; state each fact as a short third-person sentence about the user.
Return JSON only: {{"question": "...", "key_sequence": [{{"date": "YYYY-MM-DD", "fact": "..."}}]}}
in correct chronological order, using only facts above.

SOURCE MESSAGES:
{src}"""
P_NOW = """Below are the last 3 messages the user sent (oldest first), each dated.
Task:
1. Write ONE question about the user's CURRENT situation as of the last message, answered by these messages, and its
   answer key taken from the MESSAGES (<= 25 words).
2. Classify the underlying statement: "actual" (a real arrangement the user had),
   "hypothetical" (conditional, hypothetical, research, or role-play), "unclear".
Return JSON {{"q":"...","key":"...","mode":"actual|hypothetical|unclear","why":"short reason"}}
or {{"drop":true}} if no usable question about the user exists.

MESSAGES:
{msgs}"""


def ask(prompt, mt):
    """-> parsed dict or None; empty/non-JSON replies retried up to 3 times (format failures)."""
    if sc.COST["usd"] > BUDGET_USD:
        return {"_budget": True}
    for _ in range(3):
        raw = sc.llm(WRITER, prompt, mt)
        if not raw:
            continue
        try:
            o = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            if isinstance(o, dict):
                return o
        except Exception:
            continue
    return None


def gen(it):
    t = it["type"]
    base = {"type": t, "source": it["source"], "conv": it["conv"], "date": it["date"],
            "src_text": it["src_text"], "set": "p6_" + t}
    if t in ("history", "recent", "then"):
        o = ask((P_THEN if t == "then" else P_SINGLE).format(date=it["date"], msg=it["msg"], year=it["date"][:4]), 400)
        if o is None:
            return None, "not_json"
        if o.get("_budget"):
            return None, "budget"
        q, k = str(o.get("q") or "").strip(), str(o.get("key") or "").strip()
        if o.get("drop") or len(q) < 12 or len(k) < 3:
            return None, "drop"
        if t == "then" and not re.search(r"\b(19|20)\d\d\b", q):
            return None, "no_year_in_question"
        return {**base, "q": q, "key": k}, "kept"
    if t == "now":
        o = ask(P_NOW.format(msgs=it["msg"]), 500)
        if o is None:
            return None, "not_json"
        if o.get("_budget"):
            return None, "budget"
        q, k = str(o.get("q") or "").strip(), str(o.get("key") or "").strip()
        mode = str(o.get("mode", "")).strip().lower()
        if o.get("drop") or len(q) < 10 or len(k) < 3:
            return None, "drop"
        if mode != "actual":
            return None, "mode:" + (mode or "unclear")
        return {**base, "q": q, "key": k, "mode": mode}, "kept"
    o = ask(P_ORDER.format(e=it["title"], src=it["msg"]), 600)
    if o is None:
        return None, "not_json"
    if o.get("_budget"):
        return None, "budget"
    q = str(o.get("question") or "").strip()
    ks = [{"date": str(x["date"])[:10], "fact": str(x["fact"])[:200]} for x in (o.get("key_sequence") or [])
          if isinstance(x, dict) and x.get("date") and x.get("fact")][:5]
    if not q:
        return None, "no_question"
    if len(ks) < 3:
        return None, "fewer_than_3_facts"
    return {**base, "q": q, "key": "; ".join("%s %s" % (x["date"], x["fact"][:120]) for x in ks), "key_seq": ks,
            "n_source_dates": it["n_dates"]}, "kept"


def build_pool(convs, rng, ex_texts, ex_convs):
    """Candidate items per type, chosen from counts and raw text only (before any model call)."""
    ok = lambda t: MIN_LEN <= len(t) <= MAX_LEN and clean_ok(t)
    pool = defaultdict(list)
    for source, cid, title, msgs in convs:
        if cid in ex_convs:
            continue
        good = [(t, re.sub(r"\s+", " ", x)) for t, x in msgs if ok(x) and norm(x)[:100] not in ex_texts]
        if not good:
            continue
        by_date = {}
        for t, x in good:
            by_date.setdefault(day(t), x)
        dates = sorted(by_date)
        if source in ("chatgpt", "grok"):
            for d in rng.sample(dates, min(2, len(dates))):
                pool["history"].append({"source": source, "conv": cid, "date": d, "src_text": by_date[d],
                                        "msg": by_date[d][:CLIP], "title": title, "type": "history"})
            d = rng.choice(dates)
            pool["then"].append({"source": source, "conv": cid, "date": d, "src_text": by_date[d],
                                 "msg": by_date[d][:CLIP], "title": title, "type": "then"})
        if source == "session":
            for d in dates:
                if d >= RECENT_FROM:
                    pool["recent"].append({"source": source, "conv": cid, "date": d, "src_text": by_date[d],
                                           "msg": by_date[d][:CLIP], "title": title, "type": "recent"})
        if len(dates) >= 3:
            g = min(2, len(dates) // 3)
            for k in range(g):
                ds = dates[k::g]
                if len(ds) > 5:
                    ds = sorted({ds[0], ds[len(ds) // 3], ds[len(ds) // 2], ds[2 * len(ds) // 3], ds[-1]})
                if len(ds) >= 3:
                    pool["order"].append({"source": source, "conv": cid, "date": ds[0], "title": title, "type": "order",
                                          "n_dates": len(ds), "src_text": " || ".join(by_date[d] for d in ds),
                                          "msg": "\n".join("%s: %s" % (d, by_date[d][:CLIP]) for d in ds)})
        for end in (len(good), len(good) // 2):          # last window, and a mid-conversation window
            if end >= 3:
                last = good[end - 3:end]
                pool["now"].append({"source": source, "conv": cid, "date": day(last[-1][0]), "title": title, "type": "now",
                                    "src_text": " || ".join(x for _, x in last),
                                    "msg": "\n".join("[%s] %s" % (day(t), x[:CLIP]) for t, x in last)})
    return pool


def split_assign(kept):
    """Per type: cap at TARGET, then assign whole conversations to TUNE/HOLDOUT/LOCKED (40/30/30 of the
    ACTUAL count, greedy by largest deficit). A conversation never straddles splits."""
    rng = random.Random(SEED + 1)
    owner = {}
    out = {s: [] for s in SPLITS}
    dropped = Counter()
    for t in ("order", "now", "recent", "then", "history"):    # scarcest types first
        items = [q for q in kept if q["type"] == t]
        rng.shuffle(items)
        dropped["over_cap:" + t] += max(0, len(items) - TARGET[t])
        items = items[:TARGET[t]]
        want = {s: SPLITS[s] * len(items) for s in SPLITS}
        have = Counter()
        for q in items:
            c = q["conv"]
            if c not in owner:
                owner[c] = max(want, key=lambda k: (want[k] - have[k], -list(SPLITS).index(k)))
            have[owner[c]] += 1
            out[owner[c]].append(q)
    return out, dropped


def main():
    if os.path.exists(os.path.join(OUT, "MANIFEST.json")):
        print("REFUSED: %s already frozen" % OUT)
        return 2
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    convs = chatgpt_convs() + grok_convs() + session_convs()
    print("conversations:", dict(Counter(c[0] for c in convs)), flush=True)
    ex_texts, ex_convs, ex_q = prior_exclusions()
    pool = build_pool(convs, rng, ex_texts, ex_convs)
    items = []
    for t, n in OFFER.items():
        p = pool[t]
        rng.shuffle(p)
        seen, pick = Counter(), []
        for it in p:
            if seen[it["conv"]] < {"order": 2, "recent": 3, "now": 2}.get(t, 1):
                seen[it["conv"]] += 1
                pick.append(it)
        print("pool %-8s candidates=%d offered=%d" % (t, len(p), min(n, len(pick))), flush=True)
        items += pick[:n]
    by_t = {t: [i for i in items if i["type"] == t] for t in TARGET}
    items = [x for grp in zip(*[v + [None] * (max(map(len, by_t.values())) - len(v)) for v in by_t.values()]) for x in grp if x]
    # hard deadline: a call stuck on a slow socket must not block the freeze; stragglers count as "timeout"
    ex = ThreadPoolExecutor(16)
    futs = [ex.submit(gen, it) for it in items]
    deadline = time.time() + DEADLINE_S
    res = []
    for f in futs:
        try:
            res.append(f.result(timeout=max(0.1, deadline - time.time())))
        except Exception:
            res.append((None, "timeout"))
    ex.shutdown(wait=False, cancel_futures=True)
    outcomes = {t: dict(Counter(r for (q, r), it in zip(res, items) if it["type"] == t)) for t in TARGET}
    print("writer outcomes:", outcomes, "usd=%.4f" % sc.COST["usd"], flush=True)
    kept, seen_q, seen_src, rej = [], set(), set(), Counter()
    for q, r in res:
        if not q:
            continue
        nq, ns = norm(q["q"]), norm(q["src_text"])[:200]
        if nq in ex_q or nq in seen_q:
            rej["duplicate_question"] += 1
            continue
        if ns in seen_src:
            rej["duplicate_source"] += 1
            continue
        if re.search(r"\bdee\b", q["q"] + " " + q["key"], re.I):
            rej["name_in_question"] += 1
            continue
        if any(p.search(q["q"] + " " + q["key"]) for p in SECRET):
            rej["secret_or_pii_in_question"] += 1
            continue
        seen_q.add(nq)
        seen_src.add(ns)
        kept.append(q)
    splits, dropped = split_assign(kept)
    files, all_q = {}, []
    for s, qs in splits.items():
        qs.sort(key=lambda q: (q["type"], q["conv"], q["date"]))
        for i, q in enumerate(qs):
            q["id"] = "p6_%s_%03d" % (s.lower(), i)
            q["split"] = s
        path = os.path.join(OUT, "questions_p6_%s.json" % s.lower())
        with open(path, "w") as f:
            json.dump(qs, f, indent=1)
        files[s] = {"path": path, "md5": hashlib.md5(open(path, "rb").read()).hexdigest(), "n": len(qs),
                    "by_type": dict(Counter(q["type"] for q in qs))}
        all_q += qs
    man = {"files": files, "total": len(all_q), "target": TARGET,
           "offered": {t: sum(1 for i in items if i["type"] == t) for t in TARGET},
           "writer_outcomes": outcomes, "post_filter_rejects": dict(rej), "split_drops": dict(dropped),
           "sources": dict(Counter(q["source"] for q in all_q)), "writer": WRITER, "seed": SEED,
           "cost_usd": round(sc.COST["usd"], 4), "stores_opened": 0,
           "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(man, open(os.path.join(OUT, "MANIFEST.json"), "w"), indent=1)
    print("FROZEN", json.dumps(man), flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)           # do not wait for any straggling HTTP threads
