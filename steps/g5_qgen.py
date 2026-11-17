#!/usr/bin/env python3
"""G5: fresh questions for the raw-layer clean test, no leakage by construction.

Stage A (paraphraser) sees one raw user message (or a short window of them) and writes a neutral
1-2 sentence SUMMARY in its own words (no quoted phrases) plus a gold answer.
Stage B (question writer) sees ONLY that summary (+ the date, which is metadata, not message text) and
writes the question + gold answer. It never sees the message text, a store or any result.
Leak filters (exact rules, applied to the QUESTION text vs the verbatim source message(s)):
  R1 shared word 4-gram (all words, lower-cased, \\W-split) that contains >= 2 content words -> drop "leak_4gram"
  R2 longest common contiguous run of CONTENT words (message content-word sequence vs question content-word
     sequence) >= 3, i.e. the question holds MORE THAN 2 content words of the message in order -> drop "leak_content_run"
  content word = not in STOP, len >= 3, not "user".
Other drops: stage A refused/not JSON, quote marks in the summary, stage B refused/not JSON, question/gold too short,
question names "summary"/"the message", "then" question without the source year, A-gold vs B-gold share no
content word ("gold_mismatch"), "now" source not classified actual.
Candidates are drawn by seed before any model call; the first K kept per type (candidate order) are taken.

usage: g5_qgen.py OUTDIR   (reads /root/g5/pool.jsonl + all_raw.jsonl; writes OUTDIR/questions_g5.json + qgen_stats.json)
Private text stays outside the repo.
"""
import json, os, random, re, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.environ.get("MPMF_SRC", "/root/mpmf-release/src"))
from mpmf_plus.eval import scorecard as sc   # noqa: E402  (LLM client + COST only; no store is opened)

OUT = sys.argv[1]
SEED = 5505
PARAPHRASER = "deepseek/deepseek-v4.1-flash"
WRITER = "deepseek/deepseek-v4.1-flash"
GEN_CEIL_USD = 0.8                   # this process only (cap is $5; ceiling = well under 80%)
MIN_LEN, MAX_LEN, CLIP = 60, 1500, 600
RECENT_FROM = "2026-10-01"
CAND = {"history": 300, "then": 240, "recent": 90, "now": 90, "order": 60}
QUOTA = {"history": 36, "then": 24, "recent": 12, "now": 14, "order": 14}
STOP = set("""a about above after again against all also am an and any are aren as at be because been before being
below between both but by can cannot could did do does doing don down during each few for from further had has have having
he her here hers him his how i if in into is it its just let me more most my no nor not now of off on once only or other
our out over own same she should so some such than that the their them then there these they this those through to too under
until up us very was we were what when where which while who whom why will with would you your yours it's i'm i've i'd
don't can't won't didn't doesn't isn't what's that's there's user user's say said says tell told ask asked asks want wants
wanted like get got make made one two""".split())
norm = lambda s: re.sub(r"\W+", " ", (s or "").lower()).strip()
words = lambda s: norm(s).split()
content = lambda ws: [w for w in ws if w not in STOP and len(w) >= 3]


def longest_run(a, b):
    best = 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def leak(q, src):
    """-> (reason or None, longest shared word run, longest content run)"""
    qw, sw = words(q), words(src)
    sg = {tuple(sw[i:i + 4]) for i in range(len(sw) - 3)}
    r1 = any(tuple(qw[i:i + 4]) in sg and len(content(qw[i:i + 4])) >= 2 for i in range(len(qw) - 3))
    crun = longest_run(content(qw), content(sw))
    wrun = longest_run(qw, sw)
    if r1:
        return "leak_4gram", wrun, crun
    if crun >= 3:
        return "leak_content_run", wrun, crun
    return None, wrun, crun


def ask(model, prompt, mt):
    if sc.COST["usd"] > GEN_CEIL_USD:
        return {"_budget": True}
    for _ in range(3):
        raw = sc.llm(model, prompt, mt)
        if not raw:
            continue
        try:
            o = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            if isinstance(o, dict):
                return o
        except Exception:
            continue
    return None


A_SINGLE = """Below is one message a person sent on {date}. Do two things.
1. Write a neutral SUMMARY (1-2 sentences, third person, refer to the writer as "the user") in YOUR OWN words:
   do not copy phrases from the message, use no quotation marks, but keep the concrete facts (who/what/where/numbers).
2. Give a short gold ANSWER (<= 25 words) to one self-contained question about the user that this message answers.
If the message has no durable, self-contained fact about the user (small talk, a bare command, code), return {{"drop":true}}.
Return JSON {{"summary":"...","gold":"..."}} or {{"drop":true}}.

MESSAGE:
{msg}"""
A_NOW = """Below are the last 3 messages a person sent (oldest first), dated. Do three things.
1. Write a neutral SUMMARY (1-2 sentences, third person, "the user") of the person's CURRENT situation as of the last message,
   in YOUR OWN words: do not copy phrases, no quotation marks, keep the concrete facts.
2. Give a short gold ANSWER (<= 25 words) to a question about that current situation.
3. Classify: "actual" (a real arrangement), "hypothetical" (conditional, research, role-play), or "unclear".
Return JSON {{"summary":"...","gold":"...","mode":"actual|hypothetical|unclear"}} or {{"drop":true}}.

MESSAGES:
{msg}"""
A_ORDER = """Below are dated messages one person sent in one conversation, oldest first. For EACH message write a neutral
one-sentence summary (third person, "the user") in YOUR OWN words: do not copy phrases, no quotation marks, keep the concrete
facts. If fewer than 3 messages contain a durable fact about the user, return {{"drop":true}}.
Return JSON {{"events":[{{"date":"YYYY-MM-DD","summary":"..."}}]}} in the same order, or {{"drop":true}}.

MESSAGES:
{msg}"""
B_SINGLE = """You write questions for a memory benchmark. Below is a short summary of something a person told an assistant on {date}.
Write ONE self-contained question about the user that this summary answers, and its answer key (<= 25 words) taken only
from the summary. {extra}Do not mention "summary", messages, or memory. Refer to the person as "the user".
Return JSON {{"q":"...","key":"..."}} or {{"drop":true}}.

SUMMARY:
{summary}"""
B_THEN = "Phrase it with the year, e.g. \"In {year}, what/where/who ...?\". "
B_NOW = "The question must ask about the user's CURRENT situation as of {date}. "
B_ORDER = """You write questions for a memory benchmark. Below are dated event summaries about one person, oldest first.
Write ONE question asking about the CHRONOLOGICAL ORDER of these events (e.g. "In what order did ..."). Do not mention dates in
the question, nor "summary", messages or memory; refer to the person as "the user".
Answer key: the events in correct order, one short fact per event, with its date.
Return JSON {{"q":"...","key":"..."}}.

EVENTS:
{events}"""


def build_candidates(rng):
    pool = [json.loads(l) for l in open("/root/g5/pool.jsonl")]
    allraw = [json.loads(l) for l in open("/root/g5/all_raw.jsonl")]
    ok_names = {r["name"] for r in pool}
    by_conv = defaultdict(list)
    for r in allraw:
        by_conv[r["conv"]].append(r)
    for v in by_conv.values():
        v.sort(key=lambda r: (r["idx"], r["date"]))
    single = [r for r in pool if MIN_LEN <= len(r["text"]) <= MAX_LEN and r["date"] and r["conv"]]
    convs = sorted({r["conv"] for r in single})
    rng.shuffle(convs)
    used, out = set(), defaultdict(list)

    def take(typ, conv, **kw):
        used.add(conv)
        out[typ].append(dict(type=typ, conv=conv, **kw))

    byc = defaultdict(list)
    for r in single:
        byc[r["conv"]].append(r)
    # order first (needs >= 3 distinct dates among unexcluded messages), then now, then single-message types
    for c in convs:
        if len(out["order"]) >= CAND["order"]:
            break
        days = {}
        for r in byc[c]:
            days.setdefault(r["date"], r)
        if len(days) >= 3 and c not in used:
            ds = sorted(days)
            ds = sorted(rng.sample(ds, min(4, len(ds))))
            take("order", c, msgs=[days[d] for d in ds])
    for c in convs:
        if len(out["now"]) >= CAND["now"]:
            break
        rows = [r for r in by_conv[c] if r["name"] in ok_names and MIN_LEN <= len(r["text"]) <= MAX_LEN]
        if c not in used and len(rows) >= 3:
            take("now", c, msgs=rows[-3:])
    for typ, sel in (("recent", lambda r: r["date"] >= RECENT_FROM),
                     ("then", lambda r: r["date"] < RECENT_FROM),
                     ("history", lambda r: True)):
        for c in convs:
            if len(out[typ]) >= CAND[typ]:
                break
            rows = [r for r in byc[c] if sel(r)]
            if c not in used and rows:
                if typ == "recent":      # the recent pool is small: up to 3 distinct messages per conversation
                    for r in rng.sample(rows, min(3, len(rows))):
                        out[typ].append(dict(type=typ, conv=c, msgs=[r]))
                    used.add(c)
                else:
                    take(typ, c, msgs=[rng.choice(rows)])
    return out


def gen(it):
    typ, msgs = it["type"], it["msgs"]
    src = " \n".join(m["text"][:CLIP] for m in msgs)
    d = msgs[-1]["date"]
    base = {"type": typ, "conv": it["conv"], "date": d, "src_names": [m["name"] for m in msgs],
            "src_text": " || ".join(re.sub(r"\s+", " ", m["text"][:CLIP]) for m in msgs)}
    if typ == "order":
        a = ask(PARAPHRASER, A_ORDER.format(msg="\n".join("%s: %s" % (m["date"], re.sub(r"\s+", " ", m["text"][:CLIP]))
                                                           for m in msgs)), 700)
        if a is None:
            return None, "A_not_json"
        if a.get("_budget"):
            return None, "budget"
        ev = [e for e in (a.get("events") or []) if isinstance(e, dict) and e.get("summary") and e.get("date")]
        if a.get("drop") or len(ev) < 3:
            return None, "A_drop"
        summ = " | ".join("%s %s" % (str(e["date"])[:10], e["summary"]) for e in ev)
        if '"' in summ or "\u201c" in summ:
            return None, "A_quotes"
        b = ask(WRITER, B_ORDER.format(events="\n".join("%s: %s" % (str(e["date"])[:10], e["summary"]) for e in ev)), 600)
        gA = ""
    else:
        if typ == "now":
            a = ask(PARAPHRASER, A_NOW.format(msg="\n".join("%s: %s" % (m["date"], re.sub(r"\s+", " ", m["text"][:CLIP]))
                                                             for m in msgs)), 500)
        else:
            a = ask(PARAPHRASER, A_SINGLE.format(date=d, msg=src), 400)
        if a is None:
            return None, "A_not_json"
        if a.get("_budget"):
            return None, "budget"
        summ, gA = str(a.get("summary") or "").strip(), str(a.get("gold") or "").strip()
        if a.get("drop") or len(summ) < 20 or len(gA) < 2:
            return None, "A_drop"
        if '"' in summ or "\u201c" in summ or "\u2018" in summ:
            return None, "A_quotes"
        if typ == "now" and str(a.get("mode", "")).strip().lower() != "actual":
            return None, "A_mode_not_actual"
        extra = (B_THEN.format(year=d[:4]) if typ == "then" else B_NOW.format(date=d) if typ == "now" else "")
        b = ask(WRITER, B_SINGLE.format(date=d, summary=summ, extra=extra), 400)
    if b is None:
        return None, "B_not_json"
    if b.get("_budget"):
        return None, "budget"
    q, k = str(b.get("q") or "").strip(), str(b.get("key") or "").strip()
    if b.get("drop") or len(q) < 12 or len(k) < 3:
        return None, "B_drop"
    if re.search(r"\bsummar(y|ies)\b|\bthe message\b", q, re.I):
        return None, "B_mentions_summary"
    if typ == "then" and d[:4] not in q:
        return None, "then_no_year"
    reason, wrun, crun = leak(q, src)
    if reason:
        return None, reason
    if gA and not (set(content(words(gA))) & set(content(words(k)))):
        return None, "gold_mismatch"
    return {**base, "q": q, "key": k, "summary": summ, "a_gold": gA, "longest_word_run": wrun,
            "longest_content_run": crun}, "kept"


def main():
    rng = random.Random(SEED)
    cands = build_candidates(rng)
    print("candidates", {t: len(v) for t, v in cands.items()}, flush=True)
    stats = {t: Counter() for t in cands}
    kept = {t: [] for t in cands}
    t0 = time.time()
    for typ in ("history", "then", "recent", "now", "order"):
        with ThreadPoolExecutor(6) as ex:
            res = list(ex.map(gen, cands[typ]))
        for r, why in res:
            stats[typ][why] += 1
            if r:
                kept[typ].append(r)
        print(typ, dict(stats[typ]), "$%.3f" % sc.COST["usd"], flush=True)
    final = []
    for typ in ("history", "then", "recent", "now", "order"):
        for r in kept[typ][:QUOTA[typ]]:        # candidate order, fixed before any model call
            r["id"] = "g5_%03d" % (len(final) + 1)
            r["set"] = "g5"
            final.append(r)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "questions_g5.json")
    json.dump(final, open(path, "w"), indent=1)
    os.chmod(path, 0o444)
    st = {"candidates": {t: len(v) for t, v in cands.items()}, "drops": {t: dict(c) for t, c in stats.items()},
          "kept_before_quota": {t: len(v) for t, v in kept.items()},
          "final": dict(Counter(r["type"] for r in final)), "n": len(final),
          "cost_usd": round(sc.COST["usd"], 4), "calls": sc.COST["calls"], "secs": round(time.time() - t0)}
    json.dump(st, open(os.path.join(OUT, "qgen_stats.json"), "w"), indent=1)
    print(json.dumps(st), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
