#!/usr/bin/env python3
"""thinking_capture.py — capture assistant reasoning from state.db into the
thinking-trace MPMF store (`athena_thinking.store_thinking`).

DISABLED by default: `thinking_capture_enabled` is false in
$MPMF_HOME/thinking_capture.config.json. Nothing is written until the user enables it.

Filter (2026-10-13, second dry-run — the user's refined spec):
  * role='assistant' AND tool_calls empty — the message that carries the FINAL
    reply to a user turn (the next message in the session is a user turn, or the
    message is the last one), so intermediate tool-narration turns are skipped;
  * session model in {deepseek/deepseek-v4.1-flash, deepseek/deepseek-v4-flash};
  * session is NOT synthetic/cron (source not in {cron, subagent});
  * len(reasoning) >= 200;
  * md5(reasoning) dedup.
New nodes carry source_session=f"live_{int(ts)}" so they stay separable from the
pre-Sep-27 corpus.

Write path reads state.db with a CURSOR (last processed message id, persisted in
$MPMF_HOME/health/thinking_capture_state.json), caps at `daily_cap` (100) nodes
per UTC day, and backs the thinking store up before the first write.

It does NOT change what gets injected: the plugin recall/injection path is
untouched.

CLI:
  python3 thinking_capture.py --dry-run     # read-only report; writes nothing
  python3 thinking_capture.py --once        # one write pass (only if enabled)
  python3 thinking_capture.py --status      # cursor / counters
"""

import sys, os, re, json, time, random, shutil, hashlib, datetime as dt
from pathlib import Path
from . import paths as _paths

STATE_DB = os.environ.get("MPMF_STATE_DB") or str(_paths.Path.home() / ".hermes" / "state.db")
CONFIG_PATH = str(_paths.home() / "thinking_capture.config.json")
STATE_PATH = str(_paths.log_path("thinking_capture_state.json"))
THINKING_STORE = str(_paths.thinking_store())
BACKUP_DIR = str(_paths.home() / "backups")
SINCE_DEFAULT = "2026-09-01T00:00:00"

DEFAULTS = {
    "thinking_capture_enabled": False,      # the user enables when ready
    "daily_cap": 100,                        # max nodes written per UTC day
    "models": ["deepseek-v4.1-flash", "deepseek-v4-flash"],
    "exclude_sources": ["cron", "subagent"],
    "min_reasoning_chars": 200,
    "since": SINCE_DEFAULT,
    "backup_before_first_write": True,
}


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH) as f:
            cfg.update(json.load(f))
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[thinking-capture] config load failed ({e!r}); using defaults", file=sys.stderr)
    return cfg


def save_default_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULTS, f, indent=2)
    return CONFIG_PATH


def load_state() -> dict:
    try:
        with open(STATE_PATH) as f:
            s = json.load(f)
    except Exception:
        s = {}
    s.setdefault("last_id", 0)
    s.setdefault("per_day", {})
    s.setdefault("seen_md5", [])
    s.setdefault("written", 0)
    return s


def save_state(s) -> None:
    Path(STATE_PATH).parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=2, default=str)
    os.replace(tmp, STATE_PATH)


def _since_ts(cfg) -> float:
    try:
        return dt.datetime.fromisoformat(cfg["since"]).replace(tzinfo=dt.timezone.utc).timestamp()
    except Exception:
        return dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc).timestamp()


# ── selection ──────────────────────────────────────────────────────────────

#: Round 6 item 3: skip any turn whose TRIGGERING user message is synthetic
#: (background-process notice, cron preamble, compaction block, task-list or
#: async-delegation filler). Same tuple the self-correct sampler uses.
_SYNTHETIC_PREFIXES = ("[IMPORTANT: Background process",
                       "[IMPORTANT: You are running as a scheduled cron job",
                       "[CONTEXT COMPACTION", "[Your active task list",
                       "[ASYNC DELEGATION")


def _is_synthetic_user(user_text: str) -> bool:
    if not user_text:
        return False
    t = user_text.lstrip()
    return any(t.startswith(p) for p in _SYNTHETIC_PREFIXES)

def _model_clause(cfg):
    like = " OR ".join(["s.model LIKE ?"] * len(cfg["models"]))
    return like, [f"%{m}%" for m in cfg["models"]]


def select_final_replies(cfg, after_id: int = 0, limit: int = None):
    """Yield dicts for assistant FINAL replies (no tool_calls) whose reasoning
    passes the filter. Ordered by message id. `after_id` is the cursor."""
    import sqlite3
    since = _since_ts(cfg)
    mlike, mparams = _model_clause(cfg)
    src_excl = cfg["exclude_sources"]
    src_ph = ",".join("?" * len(src_excl))
    sql = f"""
        select m.id, m.session_id, m.timestamp, m.reasoning, m.content, s.model, s.source
        from messages m join sessions s on s.id = m.session_id
        where m.role='assistant'
          and (m.tool_calls is null or m.tool_calls='' or m.tool_calls='[]')
          and m.timestamp > ? and m.id > ?
          and m.reasoning is not null and length(m.reasoning) >= ?
          and ({mlike})
          and (s.source is null or s.source not in ({src_ph}))
        order by m.id
    """
    params = [since, after_id, cfg["min_reasoning_chars"]] + mparams + src_excl
    c = sqlite3.connect(STATE_DB)
    rows = list(c.execute(sql, params))
    n = 0
    for mid, sid, ts, rz, content, model, source in rows:
        # final reply: next message in the session is a user turn (or none)
        nxt = c.execute(
            "select role from messages where session_id=? and id>? order by id limit 1",
            (sid, mid)).fetchone()
        if nxt is not None and nxt[0] != "user":
            continue
        # the user turn it answered
        u = c.execute(
            "select id, content from messages where session_id=? and role='user' and id<? "
            "order by id desc limit 1", (sid, mid)).fetchone()
        user_id, user_text = (u[0], (u[1] or "")) if u else (None, "")
        # Round 6 item 3: drop synthetic-triggered turns.
        if _is_synthetic_user(user_text):
            continue
        yield {"id": mid, "session_id": sid, "ts": ts, "reasoning": rz,
               "content": content or "", "model": model, "source": source,
               "user_id": user_id, "user_text": user_text}
        n += 1
        if limit and n >= limit:
            return


def _dedup(items):
    """md5-dedup on reasoning text (first occurrence wins)."""
    seen, out = set(), []
    for it in items:
        h = hashlib.md5(it["reasoning"].encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        it["md5"] = h
        out.append(it)
    return out


def collect(cfg, after_id=0) -> list:
    return _dedup(list(select_final_replies(cfg, after_id=after_id)))


# ── concepts (matches the daemon's extractor) ──────────────────────────────

def extract_concepts(text: str, max_c: int = 5) -> list:
    concepts = set()
    for m in re.finditer(r'"([^"]{3,40})"', text):
        concepts.add(m.group(1).lower().replace(" ", "_"))
    for m in re.finditer(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', text):
        concepts.add(m.group(0).lower())
    return list(concepts)[:max_c]


# ── dry-run report ─────────────────────────────────────────────────────────

def _pct(a, p):
    if not a:
        return 0
    import math
    return a[min(len(a) - 1, math.ceil(p * len(a)) - 1)]


def dry_run(cfg) -> dict:
    items = collect(cfg)
    print(f"=== thinking_capture DRY-RUN (writes nothing) ===")
    print(f"filter: final-reply assistant msgs, no tool_calls, models={cfg['models']}, "
          f"source∉{cfg['exclude_sources']}, len(reasoning)>={cfg['min_reasoning_chars']}, md5-dedup")
    print(f"since {cfg['since']}")
    print(f"\ntotal captured-candidate messages: {len(items)}")
    per_day, per_model, per_src = {}, {}, {}
    lengths = []
    for it in items:
        d = dt.datetime.utcfromtimestamp(it["ts"]).strftime("%Y-%m-%d")
        per_day[d] = per_day.get(d, 0) + 1
        per_model[it["model"]] = per_model.get(it["model"], 0) + 1
        per_src[it["source"]] = per_src.get(it["source"], 0) + 1
        lengths.append(len(it["reasoning"]))
    print("\nper day since Sep 1:")
    for d in sorted(per_day):
        print(f"    {d}  {per_day[d]}")
    print("\nper model:")
    for m, n in sorted(per_model.items(), key=lambda x: -x[1]):
        print(f"    {m}  {n}")
    print("\nper session source:")
    for s, n in sorted(per_src.items(), key=lambda x: -x[1]):
        print(f"    {s}  {n}")
    lengths.sort()
    print(f"\nlength distribution (chars): min {lengths[0] if lengths else 0} "
          f"p25 {_pct(lengths,.25)} median {_pct(lengths,.5)} p75 {_pct(lengths,.75)} "
          f"p95 {_pct(lengths,.95)} max {lengths[-1] if lengths else 0}")
    tot = sum(lengths)
    KVEC = 1024 * 4
    print(f"store volume if ALL ingested: {len(items):,} nodes ~{(tot + len(items)*KVEC)/1e6:.1f} MB")
    print(f"existing thinking store: 3,984 nodes  |  daily cap {cfg['daily_cap']}/day")

    random.seed(13)
    sample = random.sample(items, min(10, len(items)))
    print("\n=== 10 random samples (reasoning trimmed + the user turn it answered) ===")
    for it in sample:
        d = dt.datetime.utcfromtimestamp(it["ts"]).strftime("%Y-%m-%d %H:%M")
        print(f"\n--- msg {it['id']} [{d}] model={it['model']} src={it['source']} "
              f"len={len(it['reasoning'])} ---")
        print(f"    USER TURN: {(it['user_text'] or '(none)')[:220]!r}")
        print(f"    REASONING: {it['reasoning'][:300]!r}")
    return {"total": len(items), "per_day": per_day, "per_model": per_model}


# ── write path (disabled unless thinking_capture_enabled) ──────────────────

def _backup_store():
    Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)
    dest = os.path.join(BACKUP_DIR,
                        f"athena-thinking.mpmf.tc_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(THINKING_STORE, dest)
    return dest


def write_once(cfg) -> dict:
    if not cfg.get("thinking_capture_enabled"):
        return {"enabled": False, "note": "thinking_capture_enabled=false — no writes"}
    st = load_state()
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    used_today = st["per_day"].get(today, 0)
    remaining = max(0, int(cfg["daily_cap"]) - used_today)
    if remaining <= 0:
        return {"enabled": True, "written": 0, "note": f"daily cap {cfg['daily_cap']} reached"}

    from . import paths as _p; sys.path.insert(0, str(_p.home()))
    from athena_thinking import store_thinking

    seen = set(st["seen_md5"])
    items = collect(cfg, after_id=int(st["last_id"]))
    backup = None
    written = 0
    last_id = int(st["last_id"])
    for it in items:
        if written >= remaining:
            break
        if it["md5"] in seen:
            last_id = max(last_id, it["id"])
            continue
        if written == 0 and cfg.get("backup_before_first_write"):
            backup = _backup_store()
        store_thinking(
            context=f"Final reply to user turn: {(it['user_text'] or '')[:180]}",
            thinking=it["reasoning"],
            conclusion=(it["content"] or "").strip()[:200] or "(captured final reply reasoning)",
            concepts=extract_concepts(it["reasoning"]),
            session_id=f"live_{int(it['ts'])}",
            outcome="captured",
        )
        seen.add(it["md5"])
        written += 1
        last_id = max(last_id, it["id"])
        st["per_day"][today] = st["per_day"].get(today, 0) + 1
    st["last_id"] = last_id
    st["seen_md5"] = list(seen)[-20000:]      # bounded
    st["written"] += written
    save_state(st)
    return {"enabled": True, "written": written, "cursor": last_id,
            "backup": backup, "day_used": st["per_day"].get(today, 0)}


def main():
    import argparse
    p = argparse.ArgumentParser(description="Thinking capture (state.db -> thinking store)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--once", action="store_true")
    p.add_argument("--status", action="store_true")
    args = p.parse_args()
    save_default_config()
    cfg = load_config()
    if args.dry_run:
        dry_run(cfg); return
    if args.status:
        st = load_state()
        print(json.dumps({"enabled": cfg.get("thinking_capture_enabled"),
                          "cursor": st["last_id"], "per_day": st["per_day"],
                          "written": st["written"]}, indent=2)); return
    if args.once:
        print(json.dumps(write_once(cfg), indent=2, default=str)); return
    print("nothing to do; use --dry-run | --once | --status")


if __name__ == "__main__":
    main()
