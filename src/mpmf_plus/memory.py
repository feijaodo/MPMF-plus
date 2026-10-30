#!/usr/bin/env python3
"""
memory.py — an MPMF+ memory store (upgraded from MPMF plain).

MPMF+ features ported from /root/mpmf_benchmark/ Phase 2 (mpmf+ adapter):
  - 5-plane structured extraction at ingest (deepseek-v4.1-flash, fast + cheap)
  - Edge derivation (5 grounded rules, see edge_derivation.py)
  - Widen-then-rerank query path: retrieve top-20 with full 5-plane graph
    traversal, rerank with local cross-encoder/ms-marco-MiniLM-L-6-v2, return top-5
  - Local embedder (mxbai-embed-large 1024d) + local reranker (no API at query time)
  - requests-based embed path with hard (connect=10, read=30) timeouts (FAULT-4 fix)

USAGE (from anywhere):
    import sys; sys.path.insert(0, "/root/.mpmf")
    from mpmf_plus.memory import store, recall, get, stats, link

    store("text", name="key", planes=Plane.ASSOCIATIVE.value,
          summary="", wiki={"topic": "x"}, importance=0.8)
    for hit in recall("query"): print(hit["name"], hit["text"])
    link("a", "b", Plane.ASSOCIATIVE, "RELATED_TO")
"""
from __future__ import annotations
import sys
import os
import json
import time
import re
import math
import hashlib
import struct
import threading
import requests
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any, Set

# r59g: no hardcoded filesystem default. The core ships inside this package
# (`mpmf`), so nothing is added to sys.path unless the caller overrides it.
_core_override = os.environ.get("MPMF_CORE_PATH")
if _core_override:
    sys.path.append(_core_override)
from .errors import ConfigError, EmbedderError, StoreNotFoundError
from . import store as _store
from .dedup_helpers import _text_hash, _audit_dedup, DEDUP_COSINE_THRESHOLD, check_dedup  # noqa: E402  (r59d: vendored)
import contextlib
import mpmf as _mpmf_core
from mpmf import MPMFMemory, Plane, NodeType
from mpmf.format import Plane as PlaneEnum
from mpmf.retrieval import RetrievalPipeline

# A5.1: every LLM/embedding failure must reach $MPMF_HOME/health/failures.log.
# The guard lives beside this file; import it defensively so a missing guard
# degrades to a no-op instead of taking the whole write path down with it.
try:
    from llm_call_guard import guarded  # noqa: E402
except Exception:  # pragma: no cover - optional infrastructure
    from contextlib import contextmanager as _contextmanager

    @_contextmanager
    def guarded(component: str):  # type: ignore[misc]
        yield

# ============== CONFIG ==============
# ── Portability layer (r59d) ──────────────────────────────────────────────
# Every path below is derived from MPMF_HOME or from this package's own location;
# none is hardcoded to a filesystem path. MPMF_CORE_PATH / MPMF_STORE_PATH override.
MPMF_HOME = Path(os.environ.get("MPMF_HOME") or "~/.mpmf").expanduser()
MPMF_HEALTH = MPMF_HOME / "health"
MPMF_CORE_PATH = (os.environ.get("MPMF_CORE_PATH")
                  or str(Path(_mpmf_core.__file__).resolve().parent))  # bundled core
for _d in (MPMF_HOME, MPMF_HEALTH, MPMF_HOME / "cache", MPMF_HOME / "state"):
    _d.mkdir(parents=True, exist_ok=True)

# ── Store resolution: store_mode = create | attach (r60) ──────────────────
# Resolved ONCE at import from $MPMF_STORE / $MPMF_STORE_PATH / $MPMF_STORE_MODE
# and the `store_path` / `store_mode` keys of recall.config.json (see
# mpmf_plus/store.py for the full precedence). Env values are therefore read at
# import time, exactly like OLLAMA_URL. A config file that exists but is
# unreadable or invalid raises ConfigError HERE — import must not paper over it.
# Change the store at runtime with configure_store(); tests override PATH.
_SETTINGS = _store.resolve_store()
STORE_PATH = _SETTINGS.path
STORE_MODE = _SETTINGS.mode
PATH = STORE_PATH
PATH_LEGACY = str(MPMF_HOME / "store.mpmf.v1-legacy")


def configure_store(path: Optional[str] = None,
                    mode: Optional[str] = None) -> "_store.StoreSettings":
    """Re-point this process at a store and/or change its mode.

    ``mode="attach"`` will NOT create anything: the next read or write raises
    ``StoreNotFoundError`` if ``path`` is not an existing file. Interactive use
    and tests only — an agent should set the env var or the config key instead.
    """
    global _SETTINGS, STORE_PATH, STORE_MODE, PATH
    _SETTINGS = _store.resolve_store(path=path, mode=mode)
    STORE_PATH = PATH = _SETTINGS.path
    STORE_MODE = _SETTINGS.mode
    _invalidate_open_cache()
    return _SETTINGS


def store_settings() -> "_store.StoreSettings":
    """The effective (path, mode) this process is using."""
    if _SETTINGS.path != PATH:
        # PATH was overridden in place (tests, legacy callers): that wins.
        return _store.StoreSettings(PATH, STORE_MODE, {"path": "module PATH",
                                                       "mode": "module STORE_MODE"})
    return _SETTINGS
EMBED_MODEL = "mxbai-embed-large"          # canonical per MPMF+ v1.0 spec §A1 (1024d native, no padding)
VECTOR_DIM = 1024
# ---- Extraction model config (EXPLICIT; single source of truth) ----
# This was previously an implicit, hardcoded default that DRIFTED between the
# copies of this file (1200 here vs 600 in agent_memory.py). It is now named,
# env-overridable, and documented so the write path's model + token budget have
# exactly one home:
#   MPMF_EXTRACT_MODEL        — extraction model id
#   MPMF_EXTRACT_MAX_TOKENS   — output budget per attempt
#   MPMF_EXTRACT_BUDGET_BUMPS — how many times a `length` truncation doubles it
#   MPMF_EXTRACT_DUMP_DIR     — dir to dump raw failing provider responses into
EXTRACT_MODEL = (
    os.environ.get("MPMF_EXTRACT_MODEL")
    or os.environ.get("MPMF_PLUS_LLM")            # legacy name, still honoured
    or "deepseek/deepseek-v4.1-flash"
)
EXTRACT_MAX_TOKENS = int(os.environ.get("MPMF_EXTRACT_MAX_TOKENS", "1200"))
EXTRACT_BUDGET_BUMPS = int(os.environ.get("MPMF_EXTRACT_BUDGET_BUMPS", "1"))
EXTRACT_DUMP_DIR = os.environ.get("MPMF_EXTRACT_DUMP_DIR", "")
LLM_MODEL = EXTRACT_MODEL       # back-compat alias for external importers
LLM_BASE_URL = "https://openrouter.ai/api/v1"

#: Code-revision marker — bumped on every behaviour-changing edit to this file.
#: After a gateway reload, proving the *running* process picked up the new code:
#:   python3 -c "import sys;sys.path.insert(0,'/root/.mpmf');import athena_memory as a;print(a.CODE_REV)"
CODE_REV = "2026-10-27 r55 memory-arm"

API_KEY = os.environ.get("OPENROUTER_API_KEY")
_ENV_FILE_LOADED = False


def _load_env_file(path: Optional[str] = None) -> None:
    """Parse KEY=VALUE lines into os.environ (existing values win; never prints).

    Round 12 item 4: the LLM choke point must not fail just because the process
    env lacks the key (both the re-enrich cron and the gateway were bitten by
    this). File = $MPMF_ENV_FILE or ~/.hermes/.env. Same parser as
    mpmf_re_enrich_pending.py.
    """
    path = path or os.environ.get("MPMF_ENV_FILE") or str(Path("~/.hermes/.env").expanduser())
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except FileNotFoundError:
        pass


def _api_key() -> Optional[str]:
    """The OpenRouter key, lazily loaded ONCE from MPMF_ENV_FILE if unset.

    Existing environment wins; the value is never printed."""
    global API_KEY, _ENV_FILE_LOADED
    if not API_KEY:
        if not _ENV_FILE_LOADED:
            _ENV_FILE_LOADED = True
            _load_env_file()
        API_KEY = os.environ.get("OPENROUTER_API_KEY")
    return API_KEY


def _llm_complete(prompt: str, temperature: float = 0.7,
                  max_tokens: int = 800, api_key: str = None,
                  model: str = None) -> dict:
    """Generic LLM completion call via the configured provider."""
    ak = api_key or _api_key()
    mdl = model or LLM_MODEL
    if not ak:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    resp = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {ak}", "Content-Type": "application/json"},
        json={
            "model": mdl,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]["message"]
    return {"content": choice["content"], "finish_reason": choice.get("finish_reason", ""), "model": data.get("model", mdl)}


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # local, ~90MB, CPU
WIDEN_K = 20                                # retrieve 20 then rerank
DEFAULT_K = 5
EXTRACT_WORKERS = 16
EXTRACT_TIMEOUT = 60
EMBED_TIMEOUT = (10, 30)                    # (connect, read) hard timeouts
CACHE_DIR = MPMF_HOME / "cache"
EXTRACTIONS_CACHE = CACHE_DIR / "extractions.json"

# ============== EXTRACTION (5-PLANE STRUCTURED) ==============
EXTRACTION_SYSTEM = """You are a memory extractor for an associative memory system. Given a
single user fact and optionally existing related memory nodes, output a JSON
object with 5 planes + a namespace describing the fact, plus relational fields
connecting it to existing knowledge.

Output ONLY valid JSON. No prose, no markdown fences. The shape:

{
  "summary": "<one-line, <=120 chars, the essence of the fact>",
  "namespace": "memory" | "ruminate" | "meta",
  "wiki": {"<key>": "<value>", ...},          // 2-6 short metadata keys
  "planes": {
    "associative": ["<tag1>", "<tag2>", ...], // 3-8 tags for similarity/clustering
    "causal":      "<if X then Y, else empty string>",
    "temporal":    "<if time-bound, free text date/era, else empty string>",
    "functional":  "<category: preference | fact | entity | tool | person | activity | place | project | other>",
    "priority":    <1..5, 5=critical personal context, 1=ephemeral>
  },
  "relates_to": ["<existing node name>", ...],
  "relation_semantics": "confirm" | "contradict" | "extend" | "supersede" | "none",
  "preventive_tool_or_pattern": "<tool/skill/pattern name or empty string>"
}

NAMESPACE (3 categories -- choose the right one):
- "memory"  : Real knowledge about the user, world, projects, people, tools.
              Personal facts, preferences, configurations, project details.
              Examples: "User D prefers Vim", "The Ballerina Mansion project
              lives at /root/ballerina-mansion", "MPMF uses 5 memory planes",
              "the agent name is configured, not assumed". These are CURATION-grade facts.
- "ruminate" : Operational telemetry from meta-processes -- rumination loops,
              self-improvement notes, problem analyses, trigger observations,
              procedural questions, agent self-assessments. NOT facts about
              the world; meta-commentary about agent work.
              Examples: "IMPROVEMENT: Integrate user feedback mechanisms",
              "problem_5: agent overlooked validation", "trigger_42: when
              processing malformed input", "agent_self_assessment: 3 mistakes
              noted today", "question_61: how to handle image generation".
              These are PROCESS-grade observations.
- "meta"     : Rare. Operational state about the agent system itself --
              timestamps, configuration changes, deployment events. Reserved
              for system bookkeeping (e.g. "session_20260604 started",
              "model switched to minimax-m3"). Not normally user-typed.

When in doubt: is this about the USER/WORLD (memory) or about the AGENT'S
PROCESS (ruminate)? Almost everything agent-generated from rumination loops
is ruminate.

When the user message includes "Existing MPMF nodes:" with a list of similar
existing memory nodes, you MUST analyze the new fact in relation to them and
fill in the 3 relational fields:

1. "relates_to": list of names of existing nodes that this fact is connected to
   (0-3 names max; leave empty array if none clearly relate).

2. "relation_semantics": choose one:
   - "confirm"    — this fact reinforces / agrees with existing knowledge
   - "contradict" — this fact contradicts or conflicts with existing knowledge
   - "extend"     — this fact adds detail to or builds on existing knowledge
   - "supersede"  — this fact replaces / makes obsolete existing knowledge
   - "none"       — no clear relationship to any existing node

3. "preventive_tool_or_pattern": if this fact describes an issue, bug, or
   suboptimal outcome that could have been prevented, name the tool, skill,
   pattern, or workflow that would have caught or prevented it. Empty string
   if not applicable (e.g. positive facts, general knowledge).

If no existing nodes are provided, set "relates_to": [] and
"relation_semantics": "none" and "preventive_tool_or_pattern": "".

Other guidelines:
- associative tags: lowercase, generic enough to cluster (e.g. "user", "preference",
  "communication", "model", "cost", "project", "person" -- AVOID month names and
  sentence-initial nouns)
- causal: ONLY if the fact states cause->effect; otherwise empty string
- temporal: ONLY if the fact is time-bound; otherwise empty string
- functional: ONE category from the list
- priority: 5 for personal/user-defining facts, 4 for stable preferences, 3 for
  general facts, 2 for transient details, 1 for noise

Example output:
{
  "summary": "User D is direct, high-quality, anti-alignment, English only.",
  "namespace": "memory",
  "wiki": {"subject": "user", "name": "D", "language": "english"},
  "planes": {
    "associative": ["user", "preference", "communication", "language"],
    "causal": "",
    "temporal": "",
    "functional": "person",
    "priority": 5
  },
  "relates_to": [],
  "relation_semantics": "none",
  "preventive_tool_or_pattern": ""
}

Example ruminate output:
{
  "summary": "Agent should integrate user feedback mechanisms into improvement loop.",
  "namespace": "ruminate",
  "wiki": {"process": "improvement", "topic": "feedback"},
  "planes": {
    "associative": ["process", "improvement", "feedback", "agent"],
    "causal": "",
    "temporal": "",
    "functional": "other",
    "priority": 2
  },
  "relates_to": [],
  "relation_semantics": "none",
  "preventive_tool_or_pattern": ""
}"""


def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s


def _format_existing_context(nodes: Optional[List[Dict[str, str]]]) -> str:
    """Format existing similar nodes for the extraction prompt."""
    if not nodes:
        return ""
    parts = []
    for i, n in enumerate(nodes, 1):
        name = n.get("name", "?")
        summary = n.get("summary", n.get("text", ""))[:120]
        parts.append(f"  {i}. name={name!r}, summary={summary!r}")
    return "\n".join(parts)


class _TruncatedRetry(Exception):
    """finish_reason=='length' on the FIRST attempt — bump the budget and retry once."""


class _TruncatedFatal(Exception):
    """finish_reason=='length' even after the budget bump — stop retrying.

    Repeating the same capped request at temperature 0 can only fail the same
    way, so we do NOT loop on it. The raw node is already persisted by store().
    """


def _dump_raw(dump_dir: Optional[str], data: dict, tag: str) -> None:
    """Persist a raw provider response for forensics. Never raises.

    Used to answer 'was message.reasoning populated and what did content hold?'
    without re-running a failing call by hand.
    """
    if not dump_dir:
        return
    try:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        fn = Path(dump_dir) / f"extract_{tag}_{int(time.time() * 1000)}.json"
        fn.write_text(json.dumps(data, indent=2))
        print(f"  [extract-dump] raw response -> {fn}", file=sys.stderr)
    except Exception as e:  # pragma: no cover - diagnostics must never break writes
        print(f"  [extract-dump] could not write dump: {e}", file=sys.stderr)


def call_extractor(fact_text: str, model: str = None, api_key: str = None,
                   base_url: str = None, timeout: int = None,
                   max_attempts: int = 3,
                   existing_context: Optional[List[Dict[str, str]]] = None,
                   max_tokens: int = None,
                   dump_dir: Optional[str] = None) -> dict:
    """Call the extraction LLM to produce a 5-plane structured payload.

    Args:
        existing_context: top-5 similar existing nodes, for relational analysis.
        max_tokens: output budget; defaults to EXTRACT_MAX_TOKENS.
        dump_dir: when set, raw provider responses from failed calls are written here.

    Failure semantics (rewritten 2026-09-27):
        * finish_reason == "length"  → raise _TruncatedRetry, double the budget,
          retry ONCE. If it is still truncated, raise _TruncatedFatal (no
          identical re-request — at temperature 0 that can only fail again).
        * empty/garbled content      → ValueError, retried with backoff.
        * every failure is reported to llm_call_guard, so it lands in
          the health dir instead of being swallowed.
    """
    model = model or EXTRACT_MODEL
    base_url = base_url or LLM_BASE_URL
    timeout = timeout or EXTRACT_TIMEOUT
    if not (api_key or _api_key()):
        raise RuntimeError("OPENROUTER_API_KEY not set in env")
    # CRITICAL: strip leading/trailing whitespace from the env var.
    # Some shell startup paths (notably .bashrc + .profile interaction)
    # leave a literal \n in front of the value, which the HTTP server
    # rejects with "Invalid leading whitespace, reserved character(s),
    # or return character(s) in header value" — see rumination 2026-06-09.
    ak = (api_key or _api_key()).strip()
    if not ak:
        raise RuntimeError("OPENROUTER_API_KEY blank after strip")

    budget = int(max_tokens or EXTRACT_MAX_TOKENS)
    bumps = 0
    last_err = None
    for attempt in range(max_attempts):
        wait_override = None
        try:
            # guarded() records the failure *and* re-raises, so our own retry
            # loop still sees it. This is the A5.1 wiring that was missing.
            with guarded("extraction"):
                resp = requests.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {ak}",
                             "Content-Type": "application/json",
                             "Connection": "close"},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": EXTRACTION_SYSTEM},
                            {"role": "user", "content": (
                                f"Fact: {fact_text}\n\n"
                                + (f"Existing MPMF nodes:\n{_format_existing_context(existing_context)}\n\n"
                                   if existing_context else "")
                                + "Output JSON:"
                            )},
                        ],
                        "temperature": 0.0,
                        "max_tokens": budget,
                        # EXTRACTION-ONLY: disable the model's reasoning channel.
                        # The 5-plane extractor returns a small fixed-shape JSON;
                        # a reasoning trace only burns the output budget (on
                        # deepseek-v4.x it consumes ~100% of max_tokens and can
                        # return finish_reason=length with content=None).
                        # This is set INLINE here on purpose: it must never move
                        # into a shared request-builder/helper/env var, because
                        # the agent's normal chat turns must KEEP reasoning
                        # (that trace is what distill_thinking harvests into the
                        # thinking store).
                        "reasoning": {"enabled": False},
                    },
                    timeout=timeout,
                )
                # Rate limit: honour Retry-After, then let the outer handler wait.
                if resp.status_code == 429:
                    try:
                        wait_override = int(resp.headers.get("Retry-After", "10"))
                    except (ValueError, TypeError):
                        wait_override = 10
                    raise RuntimeError(f"429 rate limited (retry-after={wait_override}s)")
                resp.raise_for_status()
                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                finish_reason = choice.get("finish_reason")
                msg = choice.get("message") or {}
                content = (msg.get("content") or "").strip()
                reasoning = msg.get("reasoning") or msg.get("reasoning_content")

                # ---- finish_reason handling (the missing piece) ----
                if finish_reason == "length":
                    _dump_raw(dump_dir, data, "length")
                    if bumps < EXTRACT_BUDGET_BUMPS:
                        bumps += 1
                        new_budget = budget * 2
                        print(f"  Extraction truncated at {budget} tokens "
                              f"(reasoning={'yes' if reasoning else 'no'}); "
                              f"retrying once at {new_budget}", file=sys.stderr)
                        budget = new_budget
                        raise _TruncatedRetry(f"length at {budget // 2} tokens")
                    raise _TruncatedFatal(
                        f"still truncated at {budget} tokens after budget bump"
                    )

                if not content:
                    _dump_raw(dump_dir, data, "empty")
                    raise ValueError(
                        f"empty content in response (finish_reason={finish_reason!r}, "
                        f"reasoning={'yes' if reasoning else 'no'}): {choice}"
                    )
                parsed = json.loads(_strip_fences(content))
                parsed["_usage"] = data.get("usage", {})
                parsed["_model"] = model
                return parsed
        except _TruncatedRetry as e:
            # Budget was widened; go straight to the next attempt (no backoff).
            last_err = e
            continue
        except _TruncatedFatal as e:
            # Truncation survived the budget bump — an identical retry cannot help.
            last_err = e
            break
        except Exception as e:
            last_err = e
            if attempt < max_attempts - 1:
                wait = wait_override if wait_override is not None else min(2 ** attempt * 2, 16)
                print(f"  Extraction error (attempt {attempt+1}/{max_attempts}): {e}, retrying in {wait}s",
                      file=sys.stderr)
                time.sleep(wait)
    raise RuntimeError(f"extraction failed after {max_attempts} attempts: {last_err}")


def compute_plane_flags(planes: dict) -> int:
    """Map populated planes to MPMF plane_flags bitmask."""
    flags = 0
    assoc = planes.get("associative")
    if isinstance(assoc, list) and len(assoc) > 0:
        flags |= Plane.ASSOCIATIVE.value
    if isinstance(planes.get("causal"), str) and planes["causal"].strip():
        flags |= Plane.CAUSAL.value
    if isinstance(planes.get("temporal"), str) and planes["temporal"].strip():
        flags |= Plane.TEMPORAL.value
    if isinstance(planes.get("functional"), str) and planes["functional"].strip():
        flags |= Plane.FUNCTIONAL.value
    flags |= Plane.PRIORITY.value  # always set, used as 0.10 importance prior
    return flags


def _query_similar_nodes(text: str, top_k: int = 5) -> List[Dict[str, str]]:
    """Query top-k similar existing MPMF nodes by cosine similarity on embeddings.

    Opens a read-only handle to the live store, embeds the query text, computes
    cosine similarity against all existing node vectors, and returns the top-k
    matches with 'name', 'summary', 'text' keys.

    Returns empty list if the store doesn't exist, has no nodes, or errors occur.
    """
    if not os.path.exists(PATH):
        return []
    try:
        m = _open()
        mem = m.mem
        node_ids = list(mem.all_node_ids() if hasattr(mem, "all_node_ids") else [])
        if not node_ids:
            return []

        # Embed the query once
        q_vec = _embed_floats_local(text)
        if not q_vec:
            return []

        scored = []
        for nid in node_ids:
            try:
                vec = mem.get_vector(nid)
                if not vec or len(vec) != VECTOR_DIM:
                    continue
                # Cosine similarity = dot product for normalized vectors
                dot = sum(a * b for a, b in zip(q_vec, vec))
                scored.append((dot, nid))
            except Exception:
                continue

        if not scored:
            return []

        scored.sort(key=lambda x: -x[0])
        results = []
        for dot, nid in scored[:top_k]:
            name = mem.get_name(nid) or ""
            summary = mem.get_summary(nid) or ""
            text_val = mem.get_text(nid) or summary
            results.append({
                "name": name,
                "summary": summary[:200],
                "text": text_val[:500],
                "score": round(dot, 4),
            })
        return results
    except Exception as e:
        print(f"[query_similar_nodes] WARN: {e}", file=sys.stderr)
        return []


# ============== EMBEDDING (LOCAL M-XBAI + REQUESTS, HARD TIMEOUTS) ==============
_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")


# --- Embed input truncation + context-length safety (items 1+2, 2026-10-10)
# mxbai-embed-large has a 512-token (~2048-char) window. prefetch embeds the
# RAW user message, so a turn longer than the window makes Ollama return 500
# ("input length exceeds the context length"); the old retry loop then burned
# 1+2+4+8 s of backoff, blew the 8 s prefetch join, and the turn got NO memory.
# Cap the input here (single choke point for every caller: recall, cognition,
# dedup, store/pack). Head + tail keeps a leading ask AND a trailing ask.
# Text <= threshold is returned UNCHANGED (byte-identical embedding).
_EMBED_TRUNC_THRESHOLD_DEFAULT = 1200
_EMBED_CTX_DEADLINE_S = 2.0


def _embed_query_cap() -> int:
    """Recall-QUERY embed cap (chars) from recall.config.json `embed_query_cap_chars`.

    Default 1200 when the key is absent. Applies ONLY to recall-query embedding
    (embed_query() and the retrieval adapter's `kind="query"` path). Document
    embeds (store / _pack / check_dedup / enrichment) never read this — they use
    the fixed 1200 default so stored vectors are byte-identical to before this
    flag existed.
    """
    try:
        with open(_RECALL_CONFIG_PATH) as f:
            cap = int(json.load(f).get("embed_query_cap_chars"))
        return cap if cap > 0 else _EMBED_TRUNC_THRESHOLD_DEFAULT
    except Exception:
        return _EMBED_TRUNC_THRESHOLD_DEFAULT


def _truncate_for_embed(text: str, cap: Optional[int] = None) -> str:
    """Head+tail cap for the embedding model's context window.

    `cap` defaults to the FIXED document cap (1200) — exactly the pre-flag
    behaviour. Callers on the recall-query path pass `cap=_embed_query_cap()`
    to honour the configurable flag. Text <= cap passes through UNCHANGED;
    longer text is head+tail capped to `cap // 2` chars on each side. The
    plugin's `_truncate_query` uses the SAME key + formula, so the capped query
    is byte-identical on both paths (shared vector cache still hits).
    """
    if cap is None:
        cap = _EMBED_TRUNC_THRESHOLD_DEFAULT
    if not text or len(text) <= cap:
        return text
    half = cap // 2
    return text[:half] + " " + text[-half:]


def _is_ctx_len_error(msg) -> bool:
    s = str(msg or "").lower()
    return "context length" in s or "input length exceeds" in s


def _embed_http_once(text: str) -> List[float]:
    """One Ollama embed round-trip: /api/embed, then legacy /api/embeddings.

    Raises RuntimeError carrying the response body so the caller can detect
    the deterministic context-length error (item 2).
    """
    body = ""
    try:
        resp = requests.post(
            f"{_OLLAMA_URL}/api/embed",
            headers={"Content-Type": "application/json", "Connection": "close"},
            json={"model": EMBED_MODEL, "input": text},
            timeout=EMBED_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.json()["embeddings"][0]
        body = resp.text
    except Exception as e:  # connection error etc. -> try legacy endpoint
        if _is_ctx_len_error(e):
            raise RuntimeError(f"/api/embed {e}")
        body = str(e)
    if _is_ctx_len_error(body):
        raise RuntimeError(f"/api/embed ctx-len error: {body[:300]}")
    resp = requests.post(
        f"{_OLLAMA_URL}/api/embeddings",
        headers={"Content-Type": "application/json", "Connection": "close"},
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=EMBED_TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"/api/embeddings returned {resp.status_code}: {resp.text[:300]}"
        )
    return resp.json()["embedding"]


def _embed_floats_local(text: str, dim: int = VECTOR_DIM, cap: Optional[int] = None) -> List[float]:
    """Embed via local Ollama. Hard timeouts via requests.

    Item 1 (2026-10-10): input is head+tail truncated to fit mxbai's
    512-token window (see _truncate_for_embed).
    Item 2: on the deterministic 'input length exceeds the context length'
    error we do NOT backoff-retry the SAME input — we halve it and retry
    ONCE, then raise, spending at most ~_EMBED_CTX_DEADLINE_S seconds.
    Transient errors keep the bounded exponential backoff.

    Canonical embedder per MPMF+ v1.0 spec A1: mxbai-embed-large @ 1024d.
    A dim mismatch is a LOUD error, never a silent pad (spec A1.2).
    """
    if not text:
        return [0.0] * dim
    cur = _truncate_for_embed(text, cap)
    last = None
    ctx_retried = False
    ctx_deadline = time.time() + _EMBED_CTX_DEADLINE_S
    for attempt in range(4):
        try:
            emb = _embed_http_once(cur)
        except Exception as e:
            last = e
            if _is_ctx_len_error(e):
                # Deterministic: retrying the SAME input can never succeed.
                if not ctx_retried and len(cur) > 64:
                    ctx_retried = True
                    cur = cur[: max(64, len(cur) // 2)]  # halve, retry once
                    if time.time() >= ctx_deadline:
                        raise RuntimeError(
                            f"local embed exceeded {_EMBED_CTX_DEADLINE_S}s "
                            f"deadline on context-length error: {e}"
                        )
                    continue
                raise RuntimeError(
                    f"local embed failed (input exceeds context length): {e}"
                )
            time.sleep(min(2 ** attempt, 8))
            continue
        # Dim check is OUTSIDE the retry path: LOUD, never retried/padded.
        if len(emb) != dim:
            raise RuntimeError(
                f"DimensionMismatchError: embedder={EMBED_MODEL!r} produced "
                f"dim={len(emb)} but store requires dim={dim}. "
                f"Refusing to silently pad/truncate. Either change the embedder "
                f"to one that produces {dim}d, or create a {len(emb)}d store."
            )
        norm = math.sqrt(sum(x * x for x in emb)) or 1.0
        return [x / norm for x in emb]
    raise RuntimeError(f"local embed failed after 4 attempts: {last}")


def _pack(text: str) -> bytes:
    """Packed little-endian float32 bytes for MPMF's vector field."""
    floats = _embed_floats_local(text)
    return struct.pack(f"<{VECTOR_DIM}f", *floats)


# ============== RERANKER (LOCAL CROSS-ENCODER) ==============
_RERANKER = None
#: Lock for lazy-init of the local cross-encoder reranker. Used inside
#: _get_reranker() to prevent two threads from initialising the model
#: at the same time (slow first call, idempotent after).
_RERANKER_LOCK = threading.Lock()
#: Module-level lock for the entire store() function. The dedup check
#: and the node create were previously a TOCTOU race: two threads could
#: both pass check_dedup (both see "no existing node") and both call
#: m.remember() with the same name → 2 nodes with identical text+name.
#: This lock makes the check-then-create atomic across all writers
#: (distiller, sync_turn, rumination, manual store calls).
#: Throughput cost: one store() at a time, ~50-200ms hold. Acceptable
#: for an agent's write rate (typically << 10 writes/sec).
_STORE_LOCK = threading.Lock()


def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        with _RERANKER_LOCK:
            if _RERANKER is None:
                from sentence_transformers import CrossEncoder
                _RERANKER = CrossEncoder(RERANKER_MODEL, max_length=512, device="cpu")
    return _RERANKER


#: Pairs per rerank batch (2026-10-21 r46d). Measured on the 46c mixed-length
#: cases: 20 short 159 ms @B8 / 175 @B4 / 168 @single; 19 short + 1 long 294 @B4
#: / 283 @B8 / 806 @single; 15 short + 5 long 572 @B4 / 699 @B8 / 1752 @single;
#: 20 long 1274 @B4 / 1343 @B8 / 1770 @single. B4 wins or ties every cell, at
#: the cost of one extra predict call per four pairs. On 100 real queries from
#: recall_trace.log the switch is 129.1 s -> 51.2 s (2.52x) end to end.
RERANK_BUCKET = 4


def _rss_kb() -> Optional[int]:
    """Resident set size of THIS process, kB. Log-only; never raises."""
    try:
        with open("/proc/self/statm") as _f:
            _resident = int(_f.read().split()[1])
        return int(_resident * (os.sysconf("SC_PAGE_SIZE") // 1024))
    except Exception:
        return None


def rerank(query: str, candidates: List[dict], top_k: int = DEFAULT_K) -> List[dict]:
    """Rerank candidates by cross-encoder (query, text) score. Returns top_k."""
    if not candidates:
        return []
    model = _get_reranker()
    pairs = [(query, c.get("text", "")) for c in candidates]
    # Length-homogeneous buckets (2026-10-21 r46d): the cross-encoder pads every
    # batch to its LONGEST pair, so one long candidate made 20 short pairs cost
    # as much as 20 long ones (measured: 184 ms -> 894 ms -> 1740 ms). Sorting
    # by candidate length and scoring in small buckets keeps padding local.
    # Scores are unchanged: padded positions are attention-masked, so a pair's
    # score does not depend on its neighbours (verified to 1e-3 over 100 real
    # queries in gate_eval/r46d_p_b.py).
    scores = [0.0] * len(pairs)
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))
    for _s in range(0, len(order), RERANK_BUCKET):
        _idxs = order[_s:_s + RERANK_BUCKET]
        _vals = model.predict([pairs[i] for i in _idxs], show_progress_bar=False)
        for _i, _v in zip(_idxs, _vals):
            scores[_i] = float(_v)
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: -float(x[1]))
    out = []
    for i, (cand, s) in enumerate(scored[:top_k], start=1):
        new = dict(cand)
        new["rank"] = i
        new["orig_rank"] = cand.get("rank", i)
        new["reranker_score"] = float(s)
        out.append(new)
    return out


# ============== EDGE DERIVATION (5 GROUNDED RULES) ==============
# Inline copy of edge_derivation.py's core functions, simplified for live use.
# (We re-derive edges on every ingest because extractions may have changed.)

GENERIC_TAGS = {"user", "preference", "fact", "the user"}
PREV_RE = re.compile(r"\b(previously|used to|no longer|since|formerly|stopped|switched|left|moved)\b", re.I)
PROPER_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
PROPER_STOP = {"The", "User", "After", "That", "This", "She", "Her", "His", "They",
               "When", "While", "Their", "It", "As", "In", "On", "At", "A", "An",
               "January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December",
               "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}


# ============== NAMESPACE CLASSIFICATION (regex-based, no LLM) ==============
# Three namespaces:
#   - "memory"  : curated knowledge about user, world, projects, people, tools
#   - "ruminate" : operational telemetry from rumination/self-improvement loops
#   - "meta"     : rare; system bookkeeping
#
# For the live store, almost all auto-generated content is ruminate. The few
# curated memory facts are the ones a human would want to remember.
DEFAULT_NAMESPACE = "memory"

# Compiled patterns for classification
_RUMINATE_PATTERNS = [
    re.compile(r"^IMPROVEMENT\b", re.I),
    re.compile(r"^fact_\d+\b"),
    re.compile(r"^problem_\d+\b"),
    re.compile(r"^trigger_\d+\b"),
    re.compile(r"^question_\d+\b"),
    re.compile(r"^task_\d+\b"),
    re.compile(r"^(delivery|framing|gateway|character|data|rumination|transparency|atemahistory|build_)\w+"),
    re.compile(r"^(next_focus_area|recommendation_|work_summary|debug_key|validation_target|crash_loop|restart_|workload_|agent_|inconsistent_|expected_|nonstandard_|realtime_|signal_|telegram_|final_output_|input_|console_|layout_|sprite_|frame_|animation_|openrouter_)"),
    re.compile(r"^(agent_success|agent_self|agent_effectiveness|recommendation_satisfaction|agent_performance)"),
]

_MEMORY_PATTERNS = [
    re.compile(r"^user_"),
    re.compile(r"^mpmf_"),
    re.compile(r"^cost_"),
    re.compile(r"^comms_"),
    re.compile(r"^openclaw_"),
    re.compile(r"^ballerina_"),
    re.compile(r"^multi_agent_"),
    re.compile(r"^image_gen(?:_|\b)"),   # image_gen_prefs; NOT image_generation_verification
    re.compile(r"^5_planes_"),
    re.compile(r"^(?:agent|assistant)$"),
]


_TRANSCRIPT_RE = re.compile(r"^(turn_|sess_)")
_RUMINATE_REVERSED_RE = re.compile(
    r"^(cog_rum_|cog_prov_|cog_hyp_|ruminate_|problem_|trigger_|question_|fact_)")
PROTECTED_NS = ("cognition", "meta")


def classify_namespace(name: str, text: str = "", current: str = None) -> str:
    """INVERTED classifier (revised): memory is the default, ruminate is opt-in.

    Order:
      1. If the node already sits in cognition/meta (pass `current`), keep it.
      2. turn_/sess_* -> "transcript" (its own class; kept out of default recall)
      3. Positive ruminate match (name prefix, or "IMPROVEMENT:" text) -> ruminate
      4. Everything else -> "memory"
    """
    if current in PROTECTED_NS:
        return current
    n = (name or "").strip()
    t = (text or "").lstrip()
    if _TRANSCRIPT_RE.match(n):
        return "transcript"
    if _RUMINATE_REVERSED_RE.match(n) or t.startswith("IMPROVEMENT:"):
        return "ruminate"
    return "memory"


# --- Edge density controls (moved out of the one-off rebuild script) --------
# A tag/proper-noun carried by more than EDGE_TAG_SKIP_PCT% of nodes is too
# ubiquitous to be signal ("process", "agent", "project", ...); it is excluded
# from the shared-tag / coref rules. After derivation each node keeps at most
# EDGE_DEGREE_CAP edges (greedy, by weight) so graph traversal stays fast.
EDGE_TAG_SKIP_PCT = 5.0
EDGE_DEGREE_CAP = 20


def _norm_tags(assoc) -> set:
    out = set()
    for t in assoc or []:
        if not isinstance(t, str):
            continue
        t = t.lower().strip()
        if t and t not in GENERIC_TAGS and len(t) >= 3:
            out.add(t)
    return out


def _proper_nouns(text: str) -> set:
    return {m for m in PROPER_RE.findall(text or "") if m not in PROPER_STOP}


def _derive_edges(extractions: Dict[str, dict], facts_by_id: Dict[str, str],
                  assoc_min_shared: int = 2,
                  tag_skip_pct: Optional[float] = EDGE_TAG_SKIP_PCT,
                  degree_cap: Optional[int] = EDGE_DEGREE_CAP) -> tuple:
    """5 grounded rules. Returns (edges_list, counts_dict).

    Density controls (live by default): tags/proper-nouns in > tag_skip_pct%
    of nodes are dropped from the rules, and each node is capped at
    degree_cap edges (greedy by weight). Pass None to disable either.
    """
    feat = {}
    for fid, e in extractions.items():
        pl = e.get("planes", {})
        text = facts_by_id.get(fid, e.get("summary", ""))
        feat[fid] = {
            "tags": _norm_tags(pl.get("associative", [])),
            "causal": (pl.get("causal") or "").strip(),
            "temporal": (pl.get("temporal") or "").strip(),
            "functional": (pl.get("functional") or "").strip().lower(),
            "proper": _proper_nouns(text),
            "text": text,
        }

    # Density filter: drop tags/proper-nouns that appear in too many nodes.
    if tag_skip_pct is not None and feat:
        from collections import Counter as _C
        _td, _pd = _C(), _C()
        for _f in feat.values():
            for _t in _f["tags"]:
                _td[_t] += 1
            for _p in _f["proper"]:
                _pd[_p] += 1
        _lim = tag_skip_pct / 100.0 * len(feat)
        for _f in feat.values():
            _f["tags"] = {t for t in _f["tags"] if _td[t] <= _lim}
            _f["proper"] = {p for p in _f["proper"] if _pd[p] <= _lim}

    edges = []
    counts = {"associative": 0, "entity_coref": 0, "temporal": 0, "supersession": 0, "causal_xfact": 0}
    fids = list(feat.keys())

    # 1. Associative (>=N shared tags)
    for i, a in enumerate(fids):
        for b in fids[i + 1:]:
            shared = feat[a]["tags"] & feat[b]["tags"]
            if len(shared) >= assoc_min_shared:
                edges.append({"src": a, "dst": b, "plane": "associative",
                              "label": f"shared_tags:{','.join(sorted(shared))[:50]}", "weight": 0.5})
                counts["associative"] += 1

    # 2. Entity coreference (shared proper noun)
    for i, a in enumerate(fids):
        for b in fids[i + 1:]:
            shared = feat[a]["proper"] & feat[b]["proper"]
            if shared:
                ents = ','.join(sorted(shared))[:50]
                edges.append({"src": a, "dst": b, "plane": "associative",
                              "label": f"coref:{ents}", "weight": 0.7})
                counts["entity_coref"] += 1

    # 3. Temporal (both temporal + >=2 shared tags)
    for i, a in enumerate(fids):
        if not feat[a]["temporal"]:
            continue
        for b in fids[i + 1:]:
            if not feat[b]["temporal"]:
                continue
            shared = feat[a]["tags"] & feat[b]["tags"]
            if len(shared) >= 2:
                edges.append({"src": a, "dst": b, "plane": "temporal",
                              "label": "co_temporal", "weight": 0.5})
                counts["temporal"] += 1

    # 4. Supersession: prev/new same func-cat + shared topic (≥1 tag overlap)
    for i, a in enumerate(fids):
        if not PREV_RE.search(feat[a]["text"]):
            continue
        for b in fids:
            if a == b:
                continue
            if not feat[a]["functional"] or feat[a]["functional"] != feat[b]["functional"]:
                continue
            if PREV_RE.search(feat[b]["text"]):
                continue  # both are "previous" — skip
            shared = feat[a]["tags"] & feat[b]["tags"]
            if shared:
                edges.append({"src": a, "dst": b, "plane": "causal",
                              "label": "supersedes", "weight": 0.9})
                counts["supersession"] += 1

    # 5. Causal cross-fact: causal text mentions another's proper noun
    for i, a in enumerate(fids):
        if not feat[a]["causal"]:
            continue
        ctext = feat[a]["causal"]
        for b in fids:
            if a == b:
                continue
            for ent in feat[b]["proper"]:
                if ent in ctext:
                    edges.append({"src": a, "dst": b, "plane": "causal",
                                  "label": f"causes:{ent}", "weight": 0.6})
                    counts["causal_xfact"] += 1
                    break

    # Degree cap: keep at most `degree_cap` edges per node, greedily by weight
    # (all planes compete for the budget). Keeps the graph traversable.
    if degree_cap is not None:
        from collections import Counter as _C
        edges.sort(key=lambda e: -e["weight"])
        _deg = _C()
        _kept = []
        for e in edges:
            if _deg[e["src"]] >= degree_cap or _deg[e["dst"]] >= degree_cap:
                continue
            _kept.append(e)
            _deg[e["src"]] += 1
            _deg[e["dst"]] += 1
        edges = _kept
        counts = {}
        for e in edges:
            counts[e["plane"]] = counts.get(e["plane"], 0) + 1

    return edges, counts


# ============== CORE MEMORY API ==============
#: Read-snapshot cache for the live store, keyed by (path, mtime_ns, size).
#: recall()/get()/stats() hit this instead of re-loading the ~37 MB binary on
#: every call. Writers must NEVER take this shared instance (they would mutate
#: the snapshot under a concurrent reader) — they use _open_rw(), and the cache
#: is invalidated after their save so the next read reloads committed state.
_OPEN_CACHE: Dict[str, tuple] = {}
_OPEN_CACHE_LOCK = threading.Lock()
#: Round 45: True when the last _open() was served from the memoized snapshot.
_LAST_OPEN_HIT = False

#: Per-phase wall times (seconds) of the most recent recall(), read by the
#: prefetch logging layer. Best-effort observability only.
_LAST_TIMINGS: Dict[str, float] = {}


def _store_key(path: str):
    """(path, mtime_ns, size) identity of the store file, or None if missing."""
    try:
        st = os.stat(path)
        return (path, st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _invalidate_open_cache() -> None:
    """Drop the memoized read snapshot so the next _open() reloads from disk."""
    with _OPEN_CACHE_LOCK:
        _OPEN_CACHE.pop(PATH, None)


def _open() -> MPMFMemory:
    """Return a memoized READ snapshot of the live store.

    Safe to share across threads: readers never mutate it, and the key includes
    mtime_ns+size, so any save (ours or another process's) transparently
    invalidates it. Loading happens outside the lock; a concurrent duplicate
    load is harmless (last writer wins), and the store is written atomically
    (tempfile + fsync + rename), so a reader can never see a half-written file.
    """
    global _LAST_OPEN_HIT
    key = _store_key(PATH)
    if key is None:
        # r59d: never silently create an empty store at the configured path, and
        # never fall back to a different file. KeyError-style failure, loudly.
        raise StoreNotFoundError(
            f"MPMF+ store not found or unreadable: {PATH!r} "
            f"(set MPMF_STORE_PATH or MPMF_HOME)")
    if key is not None:
        with _OPEN_CACHE_LOCK:
            ent = _OPEN_CACHE.get(PATH)
            if ent is not None and ent[0] == key:
                _LAST_OPEN_HIT = True
                return ent[1]
    _LAST_OPEN_HIT = False
    # Single-flight (round 46): a turn arriving while the pre-warm thread is
    # loading waits on the same lock and then finds the cache populated, instead
    # of starting a second 39 MB load.
    with _LOAD_LOCK:
        key2 = _store_key(PATH)
        if key2 is not None:
            with _OPEN_CACHE_LOCK:
                ent = _OPEN_CACHE.get(PATH)
                if ent is not None and ent[0] == key2:
                    _LAST_OPEN_HIT = True
                    if _PREWARM["thread"] is None:
                        _start_prewarm()
                    return ent[1]
        try:
            inst = MPMFMemory(PATH, vector_dimensions=VECTOR_DIM, embed_model=EMBED_MODEL)
        except Exception as e:
            # r59d: a corrupt/truncated/unreadable image is a loud failure, not a
            # fresh empty store and not a different file.
            raise StoreNotFoundError(
                f"MPMF+ store could not be opened: {PATH!r}: {type(e).__name__}: {e}") from e
        if key2 is not None:
            with _OPEN_CACHE_LOCK:
                _OPEN_CACHE[PATH] = (key2, inst)
    if _PREWARM["thread"] is None:
        _start_prewarm()
    return inst


def _open_rw() -> MPMFMemory:
    """A FRESH, private instance for writers. NEVER the shared read snapshot."""
    # r60: honour store_mode. attach -> the store must already exist (raises
    # StoreNotFoundError, never creates); create -> initialise it explicitly,
    # once, with a line on stderr instead of letting the core auto-create it.
    settings = store_settings()
    _store.attach_guard(settings)
    if not _store.exists(PATH):
        try:
            _store.create_empty(settings, vector_dimensions=VECTOR_DIM)
        except FileExistsError:
            pass  # another process created it between the check and the write
    return MPMFMemory(PATH, vector_dimensions=VECTOR_DIM, embed_model=EMBED_MODEL)


@contextlib.contextmanager
def _write_txn_open(timeout: Optional[float] = None):
    """Open a private writer INSIDE an exclusive cross-process transaction.

    Round 48c: the store is a whole-file image. Opening, mutating and saving
    without a lock lets a second writer's nodes be destroyed. Under this
    context the instance is reloaded if the file moved on, the mutation runs
    while the flock is held, and the save is atomic.

    r60: the default wait is 60 s, not 10 s, overridable with
    $MPMF_STORE_LOCK_TIMEOUT. A single store() call holds the lock across an
    Ollama embed round-trip, so a second agent writing 25 facts in a row can
    occupy it for tens of seconds; a 10 s wait turned that ordinary contention
    into a LockTimeoutError (observed: 49/50 nodes in the r60 stress run).
    Waiting longer is NOT a silent fallback — if the lock is genuinely stuck,
    the same LockTimeoutError is still raised, just later.
    """
    if timeout is None:
        try:
            timeout = float(os.environ.get("MPMF_STORE_LOCK_TIMEOUT", "60"))
        except ValueError:
            timeout = 60.0
    m = _open_rw()
    txn = getattr(getattr(m, "mem", None), "write_txn", None)
    if txn is None:  # runtime predates Round 48c
        yield m
        return
    with txn(timeout=timeout):
        yield m
    _invalidate_open_cache()


# ---- round 46: keep the read snapshot warm off the turn path ---------------
# The per-turn transcript write (plugin sync_turn -> store() -> _open_rw().save())
# drops the memoized snapshot from INSIDE this process, so every turn paid a
# full reload (~390 ms) plus a first-query penalty (~460 ms). Those writes use a
# private handle on purpose — publishing a writer instance as the read snapshot
# would break the "readers never see a mutated snapshot" invariant — so the fix
# is a pre-warm thread: watch stat(), and once the file has been quiet for
# _PREWARM_STABLE_S, reload and run one warm query. Kill switch: recall.config
# "prewarm_store" (default true), read on every poll.
_LOAD_LOCK = threading.Lock()
_PREWARM: Dict[str, Any] = {"thread": None, "stop": False, "warm_ms": None,
                            "warm_count": 0, "last_err_ts": 0.0, "vec": None}
_PREWARM_POLL_S = 1.0
_PREWARM_STABLE_S = 1.5


def _prewarm_enabled() -> bool:
    try:
        v = _recall_flags().get("prewarm_store", True)
        return True if v is None else bool(v)
    except Exception:
        return True


def _prewarm_note_error(exc) -> None:
    """Rate-limited (60 s) failure note; never raises."""
    try:
        now = time.time()
        if now - float(_PREWARM.get("last_err_ts") or 0.0) < 60:
            return
        _PREWARM["last_err_ts"] = now
        with open(str(MPMF_HEALTH / "failures.log"), "a") as f:
            f.write("%s prewarm: %r\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), exc))
    except Exception:
        pass


def _prewarm_query_vec():
    """One stored node vector, so the warm query costs no embedding."""
    v = _PREWARM.get("vec")
    if v is not None:
        return v
    try:
        inst = _open()
        for nid in list(inst.mem.all_node_ids())[:800]:
            try:
                vv = inst.mem.get_node_vector(nid)
            except Exception:
                continue
            if vv:
                _PREWARM["vec"] = list(vv) if not isinstance(vv, list) else vv
                return _PREWARM["vec"]
    except Exception as e:
        _prewarm_note_error(e)
    return None


def _mem_available_kb() -> Optional[int]:
    """MemAvailable from /proc/meminfo, kB. None if unreadable."""
    try:
        with open("/proc/meminfo") as _f:
            for _ln in _f:
                if _ln.startswith("MemAvailable"):
                    return int(_ln.split()[1])
    except Exception:
        return None
    return None


def _prewarm_guard_cfg():
    """(mode, margin_kb) for the RSS governor, read per call."""
    try:
        _f = _recall_flags() or {}
    except Exception:
        _f = {}
    _mode = str(_f.get("prewarm_rss_guard", "auto")).lower()
    try:
        _mb = float(_f.get("prewarm_rss_margin_mb", 400))
    except Exception:
        _mb = 400.0
    return _mode, _mb * 1024.0


def _prewarm_trip_state() -> Optional[str]:
    _s = _PREWARM.get("state")
    return _s if isinstance(_s, str) and _s.startswith("tripped:") else None


def _prewarm_state_str() -> str:
    _t = _prewarm_trip_state()
    if _t:
        return _t
    try:
        if not _prewarm_enabled():
            return "off"
    except Exception:
        return "off"
    return "on"


def _prewarm_trip(kind: str, detail: str) -> None:
    """Sticky for the life of the process. Logs exactly ONE line."""
    if _prewarm_trip_state():
        return
    _PREWARM["state"] = "tripped:" + kind
    _PREWARM["tripped_ts"] = time.time()
    _PREWARM["tripped_detail"] = detail
    try:
        _prewarm_note_error(RuntimeError(f"prewarm governor tripped ({kind}): {detail}"))
    except Exception:
        pass


def _prewarm_guard_trip(phase: str) -> bool:
    """True when pre-warming must stop. Fails OPEN if /proc cannot be read."""
    if _prewarm_trip_state():
        return True
    _mode, _margin_kb = _prewarm_guard_cfg()
    if _mode in ("off", "false", "0", "no", "none"):
        return False
    _rss = _rss_kb()
    _avail = _mem_available_kb()
    if _rss is None or _avail is None:
        if not _PREWARM.get("guard_read_failed"):
            _PREWARM["guard_read_failed"] = True
            try:
                _prewarm_note_error(RuntimeError(
                    "prewarm governor: /proc unreadable, failing open"))
            except Exception:
                pass
        return False
    if _avail < 1_000 * 1024:
        _prewarm_trip("memavail", f"{phase}: MemAvailable {_avail} kB < 1,000 MB")
        return True
    _base = _PREWARM.get("rss_baseline_kb")
    if _base is not None and _rss > int(_base) + int(_margin_kb):
        _prewarm_trip("rss", f"{phase}: RSS {_rss} kB > baseline {_base} + "
                             f"{int(_margin_kb)} kB")
        return True
    return False


def _warm_query_enabled() -> bool:
    """Warm query kill switch. Read per cycle so it flips without a restart."""
    try:
        return bool((_recall_flags() or {}).get("prewarm_warm_query", True))
    except Exception:
        return True


#: Fixed, deterministic query used by the pre-warm retrieval pass. The vector
#: is pre-seeded into the pipeline cache, so this string is only a cache key.
_WARM_QUERY = "prewarm retrieval pipeline warmup"
_WARM_NAMESPACES = ("memory",)


class _WarmEmbed:
    """Embedder adapter for the warm query (same shape recall() builds inline).

    Never actually embeds on the happy path: the query vector is pre-seeded
    into pipeline._qvec_cache, so embed() would only run if that seeding failed.
    """

    dim = VECTOR_DIM
    backend = "ollama-local"

    def embed(self, text, kind="document"):
        cap = _embed_query_cap() if kind == "query" else None
        return _embed_floats_local(text, cap=cap)

    def pack(self, text):
        return _pack(text)


def _warm_query_vector():
    """Fixed 1024-d vector from the sit-vector sidecar — no embedding cost."""
    try:
        _ids, _mat, _meta = _sit_vectors()
        if not _ids or _mat is None or len(_mat) == 0:
            return None
        _v = _mat[0]
        return _v.tolist() if hasattr(_v, "tolist") else list(_v)
    except Exception:
        return None


def _warm_query_once(inst) -> float:
    """ONE retrieval-only pass through the same path recall() uses.

    Same namespace mask (["memory"]), same pipeline wrapper, same depth/top_k,
    fixed vector, and no rerank / trace write / formatting. Runs AFTER the
    snapshot is published to the cache, so a turn arriving now cache-hits and
    never waits for this. Round 46c measured the first retrieval after a reload
    at 808 ms vs 391 ms once warm, with a warm query costing 347 ms.
    """
    t0 = time.perf_counter()
    vec = _warm_query_vector()
    if vec is None:
        return 0.0
    mm = inst.mem
    node_ns: Dict[int, str] = {}
    _ids = (mm.all_node_ids() if hasattr(mm, "all_node_ids")
            else range(mm.header.node_count))
    for nid in _ids:
        try:
            wiki = mm.get_wiki(nid) or {}
            node_ns[nid] = wiki.get("_ns") or classify_namespace(
                mm.get_name(nid) or "",
                mm.get_text(nid) or mm.get_summary(nid) or "")
        except Exception:
            node_ns[nid] = "memory"
    emb = _WarmEmbed()
    planes = [Plane.ASSOCIATIVE, Plane.CAUSAL, Plane.TEMPORAL,
              Plane.FUNCTIONAL, Plane.PRIORITY]
    fetch_k = 20
    if _ns_prefilter_enabled():
        allowed = {nid for nid, ns in node_ns.items() if ns in _WARM_NAMESPACES}
        pipe = _NsMaskedPipeline(mm, embedder=emb, depth=2, top_k=fetch_k,
                                 max_iterations=2, planes=planes, allowed=allowed)
    else:
        pipe = RetrievalPipeline(mm, embedder=emb, depth=2, top_k=fetch_k,
                                 max_iterations=2, planes=planes)
    pipe._qvec_cache = {_WARM_QUERY: list(vec)}
    pipe.query(_WARM_QUERY, top_n=fetch_k)
    ms = (time.perf_counter() - t0) * 1000.0
    _PREWARM["warmq_ms"] = ms
    return ms


def _prewarm_once() -> float:
    """Reload the read snapshot (optionally + one warm query). Returns ms.

    Load-only by default: phase 1 measured the first-query penalty at a median
    of ~460 ms, but with the reranker varying 0.2-5.5 s per call on this host
    that figure is inside the noise, while a background query would burn 2-6 s
    of CPU on every turn. So the default warms the snapshot (the ~390 ms that
    IS attributable) and leaves the query to the real turn. Set
    recall.config "prewarm_query": true to include it.
    """
    t0 = time.perf_counter()
    if _prewarm_guard_trip("pre"):
        return 0.0
    with _LOAD_LOCK:
        inst = MPMFMemory(PATH, vector_dimensions=VECTOR_DIM, embed_model=EMBED_MODEL)
        key = _store_key(PATH)
        if key is not None:
            old = None
            with _OPEN_CACHE_LOCK:
                old = _OPEN_CACHE.get(PATH)
                _OPEN_CACHE[PATH] = (key, inst)
            del old          # governor reads RSS at this point: old snapshot gone
        _PREWARM["rss_after_publish_kb"] = _rss_kb()
    # warm_baseline = own RSS right after the SECOND completed reload
    if _PREWARM.get("rss_baseline_kb") is None and \
            int(_PREWARM.get("warm_count") or 0) == 1:
        _PREWARM["rss_baseline_kb"] = _PREWARM.get("rss_after_publish_kb")
    if _prewarm_guard_trip("post"):
        _PREWARM["warm_ms"] = (time.perf_counter() - t0) * 1000.0
        _PREWARM["warm_count"] = int(_PREWARM.get("warm_count") or 0) + 1
        return _PREWARM["warm_ms"]
    try:
        if _warm_query_enabled():
            _warm_query_once(inst)
        if bool((_recall_flags() or {}).get("prewarm_query", False)):
            recall("prewarm store", top_n=3, query_vec=_prewarm_query_vec())
    except Exception as e:
        _prewarm_note_error(e)
    ms = (time.perf_counter() - t0) * 1000.0
    _PREWARM["warm_ms"] = ms
    _PREWARM["warm_count"] = int(_PREWARM.get("warm_count") or 0) + 1
    return ms


def _prewarm_loop() -> None:
    last_key, changed_at = None, None
    while not _PREWARM["stop"]:
        try:
            time.sleep(_PREWARM_POLL_S)
            if not _prewarm_enabled():
                last_key, changed_at = None, None
                continue
            if _prewarm_trip_state():
                continue
            key = _store_key(PATH)
            if key is None:
                continue
            if key != last_key:
                last_key, changed_at = key, time.time()
                continue
            if changed_at is None or (time.time() - changed_at) < _PREWARM_STABLE_S:
                continue
            with _OPEN_CACHE_LOCK:
                ent = _OPEN_CACHE.get(PATH)
            if ent is not None and ent[0] == key:
                changed_at = None
                continue
            _prewarm_once()
            changed_at = None
        except Exception as e:      # never raise out of the thread
            _prewarm_note_error(e)
            changed_at = None


def _start_prewarm() -> None:
    if _PREWARM["thread"] is not None:
        return
    try:
        t = threading.Thread(target=_prewarm_loop, name="mpmf-store-prewarm",
                             daemon=True)
        _PREWARM["thread"] = t
        t.start()
    except Exception as e:
        _prewarm_note_error(e)


def _maybe_migrate_legacy():
    """If the store file is the old (non-plus) version, back it up and start fresh.

    Detection: if the file exists but doesn't have any 'fact_*' named nodes with
    full 5-plane wiki, it's the legacy store. We back it up rather than risk
    overwriting D's curated memory.
    """
    if not os.path.exists(PATH):
        return
    # Heuristic: if file size is in the small range (legacy v1) and there are
    # no JSON-wiki blobs in node records, treat as legacy and back up.
    # We don't auto-detect perfectly; we require explicit user opt-in via env var
    # MPMF_PLUS_MIGRATE=1 to actually rename the file.
    if os.environ.get("MPMF_PLUS_MIGRATE") == "1" and not os.path.exists(PATH_LEGACY):
        os.rename(PATH, PATH_LEGACY)
        print(f"[migrate] backed up {PATH} -> {PATH_LEGACY}")


def _raw_extraction(text: str, name: str, summary: str = "",
                    wiki: Optional[dict] = None,
                    namespace: Optional[str] = None) -> dict:
    """The degraded payload persisted BEFORE the enrichment LLM is called.

    Uses ONLY caller-supplied values plus the regex namespace classifier. This
    is what a fact keeps forever if enrichment never succeeds — a valid, plain
    FACT node with a real summary/namespace, just without the 5-plane richness.
    """
    ns = namespace or classify_namespace(name, text)
    if ns not in ("memory", "ruminate", "meta", "cognition", "transcript",
                  "cognition_staging"):
        ns = classify_namespace(name, text)
    return {
        "summary": summary or text[:120],
        "namespace": ns,
        "wiki": dict(wiki or {}),
        "planes": {"associative": [], "causal": "", "temporal": "",
                   "functional": "", "priority": 3},
        "_model": "raw-first (pre-enrichment)",
    }


def _resolve_payload(extraction: dict, text: str, name: str, summary: str = "",
                     wiki: Optional[dict] = None, importance: float = 0.6,
                     namespace: Optional[str] = None):
    """Turn an extraction dict into concrete (summary, wiki, flags, importance, ns).

    Single source of truth for BOTH the raw-first write and the post-enrichment
    update, so the two paths can never disagree about how a node is shaped.
    """
    wiki = wiki or {}
    final_summary = (extraction.get("summary") or summary or text)[:120]
    final_wiki = dict(extraction.get("wiki") or wiki or {})
    final_planes = extraction.get("planes") or {}
    final_flags = compute_plane_flags(final_planes)
    priority = final_planes.get("priority", 3)
    if not isinstance(priority, (int, float)) or not (1 <= priority <= 5):
        priority = 3
    final_importance = importance if importance != 0.6 else (0.5 + priority * 0.1)

    # Namespace: explicit arg > regex classifier. (extraction["namespace"] is
    # deliberately ignored — the model's guessed namespace must not override.)
    final_ns = namespace or classify_namespace(name, text)
    if final_ns not in ("memory", "ruminate", "meta", "cognition", "transcript",
                        "cognition_staging"):
        final_ns = classify_namespace(name, text)

    # Bookkeeping the retrieval/decay layer expects on every node's wiki.
    final_wiki["_qid"] = "live"
    final_wiki["_ns"] = final_ns  # namespace for recall filtering
    final_wiki["_base_weight"] = str(0.5 + priority * 0.1)
    final_wiki["planes"] = json.dumps(final_planes)
    final_wiki["_extraction_model"] = extraction.get("_model", EXTRACT_MODEL)

    # Relational fields from extraction (relates_to kept as a native list).
    for rel_field in ("relates_to", "relation_semantics", "preventive_tool_or_pattern"):
        rel_val = extraction.get(rel_field)
        if rel_val is not None:
            if rel_field == "relates_to" and isinstance(rel_val, str):
                try:
                    final_wiki[rel_field] = json.loads(rel_val)
                except Exception:
                    final_wiki[rel_field] = []
            else:
                final_wiki[rel_field] = rel_val
    return final_summary, final_wiki, final_flags, final_importance, final_ns


def _apply_extraction_to_node(m: MPMFMemory, name: str, text: str, extraction: dict,
                              importance: float, auto_link: bool,
                              incremental_link: bool = True) -> bool:
    """Merge an extraction payload onto an already-persisted raw node.

    Used after enrichment succeeds (store Phase 2) and by re_enrich_pending().
    `incremental_link=True` (default) derives edges for THIS node only; pass
    False only for bulk rebuilds that intentionally re-derive the whole table.
    """
    node_id = m.mem.get_node_id_by_name(name) if hasattr(m.mem, "get_node_id_by_name") \
        else m.mem.get_id_by_name(name)
    if node_id is None:
        return False
    # Round 7 item 3: PRESERVE the namespace already set on the node. Phase 1
    # honoured the caller's explicit namespace=; re-deriving it here (no
    # namespace arg) silently flipped cognition -> memory, which is why
    # store(..., namespace="cognition") produced _ns="memory".
    try:
        _cur_ns = (m.mem.get_wiki(node_id) or {}).get("_ns")
    except Exception:
        _cur_ns = None
    fs, fw, flags, imp, _ns = _resolve_payload(
        extraction, text, name, importance=importance, namespace=_cur_ns)
    fw["_enrichment_status"] = "enriched"
    fw["_enriched_at"] = datetime.now(timezone.utc).isoformat()
    fw.pop("_needs_reenrichment", None)
    fw.pop("_enrichment_error", None)

    try:
        m.mem.set_node_summary(node_id, fs)
    except Exception:
        pass
    # update_node_wiki coerces every value to str, so pre-serialise structured ones.
    try:
        m.mem.update_node_wiki(node_id, {
            k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in fw.items()
        })
    except Exception as e:
        print(f"[enrich] wiki update failed for {name!r}: {e}", file=sys.stderr)
    node = m.mem.get_node(node_id)
    if node is not None:
        node.plane_flags = flags
    try:
        m.mem.set_node_importance(node_id, imp)
    except Exception:
        pass
    vec = _pack(fs or text)
    if vec is not None:
        try:
            m.mem.set_node_vector(node_id, vec)
        except Exception:
            pass
    if auto_link:
        # Full _rebuild_edges() is O(E * candidates) because _edge_exists scans
        # the whole edge table per candidate — unusable on a large store. For a
        # single new fact, derive edges for THAT node only (O(E + N)).
        if incremental_link:
            _rebuild_edges_for_node(m, name)
        else:
            _rebuild_edges(m)
    m.save()
    return True


def re_enrich_pending(limit: int = 50, auto_link: bool = False) -> dict:
    """Re-run enrichment for every node flagged `_enrichment_status == "pending"`.

    Raw-first writes leave a durable plain-FACT node behind when the extraction
    LLM fails; this upgrades those in place once the LLM is reachable again.
    Safe to run repeatedly. Returns counts as a dict.
    """
    # Round 48d: this used to hold ONE private instance across up to `limit`
    # LLM calls and save it repeatedly — the worst stale-image case in the
    # codebase. Read from the shared snapshot, then re-open a transaction per
    # node so each write is applied to a current image.
    snap = _open().mem
    pending = []
    for nid in snap.all_node_ids():
        if str((snap.get_wiki(nid) or {}).get("_enrichment_status", "")) == "pending":
            pending.append(nid)
    result = {"pending": len(pending), "enriched": 0, "still_pending": 0}
    for nid in pending[:limit]:
        name = snap.get_name(nid)
        text = snap.get_text(nid)
        try:
            similar = _query_similar_nodes(text, top_k=5)
            extraction = call_extractor(text, existing_context=similar,
                                        dump_dir=EXTRACT_DUMP_DIR or None)
        except Exception as e:
            result["still_pending"] += 1
            try:
                with _STORE_LOCK, _write_txn_open() as m:
                    cur = m.mem.get_node_id_by_name(name)
                    if cur is not None:
                        m.mem.update_node_wiki(
                            cur, {"_enrichment_error": f"{type(e).__name__}: {e}"[:300]})
            except Exception:
                pass
            continue
        try:
            with _STORE_LOCK, _write_txn_open() as m:
                if _apply_extraction_to_node(m, name, text, extraction, 0.6, auto_link,
                                             incremental_link=True):
                    result["enriched"] += 1
                else:
                    result["still_pending"] += 1
        except Exception:
            result["still_pending"] += 1
    _invalidate_open_cache()
    return result


def store(text: str, name: str, node_type=NodeType.FACT,
          planes=Plane.ASSOCIATIVE.value,
          summary: str = "", wiki: Optional[dict] = None,
          importance: float = 0.6,
          auto_extract: bool = True,
          auto_link: bool = True,
          namespace: Optional[str] = None) -> str:
    """Store a fact with MPMF+ behavior (raw-first, then 5-plane enrich + auto-edges).

    RAW-FIRST (2026-09-27): the node is persisted with a degraded payload BEFORE
    the enrichment LLM is ever called, then upgraded in place. A fact can
    therefore never be lost to an enrichment failure — at worst it stays a plain
    FACT flagged `_enrichment_status="pending"` until `re_enrich_pending()`
    upgrades it. Previously the extraction call ran first, and a failure raised
    straight out of store(), dropping the fact entirely with no record.

    If auto_extract=True (default), the LLM produces the structured 5-plane
    payload; the result is cached in $MPMF_HOME/cache/extractions.json keyed by
    a hash of `text`, so re-enrichment / edge-rebuilds don't re-call the LLM.

    If auto_link=True (default), edges are rebuilt for ALL facts after enrichment.
    If namespace is None it is inferred from the extraction, else the regex
    classifier. Returns the fact name.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, dict] = {}
    if EXTRACTIONS_CACHE.exists():
        try:
            with open(EXTRACTIONS_CACHE) as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    # Keyed by text-hash so a changed text (same name) gets a fresh extraction.
    text_key = _text_hash(text)

    # ---- Phase 0: build the degraded payload from caller args only ----
    raw_extraction = _raw_extraction(text, name, summary, wiki, namespace)
    fs, fw, flags, imp, ns = _resolve_payload(
        raw_extraction, text, name, summary=summary, wiki=wiki,
        importance=importance, namespace=namespace,
    )
    fw["_enrichment_status"] = "pending"  # upgraded in Phase 2 if the LLM succeeds

    # ---- Phase 1: dedup + persist the RAW node immediately ----
    # RACE-FIX: the dedup-check-then-create is a classic TOCTOU window. Two
    # threads can both pass the check and both create. _STORE_LOCK serializes
    # the whole dedup+create+save block across ALL writers (distiller, sync_turn,
    # rumination, manual store calls).
    with _STORE_LOCK, _write_txn_open() as m:
        dedup_action, kept_nid = check_dedup(
            text=text, namespace=ns, mem=m.mem, embed_fn=_embed_floats_local,
        )
        if dedup_action in ("exact_match", "near_dup"):
            try:
                kept_name = m.mem.get_name(kept_nid) if kept_nid is not None else name
                existing_wiki = m.mem.get_wiki(kept_nid) or {}
                existing_imp = m.mem.get_importance(kept_nid) if hasattr(m.mem, "get_importance") else imp
                new_imp = max(float(existing_imp or 0.0), imp)
                # NOTE: the setter is update_node_wiki, NOT set_node_wiki. The
                # previous `hasattr(m.mem, "set_node_wiki")` guard was always
                # False, so these merge markers were silently never written.
                if hasattr(m.mem, "update_node_wiki"):
                    m.mem.update_node_wiki(kept_nid, {
                        "_last_dedup_merge": datetime.now(timezone.utc).isoformat(),
                        "_last_dedup_action": dedup_action,
                    })
                if hasattr(m.mem, "set_node_importance"):
                    m.mem.set_node_importance(kept_nid, new_imp)
                m.save()
                _invalidate_open_cache()
            except Exception as e:
                print(f"[dedup] post-merge update failed for {kept_nid}: {e}", file=sys.stderr)
            return kept_name or name

        m.remember(text, name=name, node_type=node_type, planes=flags,
                   summary=fs, wiki=fw, importance=imp)

        # Embed + write the vector directly (local embedder via Ollama).
        vec = _pack(fs or text)
        # A1.2 write-boundary dim assertion: refuse to write the wrong shape.
        expected_bytes = VECTOR_DIM * 4
        if vec is not None and len(vec) != expected_bytes:
            raise RuntimeError(
                f"DimensionMismatchError at write boundary: vec has {len(vec)} bytes, "
                f"store header requires {expected_bytes} (dim={VECTOR_DIM}). "
                f"embedder={EMBED_MODEL!r}. Per spec A1.2, this is a loud failure, "
                f"not a silent pad."
            )
        node_id = m.mem.get_node_id_by_name(name) if hasattr(m.mem, "get_node_id_by_name") \
            else m.mem.get_id_by_name(name)
        if node_id is not None and vec is not None:
            try:
                m.mem.set_node_vector(node_id, vec)
            except Exception:
                pass  # older API; remember()'s vector will be used
        m.save()  # <-- the raw node is now DURABLE. Enrichment can no longer lose it.
        _invalidate_open_cache()

    if not auto_extract:
        # No enrichment requested: the raw payload IS the final payload.
        if auto_link:
            with _STORE_LOCK, _write_txn_open() as m2:
                # Single new node -> derive its own edges (O(E+N)). The full
                # _rebuild_edges() never finishes on the multi-hundred-k edge
                # table, so a single-node path must never call it.
                _rebuild_edges_for_node(m2, name)
                m2.save()
                _invalidate_open_cache()
        return name

    # ---- Phase 2: enrich (LLM; may fail) — the node is already safe ----
    extraction = cache.get(text_key)
    if extraction is None:
        try:
            similar_nodes = _query_similar_nodes(text, top_k=5)
            extraction = call_extractor(text, existing_context=similar_nodes,
                                        dump_dir=EXTRACT_DUMP_DIR or None)
            cache[text_key] = extraction
            # _rebuild_edges() maps the cache by NODE NAME (it resolves edges to
            # node ids by name), so a text-hash key is invisible to it. Keep a
            # name-keyed copy too, otherwise store() derives 0 edges for a fact.
            cache[name] = extraction
            tmp = str(EXTRACTIONS_CACHE) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(cache, f, indent=2)
            os.replace(tmp, EXTRACTIONS_CACHE)
        except Exception as e:
            # Enrichment failed. The raw node stays; flag it for re-enrichment.
            try:
                with _STORE_LOCK, _write_txn_open() as m3:
                    nid = m3.mem.get_node_id_by_name(name) if hasattr(m3.mem, "get_node_id_by_name") \
                        else m3.mem.get_id_by_name(name)
                    if nid is not None:
                        m3.mem.update_node_wiki(nid, {
                            "_enrichment_status": "pending",
                            "_needs_reenrichment": "true",
                            "_enrichment_error": f"{type(e).__name__}: {e}"[:300],
                        })
                        m3.save()
                        _invalidate_open_cache()
            except Exception as e2:
                print(f"[store] could not flag {name!r} pending: {e2}", file=sys.stderr)
            print(f"[store] enrichment failed for {name!r}; raw node KEPT, flagged pending: {e}",
                  file=sys.stderr)
            return name

    # Enrichment succeeded (or came from cache): upgrade the node in place.
    with _STORE_LOCK, _write_txn_open() as m4:
        _apply_extraction_to_node(m4, name, text, extraction, importance, auto_link,
                                  incremental_link=True)
    _invalidate_open_cache()
    return name


def link(a: str, b: str, plane=Plane.ASSOCIATIVE, relation: str = "RELATED_TO") -> None:
    """Manual link. Bypasses the 5-rule derivation (you know better here)."""
    with _write_txn_open() as m:
        m.link(a, b, plane, relation)
    _invalidate_open_cache()


def _rebuild_edges(m: MPMFMemory) -> int:
    """Rebuild ALL edges from cached extractions. Returns count created.

    O(N^2) over extractions. For a 200-fact live store this is ~80ms.
    Edges are built WITHIN namespace only (memory<->memory, ruminate<->ruminate).
    Cross-namespace edges are skipped to keep graphs clean.
    """
    if not EXTRACTIONS_CACHE.exists():
        return 0
    try:
        with open(EXTRACTIONS_CACHE) as f:
            cache = json.load(f)
    except Exception:
        return 0

    # Build facts_by_id + namespace lookup from the current store
    facts_by_id: Dict[str, str] = {}
    name_to_ns: Dict[str, str] = {}
    node_ids = (m.mem.all_node_ids() if hasattr(m.mem, "all_node_ids")
                else range(m.mem.header.node_count))
    for nid in node_ids:
        try:
            name = m.mem.get_name(nid)
            if not name:
                continue
            text = m.mem.get_text(nid) or m.mem.get_summary(nid) or ""
            facts_by_id[name] = text
            wiki = m.mem.get_wiki(nid) or {}
            ns = wiki.get("_ns") or classify_namespace(name, text)
            name_to_ns[name] = ns
        except Exception:
            continue

    # Map: name -> extractions entry
    name_to_ext = {n: e for n, e in cache.items()}

    # The edge_derivation rules key on fact IDs (e.g. fact_31). Our live store
    # uses semantic names (user_D_profile, fact_31, etc.). Build edges keyed by
    # name; the loop below translates to node IDs.
    edges, counts = _derive_edges(name_to_ext, facts_by_id)
    plane_map = {
        "associative": Plane.ASSOCIATIVE,
        "causal": Plane.CAUSAL,
        "temporal": Plane.TEMPORAL,
    }
    created = 0
    skipped_xns = 0
    for e in edges:
        src_name = e["src"]
        dst_name = e["dst"]
        # Skip cross-namespace edges
        if name_to_ns.get(src_name) != name_to_ns.get(dst_name):
            skipped_xns += 1
            continue
        src_id = m.mem.get_node_id_by_name(src_name) if hasattr(m.mem, "get_node_id_by_name") else m.mem.get_id_by_name(src_name)
        dst_id = m.mem.get_node_id_by_name(dst_name) if hasattr(m.mem, "get_node_id_by_name") else m.mem.get_id_by_name(dst_name)
        if src_id is None or dst_id is None:
            continue
        # A3.1: enforce uniqueness at edge-creation time. The mpmf package
        # does not yet enforce this itself (A3.2 — "implement ONCE in the
        # shared mpmf package" is a follow-up; meanwhile we enforce here).
        # An edge is identified by (src_id, tgt_id, plane, relation_label).
        # If such an edge already exists, this is a no-op (we skip and count).
        plane_enum = plane_map.get(e["plane"], Plane.ASSOCIATIVE)
        rel_label = e["label"]
        if _edge_exists(m.mem, src_id, dst_id, plane_enum, rel_label):
            continue
        try:
            m.mem.create_edge(
                source_node_id=src_id,
                target_node_id=dst_id,
                plane=plane_enum,
                relation_label=rel_label,
                weight=e["weight"],
            )
            created += 1
        except Exception:
            continue
    if skipped_xns:
        print(f"[edges] skipped {skipped_xns} cross-namespace edges")

    # ── Relates-to edges (from wiki `relates_to` fields) ──
    # After standard tag-based edges, create edges from explicit relational
    # links declared by the LLM extraction. These are stored as
    # `relates_to: ["node_name_1", ...]` and `relation_semantics: "extend|confirm|..."`.
    rel_created = 0
    for nid in node_ids:
        try:
            name = m.mem.get_name(nid)
            if not name:
                continue
            wiki = m.mem.get_wiki(nid) or {}
            relates_to = wiki.get("relates_to")
            if not relates_to or not isinstance(relates_to, list):
                continue
            semantics = wiki.get("relation_semantics", "related")
            for target_name in relates_to:
                if not target_name or target_name == name:
                    continue
                dst_id = (m.mem.get_node_id_by_name(target_name)
                          if hasattr(m.mem, "get_node_id_by_name")
                          else m.mem.get_id_by_name(target_name))
                if dst_id is None or dst_id == nid:
                    continue
                # Use CAUSAL plane for explicit relational links
                if not _edge_exists(m.mem, nid, dst_id, Plane.CAUSAL, str(semantics)):
                    try:
                        m.mem.create_edge(
                            source_node_id=nid,
                            target_node_id=dst_id,
                            plane=Plane.CAUSAL,
                            relation_label=str(semantics),
                            weight=0.8,
                        )
                        rel_created += 1
                    except Exception:
                        continue
        except Exception:
            continue
    if rel_created:
        print(f"[edges] created {rel_created} relates_to edges")

    return created + rel_created


def _rebuild_edges_for_node(m: MPMFMemory, name: str) -> int:
    """Create edges between ONE node and the rest of the store, using the same 5
    rules as _rebuild_edges but only for pairs involving `name`.

    O(E + N): existing-edge keys are gathered once, then each other node is
    compared once. Avoids the O(E * candidates) blow-up of the full rebuild
    (which re-scans the whole edge table per candidate via _edge_exists).
    Namespace rule matches _rebuild_edges (edges only within a namespace).
    """
    nid = (m.mem.get_node_id_by_name(name) if hasattr(m.mem, "get_node_id_by_name")
           else m.mem.get_id_by_name(name))
    if nid is None:
        return 0
    node_ids = list(m.mem.all_node_ids())

    # Build every node's features once, then apply the same ubiquity filter as
    # _derive_edges so a single-node store() derives the same edges a full
    # rebuild would for that node.
    feats = {}
    for x in node_ids:
        wiki = m.mem.get_wiki(x) or {}
        pl = wiki.get("planes")
        if isinstance(pl, str):
            try:
                pl = json.loads(pl)
            except Exception:
                pl = {}
        if not isinstance(pl, dict):
            pl = {}
        text = m.mem.get_text(x) or m.mem.get_summary(x) or ""
        feats[x] = {"ns": wiki.get("_ns"),
                    "tags": _norm_tags(pl.get("associative", [])),
                    "causal": (pl.get("causal") or "").strip(),
                    "temporal": (pl.get("temporal") or "").strip(),
                    "functional": (pl.get("functional") or "").strip().lower(),
                    "proper": _proper_nouns(text), "text": text}
    if EDGE_TAG_SKIP_PCT is not None and feats:
        from collections import Counter as _C
        _td, _pd = _C(), _C()
        for _f in feats.values():
            for _t in _f["tags"]:
                _td[_t] += 1
            for _p in _f["proper"]:
                _pd[_p] += 1
        _lim = EDGE_TAG_SKIP_PCT / 100.0 * len(feats)
        for _f in feats.values():
            _f["tags"] = {t for t in _f["tags"] if _td[t] <= _lim}
            _f["proper"] = {p for p in _f["proper"] if _pd[p] <= _lim}

    fnew = feats[nid]
    labels = getattr(m.mem, "_edge_labels", {})
    existing = set()
    deg = {}
    for e in getattr(m.mem, "_edges", []):
        deg[e.source_node_id] = deg.get(e.source_node_id, 0) + 1
        deg[e.target_node_id] = deg.get(e.target_node_id, 0) + 1
        if e.source_node_id == nid or e.target_node_id == nid:
            other = e.target_node_id if e.source_node_id == nid else e.source_node_id
            existing.add((other, e.plane_id, labels.get(e.edge_id, "")))

    plane_map = {"associative": Plane.ASSOCIATIVE, "causal": Plane.CAUSAL,
                 "temporal": Plane.TEMPORAL}
    cands = []
    for x in node_ids:
        if x == nid:
            continue
        fx = feats[x]
        if fx["ns"] != fnew["ns"]:
            continue
        shared = fnew["tags"] & fx["tags"]
        if len(shared) >= 2:
            cands.append((x, "associative", "shared_tags:" + ",".join(sorted(shared))[:50], 0.5))
        sp = fnew["proper"] & fx["proper"]
        if sp:
            cands.append((x, "associative", "coref:" + ",".join(sorted(sp))[:50], 0.7))
        if fnew["temporal"] and fx["temporal"] and len(shared) >= 2:
            cands.append((x, "temporal", "co_temporal", 0.5))
        if (PREV_RE.search(fnew["text"]) and fnew["functional"]
                and fnew["functional"] == fx["functional"] and shared
                and not PREV_RE.search(fx["text"])):
            cands.append((x, "causal", "supersedes", 0.9))
        if fnew["causal"]:
            for ent in fx["proper"]:
                if ent in fnew["causal"]:
                    cands.append((x, "causal", "causes:" + ent, 0.6))
                    break
    cands.sort(key=lambda c: -c[3])  # weight desc, so the cap keeps strongest
    created = 0
    for x, plane, label, w in cands:
        if EDGE_DEGREE_CAP is not None and (deg.get(nid, 0) >= EDGE_DEGREE_CAP
                                            or deg.get(x, 0) >= EDGE_DEGREE_CAP):
            continue  # never push an already-capped node over the cap
        pe = plane_map[plane]
        if (x, pe.value, label) in existing:
            continue
        try:
            m.mem.create_edge(source_node_id=nid, target_node_id=x, plane=pe,
                              relation_label=label, weight=w)
            existing.add((x, pe.value, label))
            deg[nid] = deg.get(nid, 0) + 1
            deg[x] = deg.get(x, 0) + 1
            created += 1
        except Exception:
            pass
    return created


def _edge_exists(mem, src_id: int, dst_id: int, plane, rel_label: str) -> bool:
    """Per spec A3.1: edge identity = (src, tgt, plane, relation_type).

    After the library-side rewrite (repair_store.py, 2026-06-12), the
    _edge_labels dict is keyed by edge_id and is 100% populated, so we can
    use the label string for proper dedup.

    Identity check: (src, tgt, plane_id, relation_label) — the label
    is the string the original creator passed to link()/create_edge().
    """
    plane_id = plane.value if hasattr(plane, "value") else int(plane)
    edge_labels = getattr(mem, "_edge_labels", {})
    for e in getattr(mem, "_edges", []):
        if e.source_node_id != src_id: continue
        if e.target_node_id != dst_id: continue
        if e.plane_id != plane_id: continue
        # _edge_labels is keyed by edge_id (verified post-rewrite)
        existing_label = edge_labels.get(e.edge_id)
        if existing_label == rel_label:
            return True
    return False


def embed_query(text: str) -> List[float]:
    """Public helper (item 2, 2026-10-12): embed a query with the canonical
    local embedder so a SINGLE vector can be shared between memory recall and
    thinking recall. Uses the same head+tail truncation as the internal embed
    path, so the returned vector is byte-equivalent to what recall() computes.
    """
    try:
        vec = _embed_floats_local(text, cap=_embed_query_cap())
    except Exception as e:  # noqa: BLE001 - re-raised as a typed error
        raise EmbedderError(f"embedder failed for query: {type(e).__name__}: {e}") from e
    if not vec:
        raise EmbedderError("embedder returned an empty vector")
    return vec


# ---------------------------------------------------------------------------
# Namespace pre-filter — variant B (2026-10-15)
# ---------------------------------------------------------------------------
# Problem: recall() retrieved only `widen_k * len(namespaces)` nodes from the
# WHOLE store, then dropped non-matching namespaces post-hoc. With the ruminate
# layer ~50% of the store, valid memory nodes were crowded out of the small
# pool before ranking. Fix: restrict the candidate set at the vector/candidate
# stage so excluded namespaces can never occupy a slot.
_RECALL_CONFIG_PATH = MPMF_HOME / "recall.config.json"


def _ns_prefilter_enabled() -> bool:
    """Read the `ns_prefilter` flag from recall.config.json (default True).

    Read on EVERY recall() call so the flag can be flipped without a restart.
    Any error (missing file, bad JSON) falls back to the default (True).
    """
    try:
        with open(_RECALL_CONFIG_PATH) as f:
            return bool(json.load(f).get("ns_prefilter", True))
    except Exception:
        return True


def _memory_arm() -> str:
    """Recall-arm selector: recall.config.json `memory_arm`.

    "production" (default) — the 5-plane graph/wiki/lexical pipeline.
    "d20"                  — cosine top-20 over the allowed nodes, one
                             cross-encoder pass, reciprocal-rank fusion (k=20)
                             of cosine rank and rerank rank, then top-k.
                             Round 54 measured +13 combined hit@8 over
                             production with no loss on natural wording.

    Read on EVERY recall() call so it can be flipped without a restart. Any
    error, missing key, or unrecognised value falls back to "production".
    """
    try:
        with open(_RECALL_CONFIG_PATH) as f:
            val = str(json.load(f).get("memory_arm", "production")).strip().lower()
    except Exception:
        return "production"
    return "d20" if val == "d20" else "production"


# Round 55: cached (ids, matrix, norms) for the d20 arm, keyed by node count.
_D20_MATRIX: Dict[str, Any] = {"count": -1, "ids": None, "M": None, "norms": None}


def _d20_node_matrix(mem):
    """(ids, M, norms) for every node carrying a vector — cached by node count."""
    import numpy as _np

    count = int(getattr(mem.header, "node_count", -1))
    if _D20_MATRIX["count"] == count and _D20_MATRIX["M"] is not None:
        return _D20_MATRIX["ids"], _D20_MATRIX["M"], _D20_MATRIX["norms"]
    ids: List[int] = []
    rows: List[list] = []
    for nid in mem.all_node_ids():
        v = mem.get_vector(nid)
        if v is not None:
            ids.append(int(nid))
            rows.append(v)
    M = _np.asarray(rows, dtype=_np.float32)
    norms = _np.linalg.norm(M, axis=1)
    _D20_MATRIX.update({"count": count, "ids": ids, "M": M, "norms": norms})
    return ids, M, norms


def _recall_d20(m, emb, query: str, top_n: int, namespaces, node_ns,
                _t_open: float, _t_ns_scan: float, _bstart: float,
                _t0: float, query_vec: Optional[List[float]] = None) -> List[dict]:
    """Round 55 arm "d20": cosine top-20 -> cross-encoder -> RRF(k=20) -> top_n.

    Same output shape as recall(): list of dicts with rank / node_id / text /
    summary / name / wiki / _ns / score. Rerank keeps production's budget
    behaviour (skip if already over rerank_skip_after_ms, timeout falls back to
    the cosine order rather than dropping results).
    """
    import numpy as _np

    N_COS = 20
    K_RRF = 20

    _s = time.perf_counter()
    qv = list(query_vec) if query_vec is not None else emb.embed(query, kind="query")
    _t_embed = 0.0 if query_vec is not None else (time.perf_counter() - _s)

    _s = time.perf_counter()
    ids, M, norms = _d20_node_matrix(m.mem)
    pos = {nid: i for i, nid in enumerate(ids)}
    rows = [pos[n] for n in ids if node_ns.get(n, "memory") in namespaces]
    cos_order: List[int] = []
    if rows:
        sub = M[rows]
        q = _np.asarray(qv, dtype=_np.float32)
        qn = float(_np.linalg.norm(q)) or 1.0
        sims = (sub @ q) / (norms[rows] * qn + 1e-9)
        cos_order = [ids[rows[i]] for i in _np.argsort(-sims)]
    _t_cos = time.perf_counter() - _s

    top_cos = cos_order[:N_COS]
    candidates = []
    for i, nid in enumerate(top_cos, start=1):
        text = m.mem.get_text(nid) or m.mem.get_summary(nid) or ""
        candidates.append({
            "rank": i,
            "node_id": nid,
            "text": text,
            "summary": m.mem.get_summary(nid),
            "name": m.mem.get_name(nid),
            "wiki": m.mem.get_wiki(nid),
            "_ns": node_ns.get(nid, "memory"),
            "score": float(N_COS - i + 1) / N_COS,
        })

    if not candidates:
        _LAST_TIMINGS.clear()
        _LAST_TIMINGS.update({
            "open": _t_open, "open_cache_hit": bool(_LAST_OPEN_HIT),
            "ns_scan": _t_ns_scan, "embed": _t_embed, "retrieval": _t_cos,
            "rerank": 0.0, "total": time.perf_counter() - _t0, "hits": 0,
            "nsfilter": "pre" if _ns_prefilter_enabled() else "post",
            "rerank_status": "skipped", "rerank_timed_out": False,
            "n_pre": 0, "n_post": 0, "query_len": len(query or ""),
            "torch_threads": _torch_threads(), "warm_query_ms": _PREWARM.get("warmq_ms"),
            "rss_kb": _rss_kb(), "prewarm_state": _prewarm_state_str(), "arm": "d20",
        })
        return []

    _elapsed_ms = (time.perf_counter() - _bstart) * 1000.0
    rerank_status = "skipped"
    rerank_timed_out = False
    _t_rerank = 0.0
    ranked_ids: List[int] = []
    if _elapsed_ms < _rerank_skip_after_ms():
        remaining_s = max((_prefetch_budget_ms() - _elapsed_ms) / 1000.0, 0.05)
        _s = time.perf_counter()
        rr = _rerank_with_timeout(query, candidates, len(candidates), remaining_s)
        _t_rerank = time.perf_counter() - _s
        if rr is None:
            rerank_status = "timeout"
            rerank_timed_out = True
        else:
            rerank_status = "done"
            ranked_ids = [int(r["node_id"]) for r in rr]

    # Reciprocal-rank fusion of the cosine order and the rerank order.
    def _rrf(nid: int) -> float:
        s = 0.0
        if nid in _COS_POS:
            s += 1.0 / (K_RRF + _COS_POS[nid] + 1)
        if nid in _RR_POS:
            s += 1.0 / (K_RRF + _RR_POS[nid] + 1)
        return s

    _COS_POS = {nid: i for i, nid in enumerate(cos_order)}
    _RR_POS = {nid: i for i, nid in enumerate(ranked_ids)}
    pool = list(dict.fromkeys(top_cos + ranked_ids))
    if ranked_ids:
        fused = sorted(pool, key=lambda n: (-_rrf(n), _COS_POS.get(n, 10 ** 6)))
    else:
        fused = top_cos

    result = []
    for nid in fused[:top_n]:
        c = next((x for x in candidates if x["node_id"] == nid), None)
        if c is None:
            c = {
                "rank": len(result) + 1, "node_id": nid,
                "text": m.mem.get_text(nid) or m.mem.get_summary(nid) or "",
                "summary": m.mem.get_summary(nid), "name": m.mem.get_name(nid),
                "wiki": m.mem.get_wiki(nid), "_ns": node_ns.get(nid, "memory"),
                "score": 0.0,
            }
        else:
            c = dict(c)
        c["rank"] = len(result) + 1
        result.append(c)

    _LAST_TIMINGS.clear()
    _LAST_TIMINGS.update({
        "open": _t_open, "open_cache_hit": bool(_LAST_OPEN_HIT),
        "ns_scan": _t_ns_scan, "embed": _t_embed,
        "retrieval": _t_cos, "rerank": _t_rerank,
        "total": time.perf_counter() - _t0, "hits": len(result),
        "nsfilter": "pre" if _ns_prefilter_enabled() else "post",
        "rerank_status": rerank_status, "rerank_timed_out": rerank_timed_out,
        "n_pre": len(candidates), "n_post": len(result),
        "query_len": len(query or ""), "torch_threads": _torch_threads(),
        "warm_query_ms": _PREWARM.get("warmq_ms"), "rss_kb": _rss_kb(),
        "prewarm_state": _prewarm_state_str(), "arm": "d20",
    })
    return result


class _NsMaskedPipeline(RetrievalPipeline):
    """RetrievalPipeline masked to an allowed node-id set.

    The namespace restriction is applied at the vector/candidate stage: only
    allowed nodes are vector-scored, traversed, wiki-expanded and ranked, so
    non-allowed nodes (e.g. the ruminate layer) can never crowd out allowed
    ones. Min-max renormalisation is monotone, so the relative order of the
    allowed nodes is unchanged — this reproduces "namespace filter before
    ranking" without the wasteful retrieve-300-then-drop step.
    """

    def __init__(self, *a, allowed: Optional[Set[int]] = None, **kw):
        super().__init__(*a, **kw)
        self._allowed_ns = allowed

    def _ok(self, nid: int) -> bool:
        return self._allowed_ns is None or nid in self._allowed_ns

    def _vector_scores(self, query, context_ids=None):
        out = super()._vector_scores(query, context_ids)
        if self._allowed_ns is None or not out:
            return out
        sub = {nid: s for nid, s in out.items() if nid in self._allowed_ns}
        if not sub:
            return {}
        lo = min(sub.values())
        hi = max(sub.values())
        span = (hi - lo) or 1.0
        return {nid: (s - lo) / span for nid, s in sub.items()}

    def _traverse(self, seeds, depth):
        seeds = {s for s in seeds if self._ok(s)}
        visited, depth_map = super()._traverse(seeds, depth)
        if self._allowed_ns is None:
            return visited, depth_map
        return ({n for n in visited if self._ok(n)},
                {n: d for n, d in depth_map.items() if self._ok(n)})

    def _wiki_expand(self, working, query):
        add = super()._wiki_expand({n for n in working if self._ok(n)}, query)
        if self._allowed_ns is None:
            return add
        return {n for n in add if self._ok(n)}


def recall(query: str, top_n: int = DEFAULT_K, use_rerank: bool = True,
           widen_k: int = WIDEN_K,
           namespaces: Optional[List[str]] = ("memory",),
           query_vec: Optional[List[float]] = None,
           budget_start: Optional[float] = None) -> List[dict]:
    """MPMF+ query path: full 5-plane graph traversal over top-widen_k, then
    cross-encoder rerank, return top_n.

    Args:
      namespaces: list of namespaces to include in results.
                  - "memory"  : curated knowledge (user, world, projects)
                  - "ruminate" : operational telemetry from rumination loops
                  - "meta"     : system bookkeeping (rare)
                  Default behavior when this argument is OMITTED: ["memory"] only.
                  This means normal recall() is clean; ruminate is opt-in.
                  Pass ["memory", "ruminate"] to include the ruminate layer.
                  Pass None to include ALL namespaces (back-compat with v1).
    """
    if namespaces is None:
        namespaces = ["memory", "ruminate", "meta"]  # back-compat: all
    elif not namespaces:
        # Empty list means "no namespace" -- interpret as default memory
        namespaces = ["memory"]
    _t0 = time.perf_counter()
    # Round 8 item 1: the prefetch deadline counts from the START of the
    # plugin's prefetch() (including the shared embed), not from here. The
    # plugin passes its own perf_counter() start; fall back to _t0 when absent.
    _bstart = budget_start if budget_start is not None else _t0
    m = _open()
    _t_open = time.perf_counter() - _t0
    _open_hit = bool(_LAST_OPEN_HIT)
    # Build a custom embedder adapter that uses our local embed path
    class _LocalEmbedAdapter:
        dim = VECTOR_DIM
        backend = "ollama-local"

        def embed(self, text, kind="document"):
            # Recall-query embedding honours the configurable cap; document
            # embeds keep the fixed default (unchanged pre-flag behaviour).
            cap = _embed_query_cap() if kind == "query" else None
            return _embed_floats_local(text, cap=cap)

        def pack(self, text):
            return _pack(text)

    emb = _LocalEmbedAdapter()
    all_planes = [Plane.ASSOCIATIVE, Plane.CAUSAL, Plane.TEMPORAL,
                  Plane.FUNCTIONAL, Plane.PRIORITY]

    # Build a node-id -> namespace lookup so we can filter the pipeline's
    # candidate set. (The package's RetrievalPipeline doesn't support a
    # node filter; we filter post-traversal.)
    node_ns: Dict[int, str] = {}
    _s_ns = time.perf_counter()
    node_ids_iter = (m.mem.all_node_ids() if hasattr(m.mem, "all_node_ids")
                     else range(m.mem.header.node_count))
    for nid in node_ids_iter:
        try:
            wiki = m.mem.get_wiki(nid) or {}
            name = m.mem.get_name(nid) or ""
            text = m.mem.get_text(nid) or m.mem.get_summary(nid) or ""
            ns = wiki.get("_ns") or classify_namespace(name, text)
            node_ns[nid] = ns
        except Exception:
            node_ns[nid] = "memory"
    _t_ns_scan = time.perf_counter() - _s_ns

    # Round 55: alternate recall arm. Selected per call from recall.config.json
    # `memory_arm`; "production" (default) falls through to the pipeline below.
    if _memory_arm() == "d20":
        return _recall_d20(m, emb, query, top_n, namespaces, node_ns,
                           _t_open, _t_ns_scan, _bstart, _t0, query_vec=query_vec)

    widen = max(widen_k, top_n)
    # Namespace handling (2026-10-15): default is a PRE-filter — the candidate
    # set is masked to the requested namespaces BEFORE ranking, so excluded
    # namespaces (the ruminate layer is ~50% of the store) cannot crowd valid
    # nodes out of the pool. Flag is read per call so it flips without restart.
    ns_mode = "pre" if _ns_prefilter_enabled() else "post"
    if ns_mode == "pre":
        _allowed = {nid for nid, ns in node_ns.items() if ns in namespaces}
        fetch_k = widen  # the mask does the work; no over-fetch needed
        pipeline = _NsMaskedPipeline(
            m.mem, embedder=emb, depth=2, top_k=fetch_k,
            max_iterations=2, planes=all_planes, allowed=_allowed,
        )
    else:
        # Legacy path: request a wider pool to compensate for post-filtering.
        fetch_k = widen_k * max(len(namespaces), 1)
        pipeline = RetrievalPipeline(
            m.mem, embedder=emb, depth=2, top_k=fetch_k,
            max_iterations=2, planes=all_planes,
        )
    # Item 2 (2026-10-12): when the caller supplies a pre-computed query vector
    # (the prefetch plugin embeds once and shares it with thinking recall),
    # pre-seed the pipeline's per-query embed cache so its embed step is a
    # cache hit — no second Ollama round-trip. Keyed by the exact query string
    # recall() received, which is what pipeline.query() will look up.
    if query_vec is not None:
        try:
            pipeline._qvec_cache = {query: list(query_vec)}
        except Exception:
            pass
    _s = time.perf_counter()
    res = pipeline.query(query, top_n=fetch_k)
    _t_retrieval = time.perf_counter() - _s

    candidates = []
    for i, nid in enumerate(res.top_nodes, start=1):
        if node_ns.get(nid, "memory") not in namespaces:
            continue
        text = m.mem.get_text(nid) or m.mem.get_summary(nid) or ""
        candidates.append({
            "rank": len(candidates) + 1,
            "node_id": nid,
            "text": text,
            "summary": m.mem.get_summary(nid),
            "name": m.mem.get_name(nid),
            "wiki": m.mem.get_wiki(nid),
            "_ns": node_ns.get(nid, "memory"),
            "score": float(widen - i + 1) / widen,
        })
        if len(candidates) >= widen:
            break

    if not candidates:
        _trace_recall(query, namespaces, [], [])
        _LAST_TIMINGS.clear()
        _LAST_TIMINGS.update({
            "open": _t_open,
            "open_cache_hit": _open_hit,
            "ns_scan": _t_ns_scan,
            "embed": float(getattr(res, "timings", {}).get("embed", 0.0)),
            "retrieval": _t_retrieval,
            "rerank": 0.0,
            "total": time.perf_counter() - _t0,
            "hits": 0,
            "nsfilter": ns_mode,
            "rerank_status": "skipped",
            "rerank_timed_out": False,
            "n_pre": 0,
            "n_post": 0,
            "query_len": len(query or ""),
            "torch_threads": _torch_threads(),
            "warm_query_ms": _PREWARM.get("warmq_ms"),
            "rss_kb": _rss_kb(),
            "prewarm_state": _prewarm_state_str(),
        })
        return []

    # ---- rerank with a budget (round 7, item 2c) --------------------------
    # Degrade, don't drop: if retrieval already blew rerank_skip_after_ms, keep
    # the pre-rerank order; otherwise rerank under a (budget - elapsed) timeout
    # and fall back to the pre-rerank order on timeout.
    _elapsed_ms = (time.perf_counter() - _bstart) * 1000.0
    budget_ms = _prefetch_budget_ms()
    skip_after_ms = _rerank_skip_after_ms()
    rerank_status = "skipped"
    if use_rerank and _elapsed_ms < skip_after_ms:
        remaining_s = max((budget_ms - _elapsed_ms) / 1000.0, 0.05)
        _s = time.perf_counter()
        _rr = _rerank_with_timeout(query, candidates, top_n, remaining_s)
        _t_rerank = time.perf_counter() - _s
        if _rr is None:
            result = candidates[:top_n]
            rerank_status = "timeout"
        else:
            result = _rr
            rerank_status = "done"
    else:
        result = candidates[:top_n]
        _t_rerank = 0.0
    _trace_recall(query, namespaces, candidates, result)
    _LAST_TIMINGS.clear()
    _LAST_TIMINGS.update({
        "open": _t_open,
        "open_cache_hit": _open_hit,
        "ns_scan": _t_ns_scan,
        "embed": float(getattr(res, "timings", {}).get("embed", 0.0)),
        "retrieval": _t_retrieval,
        "rerank": _t_rerank,
        "total": time.perf_counter() - _t0,
        "hits": len(result),
        "nsfilter": ns_mode,
        "rerank_status": rerank_status,
        "rerank_timed_out": (rerank_status == "timeout"),
        "n_pre": len(candidates),
        "n_post": len(result),
        "query_len": len(query or ""),
        "torch_threads": _torch_threads(),
        "warm_ms": _PREWARM.get("warm_ms"),
        "warm_count": _PREWARM.get("warm_count"),
        "warm_query_ms": _PREWARM.get("warmq_ms"),
        "rss_kb": _rss_kb(),
        "prewarm_state": _prewarm_state_str(),
    })
    return result


#: Node id we are specifically watching in the recall trace (D's home_channel).
TRACE_WATCH_NODE = 4227


def _trace_recall(query, namespaces, pre, post) -> None:
    """Append one JSON line describing a recall() call to
    health/recall_trace.log: whether it fired, the query, the pre-rerank
    candidates + ranks, which survived the cross-encoder, and whether the
    watched node reached the result. Never allowed to break recall()."""
    try:
        import time as _t
        rec = {
            "ts": _t.strftime("%Y-%m-%dT%H:%M:%S"),
            "fired": True,
            "query": (query or "")[:200],
            "namespaces": list(namespaces) if namespaces else None,
            "pre_rerank": [{"rank": c.get("rank"), "node_id": c.get("node_id"),
                            "name": c.get("name")} for c in (pre or [])],
            "post_rerank": [{"node_id": c.get("node_id"), "name": c.get("name"),
                             "reranker_score": c.get("reranker_score")} for c in (post or [])],
            "n_pre": len(pre or []), "n_post": len(post or []),
            f"watch_{TRACE_WATCH_NODE}_pre": any(c.get("node_id") == TRACE_WATCH_NODE for c in (pre or [])),
            f"watch_{TRACE_WATCH_NODE}_post": any(c.get("node_id") == TRACE_WATCH_NODE for c in (post or [])),
        }
        with open(str(MPMF_HEALTH / "recall_trace.log"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def get(name: str) -> Optional[Any]:
    return _open().get(name)


def stats() -> dict:
    st = _open().stats()
    st["store_mode"] = STORE_MODE
    return st


# ============== B3: COGNITION RE-INJECTION ==============
# Per spec Part B3: at prefetch time, query the store for prior-reasoning
# lessons (extracted by distill_thinking.py from past sessions' thinking
# traces) and format them as a <prior-reasoning> block to inject into the
# prompt. The openclaw prefetch (or any caller) imports recall_cognition()
# and prepends the returned block to its context window.
#
# Status lifecycle (verified-cognition refactor, replacing count-based):
#   hypothesis — proposed by distill_thinking or verified_cognition,
#                not yet tested; may be wrong
#   verified   — confirmed by an execution-verified test (real subprocess
#                output, not self-reported confidence)
#   refuted    — tested and shown to be false under the stated conditions
# Mode:
#   off          — never inject
#   verified     — inject only verified nodes (safe default)
#   hypothesis   — inject hypotheses too (exploration)
#   all          — inject both (used by cognition_report.py)

import os as _os  # local alias to avoid clobbering an outer `os` if the
#                    file is later refactored to use it more.

# RETIRED: Count-based promotion via VALIDATION_THRESHOLD was removed in
# the verified-cognition refactor. Promotion is now execution-verified
# (see verified_cognition.py). _inject_count is passive telemetry only.
# VALIDATION_THRESHOLD is kept as a constant for backward compat with
# code that imports it, but its value is no longer used for promotion.
VALIDATION_THRESHOLD = 0  # retired

#: How many top lessons to include in a single <prior-reasoning> block.
#: Keep this small — we're competing with the prompt for context budget.
DEFAULT_COGNITION_K = int(_os.environ.get("COGNITION_INJECT_K", "4"))

#: Default mode if env var not set. "validated" is the safe default
#: (only inject lessons that have survived re-injection >= threshold).
DEFAULT_COGNITION_MODE = _os.environ.get("COGNITION_INJECT_MODE", "verified")

#: Path to the per-turn injection log (B5).
COGNITION_INJECT_LOG = MPMF_HEALTH / "cognition_inject.log"

#: Prefixes used by the distiller and promoter to mark lesson status.
COG_PROV_PREFIX = "cog_prov_"
COG_VAL_PREFIX = "cog_val_"


def _is_cognition_node(mem, nid: int) -> bool:
    """Return True if the node is a cognition node (hypothesis, verified, or refuted).
    Primary signal is wiki _ns == "cognition". The name-prefix check is a
    fallback for legacy nodes created before the _ns convention.
    """
    wiki = mem.get_wiki(nid) or {}
    if wiki.get("_ns") == "cognition":
        return True
    name = mem.get_name(nid) or ""
    return name.startswith(COG_PROV_PREFIX) or name.startswith(COG_VAL_PREFIX)


def _lesson_status(mem, nid: int) -> str:
    """Return 'hypothesis', 'verified', 'refuted', or 'unknown' for a
    cognition node. The wiki _status field is the single source of truth.
    Legacy 'provisional' → 'hypothesis', 'validated' → 'verified'.
    Name prefix is fallback only (nodes created before the _ns convention).
    'unknown' means data integrity issue.
    """
    wiki = mem.get_wiki(nid) or {}
    wiki_status = wiki.get("_status", "")
    if wiki_status in ("verified", "refuted"):
        return wiki_status
    if wiki_status == "validated":
        return "verified"
    if wiki_status == "provisional":
        return "hypothesis"
    if wiki_status == "hypothesis":
        return "hypothesis"
    # Fallback: name prefix for legacy nodes without wiki _status
    name = mem.get_name(nid) or ""
    if name.startswith(COG_VAL_PREFIX):
        return "verified"
    if name.startswith(COG_PROV_PREFIX):
        return "hypothesis"
    return "unknown"


def _maybe_promote(mem, nid: int) -> bool:
    """If the node's injection count has reached VALIDATION_THRESHOLD and
    it's still provisional, promote it to validated by:
      1. Renaming cog_prov_<h> → cog_val_<h>
      2. Updating the wiki _status field
      3. Saving the store
    Returns True if a promotion happened.
    """
    if _lesson_status(mem, nid) != "provisional":
        return False
    wiki = mem.get_wiki(nid) or {}
    inject_count = int(wiki.get("_inject_count", 0) or 0)
    if inject_count < VALIDATION_THRESHOLD:
        return False
    # Promote.
    old_name = mem.get_name(nid) or ""
    if not old_name.startswith(COG_PROV_PREFIX):
        # Name says provisional but prefix doesn't match — leave it.
        return False
    new_name = COG_VAL_PREFIX + old_name[len(COG_PROV_PREFIX):]
    # The low-level API doesn't expose a name-rename; the wiki update path
    # is the supported way to flip the status marker.
    wiki["_status"] = "validated"
    wiki["_promoted_at"] = datetime.now(timezone.utc).isoformat()
    mem.update_node_wiki(nid, wiki)
    # Also try to rename via the high-level API; tolerate failure.
    try:
        # MPMFMemory doesn't expose a rename method either, but a future
        # revision might. We log the intent here so the operator knows
        # the wiki was updated but the name wasn't.
        audit_cognition("promote_wiki_only", name=old_name, new_name=new_name,
                         inject_count=inject_count)
    except NameError:
        pass  # audit_cognition is in distill_thinking; that's fine
    return True


def _bump_inject_count(mem, nid: int):
    """Increment the wiki _inject_count for the lesson. Used to drive
    auto-promotion to validated (see _maybe_promote)."""
    wiki = mem.get_wiki(nid) or {}
    wiki["_inject_count"] = int(wiki.get("_inject_count", 0) or 0) + 1
    wiki["_last_injected"] = datetime.now(timezone.utc).isoformat()
    mem.update_node_wiki(nid, wiki)


# ---------------------------------------------------------------------------
# recall_cognition ranking + floor (round 5, item 8) — flags default OFF/lexical
# ---------------------------------------------------------------------------
#: Cached cognition-node vectors keyed by store node count. Refreshed only when
#: the count changes, so ranking a turn costs no embedding work and no extra
#: Ollama round-trip (the plugin already computes the shared query vector).
_COG_VEC_CACHE = {"count": None, "vecs": {}}


def _recall_flags() -> Dict:
    """Read the cognition flags from recall.config.json, per call."""
    try:
        with open(_RECALL_CONFIG_PATH) as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _cognition_rank() -> str:
    """'embed' | 'lexical'. Default 'lexical' until flipped in config."""
    v = str(_recall_flags().get("cognition_rank", "lexical")).lower()
    return "embed" if v == "embed" else "lexical"


def _cognition_floor() -> float:
    try:
        return float(_recall_flags().get("cognition_floor", 0.0) or 0.0)
    except Exception:
        return 0.0


# ---- round 36: cosine retrieval scores the SITUATION clause (the "b" arm) ---
# Reference = condition b: rank each P2 lesson by cos(turn, embed(sit)), where
# sit = text.split(" | ")[0][:1200] — exactly what built p2_emb.json. Lesson
# vectors come from the OFF-PATH sidecar (built by local Ollama), so a live
# turn NEVER embeds a lesson. This replaces the round-35 "config flag" path:
# the arm now comes ONLY from the mode_ctl state file (one mechanism).
COG_RUM_PREFIX = "cog_rum_"
SIT_NPZ_PATH = MPMF_HOME / "state" / "cognition_sit_vectors.npz"
SIT_INDEX_PATH = MPMF_HOME / "state" / "cognition_sit_index.json"
_SIT_CACHE = {"key": None, "ids": [], "mat": None, "meta": {}}
_SIT_BUILD = {"running": False}
_SIT_BUILD_LOCK = None  # lazily created threading.Lock


def _sit_text(full_text: str) -> str:
    """The SITUATION clause exactly as p2_emb.json built it."""
    return (full_text or "").split(" | ")[0][:1200]


def _sit_pool_ids(mm) -> list:
    """P2 pool ids: cognition nodes ∪ cog_rum_ rumination nodes, minus refuted."""
    out = []
    try:
        ids = mm.all_node_ids() if hasattr(mm, "all_node_ids") else range(mm.header.node_count)
    except Exception:
        return out
    for nid in ids:
        try:
            name = mm.get_name(nid) or ""
            if not (_is_cognition_node(mm, nid) or name.startswith(COG_RUM_PREFIX)):
                continue
            if _lesson_status(mm, nid) == "refuted":
                continue  # known-wrong advice is never injected
            out.append(nid)
        except Exception:
            continue
    return out


def _sit_vectors():
    """Load the sit-vector sidecar -> (ids, normalised float32 matrix, meta).
    mtime-cached. Returns ([], None, {}) when absent. Never raises."""
    try:
        st = os.stat(SIT_NPZ_PATH)
        key = (st.st_mtime_ns, st.st_size)
    except Exception:
        _SIT_CACHE.update({"key": None, "ids": [], "mat": None, "meta": {}})
        return [], None, {}
    if _SIT_CACHE["key"] != key:
        try:
            import numpy as _np
            z = _np.load(SIT_NPZ_PATH, allow_pickle=False)
            ids = [int(x) for x in z["ids"]]
            mat = _np.asarray(z["vecs"], dtype=_np.float32)
            try:
                with open(SIT_INDEX_PATH) as f:
                    meta = json.load(f) or {}
            except Exception:
                meta = {}
            _SIT_CACHE.update({"key": key, "ids": ids, "mat": mat, "meta": meta})
        except Exception:
            return _SIT_CACHE["ids"], _SIT_CACHE["mat"], _SIT_CACHE["meta"]
    return _SIT_CACHE["ids"], _SIT_CACHE["mat"], _SIT_CACHE["meta"]


def build_sit_vectors(force: bool = False, max_workers: int = 3) -> dict:
    """Build/refresh the sit-vector sidecar for the P2 pool — LOCAL Ollama only,
    OFF the turn path. Re-embeds only nodes whose sit text changed. Writes the
    .npz + index atomically. Returns a stats dict; never raises."""
    stats = {"pool": 0, "embedded": 0, "reused": 0, "skipped": 0, "ok": False}
    try:
        import numpy as _np
        import hashlib as _hl
        from concurrent.futures import ThreadPoolExecutor as _TPE
        m = _open()
        ids = _sit_pool_ids(m.mem)
        stats["pool"] = len(ids)
        old_meta, old_row, old_vecs = {}, {}, None
        try:
            with open(SIT_INDEX_PATH) as f:
                old_meta = (json.load(f) or {}).get("nodes", {}) or {}
            z = _np.load(SIT_NPZ_PATH, allow_pickle=False)
            old_vecs = _np.asarray(z["vecs"], dtype=_np.float32)
            for r, x in enumerate([int(v) for v in z["ids"]]):
                old_row[x] = r
        except Exception:
            old_meta, old_row, old_vecs = {}, {}, None
        keep_ids, keep_vecs, new_items = [], [], []
        for nid in ids:
            t = _sit_text(m.mem.get_text(nid) or m.mem.get_summary(nid) or "")
            sha = _hl.sha1(t.encode("utf-8", "ignore")).hexdigest()
            if (not force and old_vecs is not None and nid in old_row
                    and (old_meta.get(str(nid)) or {}).get("sha1") == sha):
                keep_ids.append(nid); keep_vecs.append(old_vecs[old_row[nid]])
                stats["reused"] += 1
            else:
                new_items.append((nid, t, sha))
        if new_items:
            def _do(it):
                nid, t, sha = it
                try:
                    return nid, sha, _embed_floats_local(t, cap=1200)
                except Exception:
                    return nid, sha, None
            with _TPE(max_workers=max_workers) as ex:
                for nid, sha, v in ex.map(_do, new_items):
                    if v is None:
                        stats["skipped"] += 1; continue
                    keep_ids.append(nid); keep_vecs.append(v); stats["embedded"] += 1
        if keep_vecs:
            mat = _np.asarray(keep_vecs, dtype=_np.float32)
            nrm = _np.linalg.norm(mat, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
            mat = mat / nrm
        else:
            mat = _np.zeros((0, 1), dtype=_np.float32)
        SIT_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(SIT_NPZ_PATH) + ".tmp.npz"
        _np.savez(tmp, ids=_np.asarray(keep_ids, dtype=_np.int64), vecs=mat)
        os.replace(tmp, SIT_NPZ_PATH)
        nodes = {}
        for nid in keep_ids:
            t = _sit_text(m.mem.get_text(nid) or m.mem.get_summary(nid) or "")
            nodes[str(nid)] = {"sha1": _hl.sha1(t.encode("utf-8", "ignore")).hexdigest(),
                               "dim": int(mat.shape[1])}
        with open(SIT_INDEX_PATH, "w") as f:
            json.dump({"count": len(keep_ids), "dim": int(mat.shape[1]),
                       "built_at": datetime.now(timezone.utc).isoformat(),
                       "skipped": stats["skipped"], "nodes": nodes}, f, indent=1)
        _SIT_CACHE.update({"key": None, "ids": [], "mat": None, "meta": {}})
        stats["ok"] = True
    except Exception as e:
        stats["error"] = f"{type(e).__name__}: {e}"
    return stats


def ensure_sit_vectors_async():
    """Start a background build if the sidecar is absent. Never blocks a turn."""
    global _SIT_BUILD_LOCK
    try:
        import threading as _th
        if _SIT_BUILD_LOCK is None:
            _SIT_BUILD_LOCK = _th.Lock()
        with _SIT_BUILD_LOCK:
            if _SIT_BUILD["running"]:
                return False
            _SIT_BUILD["running"] = True
        def _run():
            try:
                build_sit_vectors()
            finally:
                _SIT_BUILD["running"] = False
        _th.Thread(target=_run, daemon=True, name="sit-vec-build").start()
        return True
    except Exception:
        _SIT_BUILD["running"] = False
        return False


# ---------------------------------------------------------------------------
# round 38: keep the sit sidecar in step with the live P2 pool.
# ---------------------------------------------------------------------------
#: pool-id cache keyed by store node_count (the O(N) pool scan runs only when
#: the store actually grows). sit_missing = pool ids absent from the sidecar.
_SIT_POOL_CACHE = {"count": None, "ids": set()}
#: rate limiter for the OFF-PATH sidecar build (at most once per period).
_SIT_REFRESH = {"last": 0.0}
_SIT_REFRESH_LOCK = threading.Lock()
_SIT_REFRESH_PERIOD_S = 600.0   # once per 10 minutes


def _sit_pool_ids_cached(mm=None) -> set:
    """P2 pool ids, cached by store node_count. Never raises."""
    try:
        if mm is None:
            mm = _open().mem
        count = int(getattr(mm.header, "node_count", -1))
    except Exception:
        return set(_SIT_POOL_CACHE.get("ids") or set())
    if count != _SIT_POOL_CACHE["count"]:
        try:
            _SIT_POOL_CACHE["ids"] = set(_sit_pool_ids(mm))
            _SIT_POOL_CACHE["count"] = count
        except Exception:
            pass
    return _SIT_POOL_CACHE["ids"]


def _sit_reconcile():
    """(pool_size, n_missing, n_stale): missing = pool ids absent from the
    sidecar (new lesson not yet embedded); stale = sidecar ids no longer in the
    pool (e.g. a lesson refuted/removed since the sidecar was built)."""
    ids, _mat, _meta = _sit_vectors()
    have = set(int(x) for x in (ids or []))
    pool = _sit_pool_ids_cached()
    return len(pool), len(pool - have), len(have - pool)


def _ensure_sit_vectors_rate_limited(period_s: float = _SIT_REFRESH_PERIOD_S) -> bool:
    """Start the async sidecar build at most once per period_s. Never inline."""
    try:
        now = time.time()
        with _SIT_REFRESH_LOCK:
            if now - _SIT_REFRESH["last"] < period_s:
                return False
            _SIT_REFRESH["last"] = now
        return ensure_sit_vectors_async()
    except Exception:
        return False


def _cognition_min_cos() -> float:
    try:
        return float(_recall_flags().get("cognition_min_cos", 0.65) or 0.65)
    except Exception:
        return 0.65


# ---- round 35b: self-managing arm state (no restart needed) ---------------
#: Effective arm = pinned if set, else champion, else jaccard. The plugin and
#: CLI both read this file; it is re-stat'ed every turn (cached by mtime).
STATE_PATH = MPMF_HOME / "state" / "retrieval_modes.json"
_COG_STATE_CACHE = {"key": None, "val": None}


def cognition_state() -> dict:
    """Resolve the cognition arm from the state file, mtime-cached. Never raises.
    Missing/corrupt -> jaccard with a reason string for the injection log."""
    try:
        st = os.stat(STATE_PATH)
        key = (st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        return {"effective": "jaccard", "champion": None, "pinned": None,
                "reason": "state_missing"}
    except Exception as e:
        return {"effective": "jaccard", "champion": None, "pinned": None,
                "reason": f"state_stat_error:{type(e).__name__}"}
    if _COG_STATE_CACHE["key"] != key:
        try:
            with open(STATE_PATH) as f:
                d = json.load(f)
            blk = (d or {}).get("cognition") or {}
            champ = blk.get("champion") if blk.get("champion") in ("jaccard", "cosine") else None
            pin = blk.get("pinned") if blk.get("pinned") in ("jaccard", "cosine") else None
            eff = pin or champ or "jaccard"
            reason = "pinned" if pin else ("champion" if champ else "invalid_arm->jaccard")
            val = {"effective": eff, "champion": champ, "pinned": pin, "reason": reason}
        except Exception as e:
            val = {"effective": "jaccard", "champion": None, "pinned": None,
                   "reason": f"state_corrupt:{type(e).__name__}"}
        _COG_STATE_CACHE["key"] = key
        _COG_STATE_CACHE["val"] = val
    return _COG_STATE_CACHE["val"]


# ---- round 43: lesson vet sidecar (filter for BOTH arms) -------------------
#: Verdicts live in a sidecar, never in the store. Retrieval reads it; nothing
#: here ever writes it. Mode comes from the "vet" block of the state file:
#: effective = pinned -> champion -> "off". Arms: off | filter | strict.
VET_PATH = MPMF_HOME / "state" / "lesson_vet.json"
VET_ARMS = ("off", "filter", "strict")
_VET_CACHE = {"key": None, "val": None}
_VET_MODE_CACHE = {"key": None, "val": None}
_VET_MASK_CACHE = {"key": None, "mask": None, "stats": None}


def _vet_mode() -> dict:
    """Effective vet arm from the state file's 'vet' block, mtime-cached.
    Never raises. Missing/corrupt state -> 'off'."""
    try:
        st = os.stat(STATE_PATH)
        key = (st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        return {"effective": "off", "champion": None, "pinned": None,
                "reason": "state_missing"}
    except Exception as e:  # noqa: BLE001
        return {"effective": "off", "champion": None, "pinned": None,
                "reason": f"state_stat_error:{type(e).__name__}"}
    if _VET_MODE_CACHE["key"] != key:
        try:
            with open(STATE_PATH) as f:
                d = json.load(f)
            blk = (d or {}).get("vet") or {}
            champ = blk.get("champion") if blk.get("champion") in VET_ARMS else None
            pin = blk.get("pinned") if blk.get("pinned") in VET_ARMS else None
            eff = pin or champ or "off"
            reason = "pinned" if pin else ("champion" if champ else "default")
            val = {"effective": eff, "champion": champ, "pinned": pin,
                   "reason": reason}
        except Exception as e:  # noqa: BLE001
            val = {"effective": "off", "champion": None, "pinned": None,
                   "reason": f"state_corrupt:{type(e).__name__}"}
        _VET_MODE_CACHE["key"] = key
        _VET_MODE_CACHE["val"] = val
    return _VET_MODE_CACHE["val"]


def _vet_sidecar() -> dict:
    """Load state/lesson_vet.json, mtime-cached like cognition_state().

    Returns {"state": "ok"|"missing"|"corrupt"|"implausible", "entries": {...},
    "gate": {...}, "reason": str}. 'implausible' is set lazily by _vet_mask()
    when the filter would remove more than half the candidate pool; it is
    sticky for the life of that sidecar version (fail OPEN). Never raises.
    """
    try:
        st = os.stat(VET_PATH)
        key = (st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        return {"state": "missing", "entries": {}, "gate": {},
                "reason": "sidecar_missing"}
    except Exception as e:  # noqa: BLE001
        return {"state": "corrupt", "entries": {}, "gate": {},
                "reason": f"sidecar_stat_error:{type(e).__name__}"}
    if _VET_CACHE["key"] != key:
        try:
            with open(VET_PATH) as f:
                d = json.load(f)
            ent = (d or {}).get("entries")
            if int((d or {}).get("schema_version") or 0) != 1:
                raise ValueError("schema_version")
            if not isinstance(ent, dict) or not ent:
                raise ValueError("empty_entries")
            val = {"state": "ok", "entries": ent, "gate": (d or {}).get("gate") or {},
                   "reason": "loaded", "key": key}
        except Exception as e:  # noqa: BLE001
            val = {"state": "corrupt", "entries": {}, "gate": {},
                   "reason": f"sidecar_parse:{type(e).__name__}", "key": key}
        _VET_CACHE["key"] = key
        _VET_CACHE["val"] = val
        _VET_MASK_CACHE.update({"key": None, "mask": None, "stats": None})
    return _VET_CACHE["val"]


def _vet_plausible(sc) -> bool:
    """Pool-level sanity check (round 43): would 'filter' drop >50% of the POOL?

    Cheap estimate — reject entries plus pool names with no entry at all (text
    sha1 mismatches are rare and must not trip this). Cached per sidecar version.
    A bad/absent pool returns True (do not fail open for an unrelated reason).
    """
    if "_plausible" in sc:
        return sc["_plausible"]
    ok = True
    try:
        pool = _sit_pool_ids_cached()
        n = len(pool)
        ent = sc.get("entries") or {}
        if n and ent:
            mm = _open().mem
            drop = 0
            for nid in pool:
                e = ent.get(mm.get_name(nid) or "")
                if e is None or e.get("v") == "reject":
                    drop += 1
            ok = drop <= 0.5 * n
    except Exception:  # noqa: BLE001
        ok = True
    sc["_plausible"] = ok
    return ok


def _vet_mask(ids: list, mm, mode: str):
    """(allowed: dict nid->bool | None, stats: dict) for one candidate ordering.

    Cached by (sidecar mtime/size, sha1 of the id ordering, mode) so the warm
    path is a dict compare — no text hashing per call. allowed=None means
    "do not filter" (off, bad sidecar state, or implausible).
    """
    sc = _vet_sidecar()
    base = {"vet_mode": mode, "vet_state": sc["state"],
            "vet_entries": len(sc.get("entries") or {}),
            "vet_dropped_reject": 0, "vet_dropped_unvetted": 0}
    if mode not in ("filter", "strict") or sc["state"] != "ok":
        return None, base
    # Pool-level plausibility (round 43b): if 'filter' would drop more than half
    # of the POOL, the sidecar is wrong -> fail OPEN for this call. Measured on
    # the pool, never on the per-query candidate list (which is tiny for jaccard).
    # 'strict' is an explicit narrow mode where a majority drop is expected.
    if mode == "filter" and not _vet_plausible(sc):
        stats = dict(base, vet_state="implausible")
        sc["state"] = "implausible"          # sticky for this sidecar version
        _VET_MASK_CACHE.update({"key": None, "mask": None, "stats": stats})
        return None, stats
    idx_sha = hashlib.sha1(("|".join(str(i) for i in ids)).encode()).hexdigest()
    # A text EDIT leaves the id ordering unchanged, so the store's own mtime/size
    # must be part of the key — otherwise a stale mask survives an edit.
    try:
        _sst = os.stat(PATH)
        store_key = (_sst.st_mtime_ns, _sst.st_size)
    except Exception:  # noqa: BLE001
        store_key = None
    key = (sc.get("key"), idx_sha, mode, store_key)
    if _VET_MASK_CACHE["key"] == key and _VET_MASK_CACHE["mask"] is not None:
        return _VET_MASK_CACHE["mask"], _VET_MASK_CACHE["stats"]
    ent = sc["entries"]
    allowed: Dict[int, bool] = {}
    rejected = unvetted = 0
    for nid in ids:
        e = ent.get(mm.get_name(nid) or "")
        if e is None:
            allowed[nid] = False
            unvetted += 1
            continue
        try:
            txt = mm.get_text(nid) or ""
        except Exception:  # noqa: BLE001
            txt = ""
        if hashlib.sha1(txt.encode()).hexdigest() != (e.get("sha1") or ""):
            allowed[nid] = False          # altered text = unvetted
            unvetted += 1
            continue
        if e.get("v") == "reject":
            allowed[nid] = False
            rejected += 1
            continue
        if mode == "strict" and not (e.get("v") == "keep" and e.get("audited") is True):
            allowed[nid] = False
            unvetted += 1
            continue
        allowed[nid] = True
    n = len(ids)
    stats = {"vet_mode": mode, "vet_state": "ok", "vet_entries": len(ent),
             "vet_dropped_reject": rejected, "vet_dropped_unvetted": unvetted,
             "vet_pool": n}
    _VET_MASK_CACHE.update({"key": key, "mask": allowed, "stats": stats})
    return allowed, stats


# (_cognition_pool_vectors removed in round 36 — the cosine arm now reads the
#  sit-vector sidecar via _sit_vectors(), so the store-vector cache is gone.)


def _write_cognition_inject_log(query, retrieval, top_k, top, mem):
    """One JSON line per injection (B5). Fail-soft, never raises."""
    try:
        COGNITION_INJECT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with COGNITION_INJECT_LOG.open("a") as f:
            f.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "query": (query or "")[:200],
                "retrieval": retrieval,
                "top_k": top_k,
                "n_returned": len(top),
                "names": [mem.get_name(nid) for _, nid in top],
                "statuses": [_lesson_status(mem, nid) for _, nid in top],
                "scores": [round(float(s), 4) for s, _ in top],
            }) + "\n")
    except Exception as e:
        print(f"[recall_cognition] WARN: inject log write failed: {e!r}", file=sys.stderr)


def recall_cognition_cosine(query: str, query_vec: Optional[list], top_n: int = 3,
                            min_cos: Optional[float] = None,
                            exclude_sessions: Optional[list] = None,
                            bump_counts: bool = False,
                            return_meta: bool = False,
                            log_inject: bool = True,
                            vet: bool = False):
    """Cosine retrieval of the SITUATION clause for the <prior-reasoning> block.

    Ranks each P2 lesson by cos(query_vec, embed(sit)) — the "b" arm. Vectors
    come from the OFF-PATH sidecar, so a turn costs NO Ollama round-trip: it is
    one numpy matrix-vector product. Drops anything below min_cos and returns
    top_n. Same block format and "never refuted" policy as the Jaccard path.
    exclude_sessions skips nodes whose source session is listed. "" on no hits.

    Round 43: vet=True applies the lesson-vet sidecar BEFORE top-n selection
    (reject + unvetted dropped, so the list refills from the next-best hits).
    """
    _vmode = _vet_mode()["effective"] if vet else "off"
    _vsc = _vet_sidecar() if vet else {"state": "off", "entries": {}}

    def _vfields(extra=None):
        d = {"vet_mode": _vmode, "vet_state": _vsc.get("state", "off"),
             "vet_entries": len(_vsc.get("entries") or {}),
             "vet_dropped_reject": 0, "vet_dropped_unvetted": 0}
        if extra:
            d.update(extra)
        return d

    def _empty(extra=None, status=None):
        if return_meta:
            d = {"block": "", "retrieval": "cosine", "items": [],
                 "cosine_status": status}
            d.update(_vfields())
            if extra:
                d.update(extra)
            return d
        return ""
    if not query_vec:
        return _empty(status="no_query_vec")
    if min_cos is None:
        min_cos = _cognition_min_cos()
    excl = set(s for s in (exclude_sessions or []) if s)
    ids, mat, meta = _sit_vectors()
    # round 38/39: reconcile the sidecar against the live P2 pool. Rebuild
    # (OFF-PATH, rate-limited, never inline) when the pool has ids the sidecar
    # lacks (missing) OR the sidecar has ids no longer in the pool (stale).
    try:
        pool_size, sit_missing, sit_stale = _sit_reconcile()
    except Exception:
        pool_size, sit_missing, sit_stale = (len(ids) if ids else 0), 0, 0
    if sit_missing or sit_stale:
        _ensure_sit_vectors_rate_limited()
    if mat is None or not len(ids):
        if not sit_missing:              # whole sidecar absent -> legacy behaviour
            ensure_sit_vectors_async()   # build for next turn; never blocks here
        return _empty({"pool_size": pool_size, "sit_missing": sit_missing,
                       "sit_stale": sit_stale},
                      status=("sit_missing" if sit_missing else
                              ("sit_stale" if sit_stale else "no_index")))
    try:
        import numpy as _np
        q = _np.asarray(query_vec, dtype=_np.float32)
        nq = float(_np.linalg.norm(q)) or 1.0
        q = q / nq
        scores = mat @ q                      # (N,) — one BLAS gemv
        order = _np.argsort(-scores)
    except Exception:
        order = sorted(range(len(ids)),
                       key=lambda i: _cosine(query_vec, list(mat[i])),
                       reverse=True)
        scores = [None] * len(ids)
    m = _open()
    pool_ids = _sit_pool_ids_cached()
    # round 43: boolean mask aligned with the sit index, resolved before top-n.
    vmask, vstats = (None, None)
    if vet:
        vmask, vstats = _vet_mask(ids, m.mem, _vmode)
    top = []
    k = max(0, int(top_n))
    _passed_min = 0
    _best_unused = 0.0          # round 46: best score that did NOT reach `top`
    for idx in order:
        idx = int(idx)
        s = float(scores[idx]) if scores[idx] is not None else _cosine(query_vec, list(mat[idx]))
        if s < min_cos:
            _best_unused = max(_best_unused, s)
            break
        _passed_min += 1
        nid = ids[idx]
        # round 39: skip a stale sidecar id (gone from the pool) or a lesson that
        # is now refuted — continue down the ranking so top_n is still filled.
        if nid not in pool_ids:
            _best_unused = max(_best_unused, s)
            continue
        if _lesson_status(m.mem, nid) == "refuted":
            _best_unused = max(_best_unused, s)
            continue
        # round 43: vet filter also continues down the ranking (never short).
        if vmask is not None and not vmask.get(nid, False):
            _best_unused = max(_best_unused, s)
            continue
        if excl:
            w = m.mem.get_wiki(nid) or {}
            if (w.get("_source_session") in excl) or (w.get("source_session") in excl):
                _best_unused = max(_best_unused, s)
                continue
        top.append((s, nid))
        if len(top) >= k:
            break
    if not top:
        # Round 45: tell "nothing cleared min_cos" (a correct empty answer)
        # apart from "candidates cleared it but were all filtered out".
        # Round 46: best_score = the highest score that did not make it in.
        _x = dict(vstats or {})
        _x["best_score"] = round(float(_best_unused), 4)
        return _empty(_x, status=("below_min_cos" if not _passed_min
                                  else "filtered_out"))
    lines = ["<prior-reasoning>",
             "The following lessons were distilled from prior sessions. "
             "Apply them where relevant, but do not treat them as inviolable rules."]
    for i, (s, nid) in enumerate(top, 1):
        st = _lesson_status(m.mem, nid)
        tag = f" [{st}]" if st != "verified" else ""
        text = m.mem.get_text(nid) or m.mem.get_summary(nid) or ""
        lines.append(f"{i}.{tag} {text}")
    lines.append("</prior-reasoning>")
    block = "\n".join(lines)
    if log_inject:
        _write_cognition_inject_log(query, "cosine", top_n, top, m.mem)
    if return_meta:
        d = {"block": block, "retrieval": "cosine",
             "pool_size": pool_size, "sit_missing": sit_missing,
             "sit_stale": sit_stale,
             "cosine_status": "ok",
             "max_score": round(float(top[0][0]), 4),
             "best_score": round(float(top[0][0]), 4),
             "items": [{"id": nid, "name": m.mem.get_name(nid),
                        "score": round(float(s), 4),
                        "status": _lesson_status(m.mem, nid)} for s, nid in top]}
        d.update(_vfields(vstats))
        return d
    return block


# ---- prefetch budget (round 7, item 2c) — read per call --------------------
def _prefetch_budget_ms() -> float:
    try:
        return float(_recall_flags().get("prefetch_budget_ms", 6500) or 6500)
    except Exception:
        return 6500.0


_TORCH_THREADS = None


def _torch_threads() -> int:
    """Intra-op thread count of the process running recall (0 if torch absent).

    Round 45 observability: the CPU-bound embed/rerank steps scale with this,
    so the live row records what the *gateway* process actually had, not what
    some other interpreter reported.
    """
    global _TORCH_THREADS
    if _TORCH_THREADS is None:
        try:
            import torch
            _TORCH_THREADS = int(torch.get_num_threads())
        except Exception:
            _TORCH_THREADS = 0
    return int(_TORCH_THREADS)


def _rerank_skip_after_ms() -> float:
    try:
        return float(_recall_flags().get("rerank_skip_after_ms", 4500) or 4500)
    except Exception:
        return 4500.0


def _rerank_with_timeout(query, candidates, top_n, timeout_s):
    """Run rerank() in a daemon thread; return None if it doesn't finish in
    timeout_s (the caller then falls back to the pre-rerank order)."""
    import threading
    box: Dict[str, object] = {}

    def _work():
        try:
            box["r"] = rerank(query, candidates, top_k=top_n)
        except Exception as e:  # noqa: BLE001
            box["e"] = e

    th = threading.Thread(target=_work, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        return None
    if "r" not in box:
        return None
    return box["r"]


def _cognition_vectors(mm) -> Dict[int, list]:
    """nid -> vector for cognition nodes, cached by store node count."""
    try:
        count = int(getattr(mm.header, "node_count", -1))
    except Exception:
        count = -1
    if count != _COG_VEC_CACHE["count"]:
        vecs: Dict[int, list] = {}
        ids = (mm.all_node_ids() if hasattr(mm, "all_node_ids") else range(max(count, 0)))
        for nid in ids:
            try:
                if _is_cognition_node(mm, nid):
                    v = mm.get_vector(nid)
                    if v is not None:
                        vecs[nid] = v
            except Exception:
                continue
        _COG_VEC_CACHE["count"] = count
        _COG_VEC_CACHE["vecs"] = vecs
    return _COG_VEC_CACHE["vecs"]


def _cosine(a: list, b: list) -> float:
    try:
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return sum(x * y for x, y in zip(a, b)) / (na * nb)
    except Exception:
        return 0.0


def recall_cognition(query: str, top_k: int = DEFAULT_COGNITION_K,
                     mode: str = DEFAULT_COGNITION_MODE,
                     bump_counts: bool = True,
                     query_vec: Optional[list] = None,
                     arm: Optional[str] = None,
                     return_meta: bool = False,
                     log_inject: bool = True,
                     vet: bool = False):
    """MPMF+ B3: format a <prior-reasoning> block for prompt injection.

    Args:
      query:    the current user message / context that needs prior reasoning
      top_k:    max number of lessons to include (default 4)
      mode:     "off" | "verified" | "hypothesis" | "all"
                - "off": returns "" (caller must check for empty string)
                - "verified": only execution-verified lessons (safe default)
                - "hypothesis": only untested hypotheses (exploration)
                - "all": both, with status annotated
      bump_counts: if True, increment _inject_count on each lesson
                   (drives auto-promotion). Set False in tests / reports.

    Returns:
      A formatted string like:
        <prior-reasoning>
        1. SITUATION: ... PRINCIPLE: ... RULE: ...
        2. ...
        </prior-reasoning>
      or "" if mode=="off" or no lessons qualify.

    Side effects (per spec B5):
      - Each injection is logged to COGNITION_INJECT_LOG (one JSON line).
      - If bump_counts=True, the lesson's _inject_count is incremented,
        which may trigger auto-promotion to validated.
    """
    if mode == "off":
        return ""
    if mode not in ("verified", "hypothesis", "all"):
        # Accept legacy mode names for backward compat
        if mode == "validated":
            mode = "verified"
        elif mode == "provisional":
            mode = "hypothesis"
        else:
            raise ValueError(
                f"recall_cognition: unknown mode={mode!r}; "
                f"expected 'off' | 'verified' | 'hypothesis' | 'all'"
            )

    # ---- arm resolution (round 35b) -------------------------------------
    # arm=None -> state file (pinned || champion || jaccard), mtime-cached.
    # An explicit arm always wins (the plugin passes the resolved arm, and the
    # shadow arm too). cosine requires a query_vec; without one we use jaccard.
    chosen = arm if arm in ("jaccard", "cosine") else cognition_state()["effective"]
    if chosen == "cosine" and query_vec:
        return recall_cognition_cosine(query, query_vec, top_n=top_k,
                                       min_cos=_cognition_min_cos(),
                                       bump_counts=False,
                                       return_meta=return_meta,
                                       log_inject=log_inject,
                                       vet=vet)

    # ---- find candidate cognition nodes ----
    # Readers take the shared snapshot; the (rare) bump path mutates + saves,
    # so it needs a private instance.
    m = _open()   # round 48d: reads take the SHARED snapshot; the optional
                  # count-bump below opens its own short-timeout transaction.
    candidates = []
    for nid in m.mem.all_node_ids() if hasattr(m.mem, "all_node_ids") else range(m.mem.header.node_count):
        try:
            if not _is_cognition_node(m.mem, nid):
                continue
            status = _lesson_status(m.mem, nid)
            if mode == "verified" and status not in ("verified",):
                continue
            if mode == "hypothesis" and status not in ("hypothesis",):
                continue
            # mode == "all" includes hypothesis + verified, but NEVER refuted
            # (round 5, item 8a): a refuted lesson is known-wrong advice.
            if mode == "all" and status == "refuted":
                continue
            text = m.mem.get_text(nid) or m.mem.get_summary(nid) or ""
            if not text.strip():
                continue
            candidates.append({
                "nid": nid,
                "name": m.mem.get_name(nid),
                "text": text,
                "status": status,
            })
        except Exception as e:
            # Loud: a single bad node shouldn't kill the whole injection.
            print(f"[recall_cognition] WARN: skipping nid={nid}: {e!r}",
                  file=sys.stderr)
            continue

    if not candidates:
        return ""

    # ---- rank (round 5, item 8b) ----------------------------------------
    # 'lexical' (default) = token Jaccard, as before. 'embed' = cosine against
    # the SHARED query vector the plugin already computed — no cross-encoder,
    # no extra Ollama embed (cognition-node vectors come from the store).
    rank_mode = _cognition_rank()
    scores: Dict[int, float] = {}
    if rank_mode == "embed" and query_vec:
        vecs = _cognition_vectors(m.mem)
        for c in candidates:
            v = vecs.get(c["nid"])
            scores[c["nid"]] = _cosine(query_vec, v) if v else 0.0
    else:
        rank_mode = "lexical"
        q_tokens = set((query or "").lower().split())
        for c in candidates:
            if not q_tokens:
                scores[c["nid"]] = 0.0
                continue
            c_tokens = set(c["text"].lower().split())
            scores[c["nid"]] = len(q_tokens & c_tokens) / max(len(q_tokens | c_tokens), 1)
    ranked = sorted(candidates, key=lambda c: scores.get(c["nid"], 0.0), reverse=True)
    # ---- relevance floor (item 8c): inject NOTHING if no lesson clears it ---
    floor = _cognition_floor()
    ranked = [c for c in ranked if scores.get(c["nid"], 0.0) >= floor]
    # ---- round 43: lesson-vet filter, BEFORE top-k so the list never shortens
    vet_mode = "off"
    vmask = None
    vstats = {"vet_mode": "off", "vet_state": "off", "vet_entries": 0,
              "vet_dropped_reject": 0, "vet_dropped_unvetted": 0}
    if vet:
        vet_mode = _vet_mode()["effective"]
        vmask, vstats = _vet_mask([c["nid"] for c in ranked], m.mem, vet_mode)
        if vmask is not None:
            ranked = [c for c in ranked if vmask.get(c["nid"], False)]
    top = ranked[:top_k]
    if not top:
        if return_meta:
            d = {"block": "", "retrieval": "jaccard", "items": []}
            d.update(vstats)
            return d
        return ""

    # ---- bump counts (passive telemetry only; promotion is now
    #      execution-verified via verified_cognition module) ----
    if bump_counts:
        # Round 48d: this runs on the turn path — it must never fail a turn and
        # must never wait long. Short lock timeout, bump by name on a private
        # instance, and a skipped bump is counted + logged at most once/min.
        try:
            with _write_txn_open(timeout=0.2) as mb:
                for c in top:
                    nid2 = (mb.mem.get_node_id_by_name(c["name"])
                            if hasattr(mb.mem, "get_node_id_by_name") else None)
                    if nid2 is not None:
                        _bump_inject_count(mb.mem, nid2)
        except Exception as e:
            skipped = getattr(recall_cognition, "_bump_skipped", 0) + 1
            recall_cognition._bump_skipped = skipped
            now_ts = datetime.now(timezone.utc).timestamp()
            if now_ts - getattr(recall_cognition, "_bump_last_log", 0.0) >= 60.0:
                recall_cognition._bump_last_log = now_ts
                print(f"[recall_cognition] bump skipped ({type(e).__name__}, "
                      f"total {skipped}): {e!r}", file=sys.stderr)

    # ---- format the block ----
    lines = ["<prior-reasoning>",
             "The following lessons were distilled from prior sessions. "
             "Apply them where relevant, but do not treat them as inviolable rules."]
    for i, c in enumerate(top, 1):
        final_status = _lesson_status(m.mem, c["nid"])
        tag = f" [{final_status}]" if final_status != "verified" else ""
        lines.append(f"{i}.{tag} {c['text']}")
    lines.append("</prior-reasoning>")
    block = "\n".join(lines)

    # ---- log the injection (B5) ----
    if log_inject:
        try:
            COGNITION_INJECT_LOG.parent.mkdir(parents=True, exist_ok=True)
            with COGNITION_INJECT_LOG.open("a") as f:
                f.write(json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "query": (query or "")[:200],
                    "mode": mode,
                    "retrieval": "jaccard",
                    "top_k": top_k,
                    "n_returned": len(top),
                    "names": [c["name"] for c in top],
                    "statuses": [c["status"] for c in top],
                    "scores": [round(scores.get(c["nid"], 0.0), 4) for c in top],
                    "vet_mode": vstats.get("vet_mode"),
                    "vet_state": vstats.get("vet_state"),
                    "vet_dropped_reject": vstats.get("vet_dropped_reject"),
                    "vet_dropped_unvetted": vstats.get("vet_dropped_unvetted"),
                }) + "\n")
        except Exception as e:
            print(f"[recall_cognition] WARN: inject log write failed: {e!r}",
                  file=sys.stderr)

    if return_meta:
        d = {"block": block, "retrieval": "jaccard",
             "items": [{"id": c["nid"], "name": c["name"],
                        "score": round(scores.get(c["nid"], 0.0), 4),
                        "status": c["status"]} for c in top]}
        d.update(vstats)
        return d
    return block


# ============== BULK RE-EXTRACTION (used by ingest_from_legacy.py) ==============
def extract_batch(facts: List[dict], workers: int = EXTRACT_WORKERS,
                  resume: bool = True) -> Dict[str, dict]:
    """Extract 5-plane payloads for a batch of {name, text} dicts in parallel.

    Resumes from EXTRACTIONS_CACHE. Returns the full {name: extraction} dict.
    Prints progress every 25 facts.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, dict] = {}
    if resume and EXTRACTIONS_CACHE.exists():
        try:
            with open(EXTRACTIONS_CACHE) as f:
                cache = json.load(f)
            print(f"[extract_batch] resuming with {len(cache)} cached extractions")
        except Exception:
            cache = {}

    todo = [f for f in facts if f["name"] not in cache]
    print(f"[extract_batch] extracting {len(todo)} new facts ({len(cache)} cached) "
          f"with {workers} workers using {LLM_MODEL}...")

    def _work(f):
        return f["name"], call_extractor(f["text"])

    t0 = time.time()
    done = 0
    # Periodically checkpoint for resumability
    def _save_cache():
        tmp = str(EXTRACTIONS_CACHE) + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(cache, fh, indent=2)
        os.replace(tmp, EXTRACTIONS_CACHE)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_work, f): f for f in todo}
        for fut in as_completed(futures):
            name, extraction = fut.result()
            cache[name] = extraction
            done += 1
            if done % 25 == 0 or done == len(todo):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 0.1)
                eta = (len(todo) - done) / max(rate, 0.01)
                print(f"  {done}/{len(todo)} ({elapsed:.0f}s, {rate:.1f}/s, ETA {eta:.0f}s)",
                      flush=True)
                _save_cache()
    _save_cache()
    return cache


if __name__ == "__main__":
    if len(sys.argv) > 1:
        import json as _json
        print(_json.dumps(recall(" ".join(sys.argv[1:])), indent=2))
    else:
        s = stats()
        print(f"mpmf+: {s['nodes']} nodes, {s['edges']} edges "
              f"({s['vector_dim']}d, {s['embed_backend']})")
        print(f"  extractions cache: {EXTRACTIONS_CACHE} "
              f"({'exists' if EXTRACTIONS_CACHE.exists() else 'missing'})")
