"""Path + config resolution for MPMF+. The only place that knows where things live.

Order: explicit env -> MPMF_HOME -> ~/.mpmf. Never hard-codes a host path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from .errors import ConfigError

DEFAULTS: Dict[str, Any] = {
    # f2: the user's own first name(s), from LOCAL config only (never in code). Used as
    # history stop-words and in the extraction-prompt example. Default: none.
    "user_names": [],
    # g2: date-ordered "History of X" block for temporal questions (default ON; non-temporal
    # blocks are unchanged). The nightly auto-revert writes an explicit false, never removes the key.
    "history_mode": True,
    # recall
    "ns_prefilter": True,
    "cognition_rank": "lexical",
    "cognition_floor": 0.0,
    "prefetch_budget_ms": 6500,
    "rerank_skip_after_ms": 0,
    "embed_query_cap_chars": 1200,
    "cognition_min_cos": 0.65,
    "prewarm_store": True,
    "prewarm_query": False,
    "prewarm_warm_query": True,
    "memory_arm": "d20",
    "memory_render": "full",
    "rerank_clamp_chars": 0,
    # p7: let the proposer apply a TUNE+HOLDOUT winner to a recall_config registry key (default OFF)
    "auto_apply": False,
    # p8: raw user messages as their own searchable layer, fused into prefetch (default OFF, RESEARCH)
    "raw_layer": False,
    "raw_k": 3,
    "raw_min_cos": 0.5,
    "raw_rerank": True,
    "raw_item_chars": 500,
    "raw_block_chars": 1500,
    # x1: max_tokens of the ONE retry after an extraction reply that is not valid JSON (first try
    # stays 2000). 4000: the replies that fail are cut off at the 2000-token cap (failing replies
    # in the live log die at char 6774 / 7081 = ~3.5 chars/token); the longest cut-off bench reply
    # is 9948 chars = ~2.8-3.3k tokens, so 4000 holds it. <= 0 = retry with the same budget.
    "ingest_retry_max_tokens": 4000,
    # p9: re-ask the extractor once when a fact drops a name/number/date of its evidence (default OFF)
    "keep_details": False,
    # p11: show a lesson right before a completion claim lacking a tool call / numbered items (default OFF)
    "lessons_at_action": False,
    # p12: 1-hop walk over IS_A / DERIVED_FROM / supersedes edges from the top hits (default OFF, RESEARCH)
    "typed_walk": False,
    "typed_walk_k": 2,
    "typed_walk_seeds": 4,
    "typed_walk_min_cos": 0.3,
    # p18: mpmf_recall as a second pass the model calls while reasoning (default OFF, RESEARCH)
    "second_pass": False,
    "second_pass_max_calls": 2,
    "second_pass_k": 6,
    # p17/a1: append-only write segments instead of rewriting the whole image on every save
    # (default OFF; mpmf.compact folds the segments back into a plain image).
    # PER STORE since a1: ``append_segments_stores`` lists the store files that use segments
    # (compared by resolved real path). The old global ``append_segments`` is kept here only so
    # old config files still parse; it is IGNORED by the code (it never switched anything on).
    "append_segments": False,
    "append_segments_stores": [],
    "compact_after_segments": 5000,
    # injection
    "inject_thinking": True,
    "thinking_top_k": 3,
    "thinking_max_chars": 800,
    "inject_cognition": True,
    "cognition_top_k": 3,
    "cognition_max_chars": 700,
    # r1: injection rework. EVERY default below reproduces today's bytes exactly.
    # Per-section char budgets. null = the LEGACY path for that section (see FLAGS.md):
    #   facts_max_chars    null -> memory_block_chars, else 3000 (full) / max_chars (summary)
    #   verbatim_max_chars null -> raw_block_chars
    #   lessons_max_chars  null -> cognition_max_chars
    # thinking_max_chars (above) is the thinking budget; history_max_chars was a hardcoded 1500.
    "facts_max_chars": None,
    "history_max_chars": 1500,
    "verbatim_max_chars": None,
    "lessons_max_chars": None,
    # true: when the history block fires, the facts block is rendered exactly as with history off
    # and the dated list is added as its own section (false = history REPLACES the facts block).
    # r99: default ON. Replace-mode lost 10.1 pp on DMR (130 q, off 103 vs shipped 87, McNemar
    # p=0.0009) and 20 pp on BEAM ordering+temporal (80 q, 44 vs 28, p=0.0009) because the history
    # block dropped the facts that held the answer; additive is byte-identical to history-off on
    # every non-firing question and kept the LoCoMo temporal gain (+46/-9 vs off, p<1e-5).
    "history_additive": True,
    # "legacy" (today's blocks) | "tagged" (<memory_facts>, <history entity>, <verbatim>,
    # <prior_reasoning>, <lessons>, each fact with its stored metadata)
    "render_format": "legacy",
    # storage
    "top_k": 8,
    "max_chars": 3500,
    "auto_write": True,
    "turn_namespace": "transcript",
}


def dump_config(data: Dict[str, Any], path=None) -> Path:
    """Write a recall config. The ONLY writer, next to the only reader (r103).

    Defaults to the resolved config path, so auto-revert / bench fixtures / tests all
    write the same file the readers resolve at call time.
    """
    p = Path(path) if path else config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(data, fh, indent=1)
    return p


def home() -> Path:
    """MPMF_HOME, defaulting to ~/.mpmf (never a literal absolute home)."""
    return Path(os.environ.get("MPMF_HOME") or (Path.home() / ".mpmf")).expanduser()


def config_path() -> Path:
    return Path(os.environ.get("MPMF_RECALL_CONFIG") or (home() / "recall.config.json"))


def store_path() -> Path:
    return Path(os.environ.get("MPMF_STORE_PATH") or (home() / "athena.mpmf"))


def thinking_store() -> Path:
    return Path(os.environ.get("MPMF_THINKING_STORE") or (home() / "athena-thinking.mpmf"))


def gate_eval_dir() -> Path:
    """Round-40 gate material + caches: $MPMF_GATE_EVAL, else $MPMF_HOME/gate_eval (r98)."""
    return Path(os.environ.get("MPMF_GATE_EVAL") or (home() / "gate_eval"))


def health_dir() -> Path:
    return Path(os.environ.get("MPMF_HEALTH_DIR") or (home() / "health"))


def log_path(name: str) -> Path:
    return health_dir() / name


def load_config(path=None, strict: bool = False) -> Dict[str, Any]:
    """THE config reader. The path is resolved on EVERY call, never at import.

    Defaults <- file <- env overrides (MPMF_<KEY>). ``MPMF_HOME`` and
    ``MPMF_RECALL_CONFIG`` are read at call time, so a test (or a cron) that
    changes them after this module was imported still reads the new file.

    Missing file  -> the package defaults (portable install; never an error).
    strict=True   -> an unreadable / invalid / non-object file raises ConfigError
                     instead of silently returning the defaults.
    """
    cfg = dict(DEFAULTS)
    p = Path(path) if path else config_path()
    try:
        with open(p) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        data = None
    except Exception as e:
        if strict:
            raise ConfigError(f"recall config unreadable: {p}: {e}") from e
        data = None
    if data is not None:
        if not isinstance(data, dict):
            if strict:
                raise ConfigError(f"recall config is not an object: {p}")
        else:
            cfg.update(data)
    for key in list(cfg):
        env = os.environ.get("MPMF_" + key.upper())
        if env is not None:
            try:
                cfg[key] = json.loads(env)
            except Exception:
                cfg[key] = env
    if "history_mode" in cfg:       # g2: one boolean rule for file and env ("0"/"off"/"false" -> off)
        cfg["history_mode"] = _as_bool(cfg["history_mode"])
    return cfg


_TRUE = ("1", "true", "yes", "on")


def _as_bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in _TRUE
    return bool(v)


def history_mode(path=None) -> bool:
    """g2: THE resolver for history_mode. env MPMF_HISTORY_MODE > config file > DEFAULTS (True).

    Every reader (provider._history_mode, history.enabled) goes through this, so a revert that
    writes ``"history_mode": false`` to the config file is seen by all of them.
    """
    return bool(load_config(path)["history_mode"])


def user_names(cfg: Any = None) -> list:
    """f2: first name(s) from the ``user_names`` config key (list of str; default [])."""
    if cfg is None:
        cfg = load_config()
    v = cfg.get("user_names") if isinstance(cfg, dict) else None
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, (list, tuple)):
        return []
    return [s.strip() for s in v if isinstance(s, str) and s.strip()]


def personalize_extraction_prompt(prompt: str, cfg: Any = None) -> str:
    """f2: the planes-prompt example names the first configured user, else 'the user'."""
    names = user_names(cfg)
    if not names:
        return prompt
    n = names[0]
    return (prompt.replace("The user is direct,", n + " is direct,")
                  .replace('"name": "user"', '"name": "%s"' % n))


def ensure_dirs() -> None:
    for d in (home(), health_dir()):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
