# MPMF+

## TL;DR

### 1) The problem

Agents forget between sessions. Pasting whole transcripts back into the prompt is expensive and
noisy, and plain search cannot answer "when" or "in what order" — it finds the words, not the
timeline. MPMF does something narrower: after each turn it extracts the durable facts with the date
they were said, keeps them in one local file with a search vector each, and puts the closest ones
back in the prompt as a small labelled block.

### 2) Results

Public question sets, one reader model for every arm, paired scoring.

| set | MPMF (best setting) | plain search (BM25), same questions | n |
|---|---|---|---|
| LoCoMo | **60.1%** (context matched) | 49.5% | 1,540 |
| DMR | **77.7%** (facts, no timeline) | 52.3% | 130 |
| LongMemEval-S | 63.3% (context matched) | 64.2% | 120 — tie |
| BEAM-100K | **0.389** mean rubric (41.5% ≥ 0.5) | 0.308 (32.0%) | 400 |

Bold = a significant win: LoCoMo p = 0.0001, DMR p ≤ 0.004, BEAM p = 0.0004. "Context matched" means
the retrieved block was sized to the baseline's character count — achieved on LoCoMo only.
**LongMemEval-S is a tie**: nothing there is significant (all p > 0.4) and plain search is 0.9 points
ahead, so it is reported as a null result, not a win.
Ordering questions improved from 5/64 to 32/64 on a held-out set.

**How these were measured**

- **Real, strict conditions.** Our own runs on public sets, small inexpensive model (DeepSeek v4.1
  Flash) answering from at most ~2,500 characters of memory per question — 763 to 2,539 by set,
  against 1,200 to 2,984 for the baseline. A few hundred tokens, not a transcript, and less context
  than the baseline on three of the four sets.
- **Fair comparison, limits stated.** Within a set every arm had the same questions, answer prompt
  and judge, scored in pairs with exact significance tests. Caveats: the judge model differs
  *between* sets (four models, judge noise not measured), the character budget was only equalized on
  LoCoMo (1.01×; elsewhere MPMF saw 0.72–0.86× and still won where marked), and LoCoMo's 1,540
  questions come from just 10 conversations, so its p-values are optimistic.
- **No tuning on the test.** Settings were fixed before each scored run (the matched arm's `top_k` was
  calibrated once on a small separate sample, then frozen), and the ordering result was confirmed on a
  held-out set never used during development.

### 3) How it works

- Each turn, an LLM extracts durable facts, with the date they were said.
- Facts go into one local file, each with its own search vector.
- Before each reply the closest facts are retrieved, re-ranked, and injected as a labelled block.
- Questions about time or order also get a dated timeline.
- Any LLM, any embedder; writes go to an append-only log; only the LLM calls leave your machine.

By feijaodo — <https://github.com/feijaodo/MPMF-plus>.

---

A 5-plane binary graph memory for LLM agents: an on-disk format (nodes, edges,
per-node raw text / summary / wiki / vector), a retrieval layer, and a recall
layer that renders a memory block for injection into a prompt. This package is a
port of the code that was actually measured, not a re-implementation.

**Why:** on the author's deployment, 64 "in what order…" questions were fully correct (strict) in
1 of 64 without the history block and 26 of 64 with it (dev set, not held out; 0.3.1 re-run on the
r99 defaults); on a second frozen question set (held-out, same method as dev set) strict 5 of 64 vs
32 of 64; details, the other (lenient) score and caveats in *Measured results*.

> **Installing with an agent?** Follow [`AGENT_INSTALL.md`](AGENT_INSTALL.md)
> step by step — it covers host detection, `create` vs `attach`, the two hooks,
> and the mandatory self-test.

## Measured results

Earlier evaluations (rounds 56 and 70–73: summary-pipeline comparison, rebuilt-store
head-to-head, render and rerank ablations) are kept as aggregate-only artifacts in
`CHANGES/evidence/`. Their figures are **not restated here**: this release's figures are
checked against its own derived result rows, and those older numbers are not.

### 0.3.1 results (re-run on the live r99 defaults)

Both question sets were re-run for 0.3.1 with the same harness, answer prompt, 3-judge panel
(ties → partly), sequence check and scoring code as the 0.3.0 runs below
(`/root/ops-mpmf/heldout_eval.py`, md5 `560313817fe771071635a3f1386e8308`; answer model
`deepseek/deepseek-v4.1-flash`; judges deepseek-v4.1-flash, gemini-2.5-flash, gpt-4.1-mini —
unchanged from 0.3.0). The only change is the code under test: r99 (`9fc14bb`), where the history
predicate is ordering/date cues only and `history_additive` is on by default.

Config: the **live** `recall.config.json` was copied per arm; `cfg_on` is the live config unchanged
(so `history_additive` is absent → additive), `cfg_off` differs only in `"history_mode": false`.
Both arms refuse to run unless `provider._history_mode()` and `history.enabled()` read back as the
arm requests (`ARM_CONFIG` line per arm in `CHANGES/evidence/r100_order_rerun.json`).

Store: a fresh copy of the live store at run start, md5 `6f04d4ee3a5dbd2859367a9aae51576b`
(133,610,614 B), identical before and after every arm. The 0.3.0 snapshot file no longer exists on
this host, so these bytes are **not** the 0.3.0 bytes; both arms and both sets saw the same bytes
here. Prefetch-only coverage probe before scoring: history fired 64/64 on, 0/64 off, zero empty
blocks in all four arms.

| dev set, not held out (64) | off | on |
|---|---|---|
| score (correct + ½ partly) | **2.3%** | **62.5%** |
| strict (fully correct only) | **1 / 64 (1.6%)**, Wilson 95% CI 0.3–8.3% | **26 / 64 (40.6%)**, Wilson 95% CI 29.5–52.9% |
| correct / partly / declined / wrong | 1 / 1 / 58 / 4 | 26 / 28 / 4 / 6 |
| sequence check | **1 / 64** | **21 / 64** |

| held-out v2, same method as dev set (64) | off | on |
|---|---|---|
| score (correct + ½ partly) | **10.9%** | **66.4%** |
| strict (fully correct only) | **5 / 64 (7.8%)**, Wilson 95% CI 3.4–17.0% | **32 / 64 (50.0%)**, Wilson 95% CI 38.1–61.9% |
| correct / partly / declined / wrong | 5 / 4 / 53 / 2 | 32 / 21 / 1 / 10 |
| sequence check | **4 / 64** | **29 / 64** |

Exact McNemar, dev set: strict b=25 (off not correct, on correct), c=0, p = 5.96e-08; lenient
(correct or partly) b=53, c=1, p = 6.11e-15. Held-out v2: strict b=28, c=1, p = 1.12e-07; lenient
b=45, c=1, p = 1.34e-12. Spend for all four arms: **$0.431** (cap was $2), from the harness's own
token accounting. Question files unchanged: dev md5 `b1353104e875c623af3150da92cf70dd`, held-out v2
md5 `2e3da59e7bbe9cca7a1d74f4daf126e8`.

These numbers are lower than the 0.3.0 ones on the dev set (26/64 vs 32/64 strict on; 1/64 vs 0/64
off) and higher on held-out v2 (32/64 vs 30/64 on; same 5/64 off), and the store bytes differ, so
the two runs are not a controlled comparison: each is reported against its own snapshot. Both sets
of numbers are labelled; neither is substituted for the other. Aggregates and every md5:
`CHANGES/evidence/r100_order_rerun.json`; statistics script `CHANGES/evidence/r100_stats.py`.

### 0.3.0 results (as measured for 0.3.0; kept labelled, superseded by the 0.3.1 re-run above)

**Order questions ("in what order…").** 64 questions, the same question text and ids in both arms,
`history_mode` off vs on:

**Dev set, not held out:** the 68.0% (and every number in this table) is on a development
set. The history feature was built and tuned against these 64 questions.

| dev set, not held out | off | on |
|---|---|---|
| score (correct + ½ partly) | **0.8%** | **68.0%** |
| strict (fully correct only) | **0 / 64 (0%)** | **32 / 64 (50%)** |
| correct / partly / declined / wrong | 0 / 1 / 59 / 4 | 32 / 23 / 5 / 4 |
| sequence check (answer's event order judged right) | **3 / 64** | **24 / 64** |

**Held-out result: RETIRED.** The held-out set failed its harness check (both arms read the
live `history_mode`, so the "off" arm was not off) and its store coverage was 3/10 (dev control
9/10). No held-out number is reportable, and none is quoted here: the held-out result was retired because of the harness bug and the low store coverage.

**Held-out v2 (a second, frozen question set; the first held-out attempt above stays retired).**
64 questions (md5 `2e3da59e7bbe9cca7a1d74f4daf126e8`) whose entities are disjoint from the dev set,
generated and filtered before any scoring, same store snapshot for both arms, same scoring code as the
dev set. Each arm's config was read back and checked before scoring (`ARM_CONFIG` lines in the evidence);
the store md5 was identical before and after. Scored once per arm.

| held-out, same method as dev set | off | on |
|---|---|---|
| score (correct + ½ partly) | **11.7%** | **69.5%** |
| strict (fully correct only) | **5 / 64 (7.8%)**, Wilson 95% CI 3.4–17.0% | **30 / 64 (46.9%)**, Wilson 95% CI 35.2–58.9% |
| correct / partly / declined / wrong | 5 / 5 / 53 / 1 | 30 / 29 / 1 / 4 |
| declined | 53 / 64 | 1 / 64 |
| sequence check (answer's event order judged right) | **0 / 64** | **33 / 64** |

**Strict versus strict, dev set beside held-out v2** (strict = fully correct only; 64 questions each):

| strict (fully correct only) | off | on |
|---|---|---|
| dev set, not held out | 0 / 64 (0%) | 32 / 64 (50%) |
| held-out, same method as dev set | 5 / 64 (7.8%) | 30 / 64 (46.9%) |

Strict exact McNemar (held-out): b=26 (off not-correct, on correct), c=1, p = 4.2e-07. The held-out
strict gain (+25 questions, 7.8% to 46.9%) is in the same range as the dev-set gain (0% to 50%); the
intervals overlap. Remaining caveats: the on arm still leaves 29 of 64 answers only "partly" correct. The on-arm run was interrupted once by an operator-side tool timeout
after 8 questions were scored; it was resumed with the harness's row log, so those 8 rows were kept and
not re-scored (the other 56 were scored in the resumed run, same config read-back).

Strict McNemar (dev set) b=32, c=0, p < 1e-6. Caveats, all of which matter:

- The questions are drawn from topics with **2 to 5 distinct key dates** (2 dates: 4 questions;
  3: 30; 4: 23; 5: 7); this measures ordering where history exists, not a random question.
- n = 64, one author's data, one run per arm. This is a development set (the history feature was built and tuned against it), not a held-out set; the first held-out attempt is retired (see above) and the v2 held-out result is reported above. The 95% Wilson score interval on the strict 32/64 is **38.1–61.9%** (on the 0/64 off arm: 0–5.7%); computed by `CHANGES/evidence/r92_stats.py`.
- The 0.8% "off" figure is mostly *declines* (59 of 64): without the history block the
  system says it does not know. It is not mostly wrong answers.
- A nightly guard watches a separate HOLDOUT and reverts `history_mode` (and the
  reranker setting) on a >= 5.4-point drop with McNemar p < 0.10 (`REVERT_DROP` in
  `eval/nightly.py`). The separate nightly drift *alert* fires at >= 8 points, p < 0.10. No
  HOLDOUT result is quoted here.

Aggregates: `CHANGES/evidence/r92_order_history.json` (no question or answer text).

### r99 history A/B: DMR, LoCoMo temporal, BEAM

A separate A/B on three third-party question sets, run for the r99 change (detection narrowed to
ordering/date cues; the dated list additive instead of replacing the facts block). Three arms,
**one judge per benchmark** — this A/B ran a single judge per set (the order runs above used a
three-judge panel; `scripts/bench8/history_ab.py:48-55`) — same ids in every pair: **off**; **on as shipped before r99**
(`history_mode` on, broad cue list, dated list REPLACES the facts block); **r99** (ordering/date
cues only, additive).

| set | n | off correct (mean score) | replace, pre-r99 (mean score) | r99 additive (mean score) |
|---|---|---|---|---|
| DMR | 130 | 103 (0.654) | 87 (0.595) | **100 (0.638)** |
| BEAM event_ordering + temporal | 80 | 44 (0.385) | 28 (0.283) | **47 (0.430)** |
| LoCoMo temporal | 321 | 104 (0.138) | 151 (0.123) | **141 (0.141)** |

Exact McNemar (b = earlier arm wrong, later arm right; c = the reverse):

- off → replace: DMR b=3, c=19, p=0.00086; BEAM b=3, c=19, p=0.00086 (replace is a **regression** on both);
  LoCoMo b=63, c=16, p<1e-5 (on LoCoMo temporal, replace helped).
- off → r99: DMR b=1, c=4, p=0.375; BEAM b=7, c=4, p=0.549; LoCoMo b=46, c=9, p<1e-5.
- replace → r99: DMR b=19, c=6, p=0.0146; BEAM b=20, c=1, p=2.1e-05; LoCoMo b=17, c=27, p=0.174.

So r99 gives back the accuracy replace-mode lost on DMR and BEAM, keeps a significant gain over off on
LoCoMo temporal, and costs the extra list characters only where the block fires (non-firing DMR
questions: facts byte-identical to off, 109/109). On DMR the r99-minus-off difference is **not
detectable at this sample size**, not absent: b=1, c=4 is four discordant pairs, which cannot resolve a
change of a few points (p=0.375). Read that as absence of evidence, not evidence of equivalence — the
same applies to BEAM (b=7, c=4, p=0.549). Spend for the whole A/B (three arms × 531 questions, reader +
judge): **$1.40**.
Aggregates recomputed from each subset's `results.jsonl`: `CHANGES/evidence/hist_r99_ab.json`;
the report the run printed: `CHANGES/evidence/hist_r99_report.md`.

**Set files, for anyone re-running.** DMR and BEAM are read as in `scripts/bench8/`. LongMemEval
defaults to the **oracle** file — `bench/datasets.py:225` has `lme_file="longmemeval_oracle.json"`,
so a run that does not pass `lme_file=` explicitly is scoring the oracle set and must be labelled
`LME-oracle`; the non-oracle `S` label in `bench/prompts.py:112-114` applies only to runs that pass
the non-oracle file. No LongMemEval result is quoted in this README.

**Latency (prefetch), real sessions only.** On the author's deployment, the first 25 prefetches
made in real sessions, 2026-11-06 to 2026-11-10 (n = 25 calls from 3 sessions; benchmark,
scorecard, held-out harness, watchdog-probe and inspect calls excluded), took
**p50 1.81 s, p90 4.31 s**, max 5.02 s. That is the number to plan around: it includes the box
doing other work (cron jobs, the embedder shared with ingest). Caveats: 15 of the 25 calls are
from a single day (2026-11-06), the sample is one deployment, and percentiles are nearest-rank.
The log has grown since this window was cut; later calls are not included.

> Footnote (benchmark, not real sessions) — idle-host evaluation harness on the 64 dev-set order
> questions: prefetch p90 was 1,265 ms (history on) and 1,533 ms (off). This is a lower bound for an
> unloaded host and is not the latency to expect.

**The previous store had 26% mismatched vectors.** Before round 83e, 6,326 of 24,339 nodes
(26.0%) had a stored vector that disagreed with a fresh embedding of the node's own text
(cosine < 0.99; 4,350 below 0.90; mean 0.954). All were re-embedded and verified byte-exact.
Evidence:
`CHANGES/evidence/r83e_vector_audit.json`.

The evaluation data is personal and not published; aggregates only, with no questions,
answers, facts or conversation text, are in `CHANGES/evidence/`.

Artifacts, kept in this repository but **not shipped in the package**:
`CHANGES/evidence/r56_stats.json` (Round 56 judge results, aggregates only) and
`CHANGES/evidence/r59d_parity.json` + `r59d_parity.log` (Round 59d parity;
redacted to per-probe flags and totals).

## Switches

One row per config flag or numeric knob that changes behaviour or results. Every number below comes from
this README, `CHANGELOG.md` or a `steps/*.notes.md` file (named in the *measured* column); where nothing was
measured the cell says **not measured**. Labels follow *Measured results*: **dev set, not held out**,
**held-out**, **strict** (fully correct only). **RESEARCH** = built and measured, left off, not recommended.

How to toggle. Config keys live in `$MPMF_HOME/recall.config.json` (copy `config.example.json`). **file+env**:
the key is read through `paths.load_config`, so `MPMF_<KEY>` (upper-case, JSON value, e.g. `MPMF_HISTORY_MODE=0`)
overrides the file, and the env wins. **file**: read straight from the file on every call; there is no env
override. Neither needs a restart except where the row says so (the host adapter reads its keys once, at start).

| flag | what it does | measured | cost | default | toggle |
|---|---|---|---|---|---|
| `history_mode` | temporal questions get a date-ordered `## History of X` block; other questions render as before | order questions, strict: 0.3.1 re-run dev set, not held out 1/64 -> 26/64; held-out (v2) 5/64 -> 32/64 (0.3.0, as measured then: 0/64 -> 32/64 and 5/64 -> 30/64; *Measured results*). Non-temporal blocks: byte-identical off vs on for 23 of 23 `now`/`recent` questions; the 6 questions whose wording trips the temporal pattern get the history block (`steps/g2.notes.md`) | prefetch p90 1,265 ms on vs 1,533 ms off (idle host, dev set, benchmark footnote); money: not measured | **on** (since g2) | `history_mode` (file+env); `MPMF_HISTORY_MODE=0` or `"history_mode": false` turns it off; the nightly auto-revert writes `false` |
| `history_additive` | when the history block fires: **on** renders the facts block exactly as with history off and appends the dated list; **off** = the dated list REPLACES the facts block | DMR (130 q): off 103 -> 87 correct replace vs 100 additive (McNemar off-vs-replace p=0.00086, off-vs-additive p=0.375); BEAM event_ordering+temporal (80 q): 44 -> 28 replace vs 47 additive (p=0.00086 / p=0.549); LoCoMo temporal (321 q): 104 -> 151 replace vs 141 additive (off-vs-both p<1e-5); non-firing questions: facts byte-identical to history-off (109/109 DMR) (`CHANGES/evidence/hist_r99_ab.json`, *r99 history A/B*) | additive costs the extra list chars on firing questions (LoCoMo mean block 799 -> 1,624 chars); reader+judge spend for the whole A/B $1.40 | **on** (since r99) | `history_additive` (file+env); `"history_additive": false` or `MPMF_HISTORY_ADDITIVE=0` restores replace mode |
| `memory_arm` | recall arm: `d20` = cosine top-20, one cross-encoder pass, rank fusion; `production` = the 5-plane pipeline | not restated (earlier rounds, aggregate-only in `CHANGES/evidence/`) | not measured | `d20` in `DEFAULTS` and the live file; **`production` when the file exists without the key** (the provider reads the file directly) | `memory_arm` (file) |
| `memory_render` | `full` renders each memory's full text; `summary` renders the summary | not restated (render ablation, `CHANGES/evidence/`) | not measured | `full` when the config file is missing; **`summary` when the file exists without the key**; an invalid value or file raises `ConfigError` | `memory_render` (file) |
| `memory_item_chars`, `memory_block_chars` | per-item and per-block character caps of the rendered block | not measured | not measured | 1200 / 3000 (`full`) | same keys (file); not in `DEFAULTS` |
| `ns_prefilter` | restrict candidates to the requested namespaces before ranking, so excluded namespaces cannot take slots | not measured | not measured | on | `ns_prefilter` (file) |
| `rerank_skip_after_ms` | skip the cross-encoder when retrieval has already taken this long | not measured | not measured | `0` in `paths.DEFAULTS`, `4500` in `config.example.json` and in the provider's own fallback; the provider reads `0` as 4500 (`float(v or 4500)`), so `0` does not switch the reranker off. Found in g2, not changed (`steps/g2.notes.md`) | `rerank_skip_after_ms` (file) |
| `prefetch_budget_ms` | time budget for the rerank step of one prefetch | not measured as a knob; real-session prefetch p50 1.81 s, p90 4.31 s (n = 25), HOLDOUT prefetch p90 559 ms with `second_pass` on (`steps/g4.notes.md`) | not measured | 6500 | `prefetch_budget_ms` (file); `0` reads as 6500 |
| `rerank_clamp_chars` | clamp the candidate text fed to the cross-encoder (rendered text unchanged) | not measured | not measured | 0 (off) | `rerank_clamp_chars` (file) |
| `embed_query_cap_chars` | cap on the query text sent to the embedder | not measured | not measured | 1200 | `embed_query_cap_chars` (file) |
| `prewarm_store` | load and warm the store in the background | not measured | not measured | on | `prewarm_store` (file) |
| `prewarm_warm_query` | one retrieval-only warm-up query after a prewarm | not measured | not measured | on | `prewarm_warm_query` (file) |
| `prewarm_query` | also run a full recall as part of the prewarm | not measured | not measured | off | `prewarm_query` (file) |
| `prewarm_rss_guard`, `prewarm_rss_margin_mb` | memory governor for the prewarm | not measured | not measured | `auto` / 400 | same keys (file); not in `DEFAULTS` |
| `cos_pin_k` | keep the top-k cosine hits in the final list | not measured | not measured | 0 (off) | `cos_pin_k` (file); not in `DEFAULTS` |
| `cognition_rank` | lesson ranking mode in the `<prior-reasoning>` path: `lexical` or `embed` | not measured | not measured | `lexical` | `cognition_rank` (file) |
| `cognition_floor` | lessons scoring below this are dropped (0 keeps all) | not measured | not measured | 0.0 | `cognition_floor` (file) |
| `cognition_min_cos` | minimum cosine for a lesson on the cosine arm | not measured | not measured | 0.65 | `cognition_min_cos` (file) |
| `inject_thinking` | add the thinking-trace block to the prompt | not measured | not measured | on | `inject_thinking` (file+env, read at start) |
| `thinking_top_k`, `thinking_max_chars` | traces shown / characters allowed in that block | not measured | not measured | 3 / 800 | same keys (file+env, read at start) |
| `inject_cognition` | add the `<prior-reasoning>` block to the prompt | not measured | not measured | on | `inject_cognition` (file+env, read at start) |
| `cognition_top_k`, `cognition_max_chars` | lessons shown / characters allowed in that block | not measured | not measured | 3 / 700 | same keys (file+env, read at start) |
| `top_k`, `max_chars` | memories in the block / characters allowed in it | not measured | not measured | 8 / 3500 | same keys (file+env, read at start) |
| `auto_write` | store each completed turn as memory | not measured | not measured | on | `auto_write` (file+env, read at start) |
| `turn_namespace` | namespace the per-turn writes go to | not measured | not measured | `transcript` | `turn_namespace` (file+env, read at start) |
| `pre_tool_causal` | surface causal and meta lessons before a side-effect tool call | not measured | not measured | on | `pre_tool_causal` (file+env, read at start); not in `DEFAULTS` |
| `user_names` | the user's own first name(s): history stop-words and the extraction-prompt example | not measured | none | `[]` | `user_names` (file) |
| `show_said_date` | add the said-date to each rendered item | not measured | not measured | off | `show_said_date` in the file is **not read** by `render_cfg` since r89c, so it cannot be switched on (`steps/d.notes.md`) |
| `raw_layer` | **RESEARCH.** the user's raw messages as their own searchable layer, appended as a separate section | 348 questions (r85b 250 + TUNE 98), dev, not held out: accuracy 0.4813 -> 0.6466, strict 143 -> 198, McNemar p = 5.2e-11; upper bound, because the questions were written from these messages (`steps/p8.notes.md`) | prefetch p50 475 -> 986 ms, p90 550 -> 1436 ms; about $0.20 per arm; needs a backfill of 16,678 raw nodes | off | `raw_layer` (file+env) |
| `raw_k`, `raw_min_cos`, `raw_rerank`, `raw_item_chars`, `raw_block_chars` | knobs of the raw layer: hits, cosine floor, rerank, item and block caps | not measured separately | not measured | 3 / 0.5 / on / 500 / 1500 | same keys (file+env) |
| `keep_details` | **RESEARCH.** re-ask the extractor once when a fact drops a name, number or date of its evidence | 80 judged turns (two samples, not a question set): specificity 4.000 -> 4.008, retry fired on 14 of 80 turns (`steps/p9.notes.md`) | one extra extraction call on about 17% of turns | off | `keep_details` (file+env) |
| `lessons_at_action` | **RESEARCH.** show a lesson at a completion claim that lacks a tool call or numbered items | frozen 38-task set (not a question set): 12/38 off vs 17/38 on, McNemar p = 0.2668, no effect (`steps/p11b.notes.md`) | about $0.32 off vs $0.46 on, 38 tasks | off | `lessons_at_action` (file+env) |
| `typed_walk` | **RESEARCH.** one-hop walk over `IS_A` / `DERIVED_FROM` / `supersedes` edges from the top hits | MAIN n = 40, single set, no held-out run recorded: strict 11 -> 14, p = 0.25; both source facts in the block 12 -> 20, p = 0.039 (`steps/p12.notes.md`) | walk 0.4 ms mean; prefetch p50 582.5 -> 579.3 ms | off | `typed_walk` (file+env) |
| `typed_walk_k`, `typed_walk_seeds`, `typed_walk_min_cos` | knobs of the walk: neighbours kept, seed hits, cosine floor | not measured separately | not measured | 2 / 4 / 0.3 | same keys (file+env) |
| `second_pass` | **RESEARCH.** the `mpmf_recall` tool returns the prefetch rendering, never repeats a memory, capped per turn | held-out (HOLDOUT, 74 questions), strict: shipped tool 33/74 vs flag on 29/74, McNemar b = 0, c = 4, p = 0.125, no effect (`steps/g4.notes.md`); TUNE, dev: no tool 0.434 / 0.424 -> new tool 0.510 / 0.510, shipped tool 0.520 / 0.526 (`CHANGELOG.md`) | prefetch p90 559 ms; tool path extra mean 733 ms, p90 1180 ms; about $0.050 vs $0.053 per arm | off | `second_pass` (file+env) |
| `second_pass_max_calls`, `second_pass_k` | knobs of the second pass: calls per turn, hits per call | not measured separately | not measured | 2 / 6 | same keys (file+env) |
| `append_segments` | append one small segment per save instead of rewriting the whole image | 131 MB store: median write 776 ms -> 42 ms, 131 MB -> 4 KB written (`CHANGELOG.md`) | older code refuses a file that still has segments; `python -m mpmf.compact` folds them back, and `./deploy.sh rollback <tag>` does that (verified) before it moves the live tree | off | `append_segments` (file+env), `MPMF_APPEND_SEGMENTS=1` |
| `compact_after_segments` | fold the segments into a new base after this many | not measured | not measured | 5000 | `compact_after_segments` (file+env) |
| `auto_apply` | let the proposer apply a TUNE+HOLDOUT winner to one registry key; the nightly reverts it on a HOLDOUT drop | simulated only (`tests/test_autoapply.py`); no live result | not measured | off | `auto_apply` (file+env) |
| `thinking_capture_enabled` | write thinking traces from sessions of ANY model into the thinking store (skips are counted by reason and model) (own file, `thinking_capture.config.json`) | not measured | not measured | off | `thinking_capture_enabled` in that file |

**Which agents inherit the `history_mode` default.** The default only matters where a config file has no
`history_mode` key, and the nightly revert no longer removes the key (it writes `false`). On the author's
deployment, three agents load this package tree (checked by reading their code and process environments, not by writing to anything): Athena (the Hermes gateway plugin), Espy (the HTTP memory
server) and Ayu (the Telegram bot on the harness). None of the three sets `MPMF_HOME` or `MPMF_RECALL_CONFIG`
in its process environment, so all three resolve the same `~/.mpmf/recall.config.json`, which already has
`"history_mode": true`. All three therefore inherit **on**, and nothing changes for any of them at the next
restart (the code ships with the next deploy; the file already says `true`, so the live behaviour is the same before and after). Only Athena renders the memory block (`prefetch` -> `render_memory_block`), so only her answers can
carry a history block; Espy's server returns recall hits and Ayu's harness uses the package for ingest, and
in the code read, neither calls the block renderer, before or after.

## Install

Python **3.10 or newer** (`requires-python = ">=3.10"`).

```bash
pip install .                 # core only (needs requests)
pip install ".[rerank]"       # + sentence-transformers, for the cross-encoder
pip install ".[fast]"         # + numpy, for the matrix path
```

From git (tagged, no PyPI publication yet):

```bash
pip install "git+https://github.com/feijaodo/MPMF-plus.git@v0.3.2"
```

### What gets installed

| path | what it is |
|---|---|
| `src/mpmf/` | the core format + retrieval (`format.py`, `memory.py`, `retrieval.py`, `query.py`, `embedding.py`, `openai_embed.py`, `ingestion/`) |
| `src/mpmf_plus/memory.py` | port of the live store/recall wrapper at this release (0.3.2); path/name changes only, no logic changes — `CHANGES/diff/01_memory_port.diff` |
| `src/mpmf_plus/block.py` | port of the live block builder (`render_cfg`, `wrapper_recall`, the render loop); mechanical extraction, no logic changes — `CHANGES/diff/02_block_and_adapter.md` |
| `src/mpmf_plus/store.py` | store resolution and `store_mode = create \| attach` (shared memory bank) |
| `src/mpmf_plus/selftest.py` | `python -m mpmf_plus.selftest --store PATH` — the install gate |
| `src/mpmf_plus/hermes_adapter.py` | the only Hermes-coupled file; holds no memory logic |
| `src/mpmf_plus/errors.py` | the three failure types below |
| `CHANGES/` | port diff, before/after sources, manifest, `evidence/` (in the repo, not shipped in the package) |
| `examples/store_and_recall.py` | offline end-to-end example (temp store, no model) |

`memory.py` contains unused lesson-injection internals inherited from the author's agent; they are inactive and are scheduled for removal.

## Requirements

**Embedder — Ollama, `mxbai-embed-large` (1024-d), required for real recall.**
The client posts to `http://127.0.0.1:11434` and uses the canonical model name
with no padding. Point it at another host or port with the `OLLAMA_URL`
environment variable (there is no config-file key: it is env-only, read at
import time). The model must already be pulled:

```bash
ollama pull mxbai-embed-large
```

**Reranker — `cross-encoder/ms-marco-MiniLM-L-6-v2`, downloaded on first use.**
It runs on CPU (`max_length=512`) through `sentence-transformers`, and the
weights are fetched from the Hugging Face Hub the first time a rerank
happens, so that one call needs network access; afterwards it loads from
`~/.cache/huggingface`. Without the optional dependency the rerank stage is
skipped and recall degrades to cosine order (arm C behaviour), it does not
silently fail.

**Disk:** `sentence-transformers` itself is small but it pulls torch and
transformers, which are large (a CPU-only torch wheel is smaller). The core (no rerank extra)
needs only `requests`.

**Not required:** numpy (optional, `[fast]`), a GPU (everything is CPU), any API
key. The example and the test suite run with no model and no network.

## Quickstart

The whole offline flow — write three facts to a throwaway store, reopen it, recall
and render the block. Runs as written, no Ollama, no network:

```python
import json, os, tempfile
d = tempfile.mkdtemp(); os.environ["MPMF_HOME"] = d
os.environ["MPMF_STORE_PATH"] = os.path.join(d, "demo.mpmf")
json.dump({"memory_render": "full"}, open(os.path.join(d, "recall.config.json"), "w"))
from mpmf.memory import MemoryManager
from mpmf_plus import block
store = MemoryManager(os.environ["MPMF_STORE_PATH"], vector_dimensions=64); store.load()
with store.write_txn():                    # a write outside this guard is not saved
    store.create_node(name="backup_time", node_type=0, plane_flags=1, text="The nightly backup runs at 02:00 UTC.",
                      summary="backup at 02:00 UTC", wiki={"_ns": "memory"}, importance=0.5)
store = MemoryManager(os.environ["MPMF_STORE_PATH"], vector_dimensions=64); store.load()
nid = [i for i in store.all_node_ids() if store.get_name(i) == "backup_time"][0]
shim = block.RecallShim(); shim.top_nodes = [nid]
shim.raw_texts = shim.summaries = {nid: "The nightly backup runs at 02:00 UTC."}
print(block.render_memory_block("when does the backup run?", shim, store, top_k=1)[0])
```

The recall here is a hand-built shim on purpose: real recall is
`mpmf_plus.memory.recall()`, which needs the Ollama embedder above. See
`examples/store_and_recall.py` for the full commented version.

## Configuration

Paths come from the environment or from the package's own location; none is
hardcoded to a filesystem path:

| variable | default | meaning |
|---|---|---|
| `MPMF_HOME` | `~/.mpmf` | base directory for store, cache, state, health |
| `MPMF_STORE` | `$MPMF_HOME/store.mpmf` | the store file (preferred name) |
| `MPMF_STORE_PATH` | `$MPMF_HOME/store.mpmf` | legacy alias for `MPMF_STORE`; `MPMF_STORE` wins |
| `MPMF_STORE_MODE` | `create` | `create` or `attach` — see *Shared memory bank* |
| `MPMF_STORE_LOCK_TIMEOUT` | `60` | seconds to wait for the cross-process write lock before `LockTimeoutError` |
| `MPMF_RECALL_CONFIG` | `$MPMF_HOME/recall.config.json` | recall/render config |
| `MPMF_APPEND_SEGMENTS` | off | `1` appends each write as a small segment instead of rewriting the whole file (same as `append_segments` in recall.config.json); fold back with `python -m mpmf.compact STORE` |
| `MPMF_CORE_PATH` | the bundled `mpmf` core | override only; by default the core shipped inside this package is used, and nothing outside it is imported |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | where the embedder lives |

The same two store settings also exist as config keys in
`$MPMF_HOME/recall.config.json`: `"store_path"` and `"store_mode"`.

**Nothing fails silently.** A missing or unreadable store raises
`StoreNotFoundError`, a missing/invalid config raises `ConfigError`, and a dead
or empty embedder raises `EmbedderError`. There is no fallback to another file
and no "return empty on error" (`src/mpmf_plus/errors.py`;
`tests/test_no_silent_fallbacks.py`).

### Shared memory bank

Several agents — Hermes, OpenClaw, a cron script — can use one store. Two
settings decide which file it is and whether this process may create it:

```bash
export MPMF_STORE=/srv/shared/bank.mpmf      # the one file everyone opens
export MPMF_STORE_MODE=attach                # join an existing bank
# first agent only: MPMF_STORE_MODE=create
```

| `store_mode` | behaviour when the file is missing |
|---|---|
| `create` | creates it once, with a line on stderr; refuses to overwrite an existing store |
| `attach` | raises `StoreNotFoundError` and creates **nothing** — no empty store, no new file, no fallback path |

Precedence: argument (`--store`, `configure_store()`) → `MPMF_STORE` /
`MPMF_STORE_PATH` / `MPMF_STORE_MODE` → config keys `store_path` / `store_mode`
→ the defaults above. An unknown or blank mode raises `ConfigError`; a blank
path raises `ConfigError` rather than silently becoming the default.

Writers take an exclusive `flock` on `<store>.lock` and re-read the image if
another process saved in the meantime, so two agents writing at once do not
erase each other's nodes (stress logs in `CHANGES/evidence/`). A single
`store()` holds that lock across an embed round-trip, hence the 60 s default
wait; a genuinely stuck lock still raises `LockTimeoutError`.

Same machine only — `flock` over a network filesystem is not safe. All agents
must use the same embedder. Since 0.2.0 the store records the model and
dimension it was built with (a reserved wiki node, `_mpmf_embedder`, namespace
`meta`, so it never appears in recall): a store with **no** record — every
v0.1.x store — is accepted and stamped on its first write, and a store recorded
with a different model or dimension raises `ConfigError` naming both. Nothing is
padded, converted or silently re-embedded.

### Write-time scaling

`store()` rewrites the whole image, so the cost of one write grows with the
store rather than with what changed. Two costs dominate: the O(N) dedup scan over every
node, and serialising + fsyncing + renaming the whole file. The Ollama embedding
round-trip is paid outside the write lock. No timing table is given here (the earlier
one is not backed by this release's evidence). **Available since 0.3.1, opt-in and off by
default:** an append-only record log (`append_segments_stores`, per store; `MPMF_APPEND_SEGMENTS`
or `MemoryManager(append_segments=True)` also arm it) where a write appends one record instead of
re-writing the image, and full compaction moves to a background task (`mpmf.compact`). A store is
only ever segmented if its resolved real path is listed, so the default stays a plain image. Any
process reading a segmented store must be on 0.3.1 or later; `./deploy.sh rollback <tag>`
compacts first for exactly that reason. Until you arm it, keep a shared bank in the low tens of
thousands of nodes, or accept multi-second writes.

## Claims and their artifacts

Every claim below is traceable to a file. Nothing else is claimed.

| # | claim | artifact |
|---|---|---|
| 1 | The port is mechanical: the shipped core and wrapper are the measured code, not a rewrite | `CHANGES/manifest.md`, `CHANGES/diff/01_memory_port.diff`, `CHANGES/diff/02_block_and_adapter.md`, `CHANGES/port_report.json` |
| 2 | Failures are loud: the three exception types are raised, not swallowed | `src/mpmf_plus/errors.py`, `src/mpmf_plus/memory.py` (`_open`), `tests/test_no_silent_fallbacks.py` |
| 3 | **The write-lock works at package level.** Concurrent writers through `mpmf_plus.memory._write_txn_open` into one store copy lost no nodes (see the artifact), and a writer holding a stale image is refused with `StaleImageError` instead of silently overwriting another writer's nodes | `CHANGES/evidence/r59e_stress.log` (Round 59e, step 1) |
| 4 | Config is environment-driven: store/recall defaults are the historical ones, and the core path defaults to the core bundled in this package | `src/mpmf_plus/memory.py`, `src/mpmf_plus/block.py`, `config.example.json` |

Claim 3 covers only concurrent writers against one file, not multi-host
deployment.

## Deliberately not claimed

- No claim that the graph improves answers.
- No latency guarantee: real-session prefetch p90 was 4.31 s over 25 calls (see *0.3.0 results*); the
  idle-host bench is roughly a third of that.
- No multi-process deployment claim beyond the write-lock test in claim 3.
- No quality claim about edges or the "Related" hop; the eval harnesses used in
  earlier rounds live outside this package and are not shipped, so their
  numbers are not restated here.

## Contributing

Contributions require a signed Contributor License Agreement granting the author
the right to relicense — see `CONTRIBUTING.md`. Unsigned contributions are not
merged.

## Licence

PolyForm Noncommercial License 1.0.0 — free for noncommercial use; commercial use
requires a separate licence. Full text in [`LICENSE`](LICENSE).

## Install, deploy and roll back

### Install from the wheel (fresh environment)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install mpmf_plus-0.3.2-py3-none-any.whl          # or: pip install .
export MPMF_HOME=/path/to/state                       # store, config, logs, eval live here
python -m mpmf_plus.selftest --store "$MPMF_HOME/store.mpmf" --mode create
```

`selftest` writes a marker, reads it back, recalls it and renders it; exit 0 = pass.
Write `{"memory_render": "full"}` to `$MPMF_HOME/recall.config.json` first (the renderer refuses to
guess a missing config; `AGENT_INSTALL.md` §3). Real recall needs the Ollama embedder (see
*Requirements*); the selftest takes it from `OLLAMA_URL`. Nothing is read from or written to any directory outside `MPMF_HOME` and
the Python environment.

Useful entry points: `python -m mpmf_plus.ingest` (one turn -> facts),
`python -m mpmf_plus.eval {scorecard,nightly,proposer}`, `mpmf-rebuild`.

### Deploy (host-side, with the gate)

The deployed tree is a git worktree pinned to a tag; the running agents import from it and
never execute the development checkout. Develop only in the release repository.

```bash
./deploy.sh --check-only          # run the gate on the working tree, move nothing
./deploy.sh deploy-<n> [<commit>] # gate the commit in a detached worktree, tag it, switch the live worktree
```

The gate runs, in order: host-contract test, auto-revert pairing/revert test, real-turns-only
test, re-enrich accounting test, end-to-end store growth (`sync_turn` must grow the store),
then the 7 cron dry-runs. Any failure prints `DEPLOY REFUSED` and the live tree is left
untouched; deploy also refuses while the live worktree has local edits. `deploy.sh` is
host-specific: set `MPMF_LIVE` (deploy-target worktree) and `MPMF_PY` (interpreter).

### Roll back

```bash
./deploy.sh rollback <previous deploy tag>    # compacts a segmented store FIRST, then checks out
# then restart the host process that imports the package (e.g. the gateway /restart)
```

Do not roll back with a bare `git checkout --detach <tag>`: code older than 0.3.1 cannot read a
segmented store, and the checkout alone leaves the store in a format that process cannot open.
`deploy.sh rollback` resolves the tag, compacts the store under its write lock (verifying counts,
name index and the node/edge/vector signature before the rename), writes `"append_segments": false`
for that store, and only then moves the tree.

Back up a store before any change that touches it, with a copy kept outside the live
directory; a store swap (`mpmf_plus.swap_store`) keeps the old file.
