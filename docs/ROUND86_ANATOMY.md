# Round 86 — miss anatomy (propose-only)

Questions: `/root/r85b/questions.json`, md5 d6192faafb6466cc0086b0b8a1c27193, n=250.
Baseline: `/root/r85b/base.jsonl` (TUNE 0.488 / HOLDOUT 0.480).

## Anatomy of the 117 non-correct core answers (history/now/then/order)

| set | not in store | outside cos top-40 | cut by rerank/RRF | in block, wrong/declined |
|---|---|---|---|---|
| history | 28 | 1 | 15 | 8 |
| now | 8 | 1 | 13 | 5 |
| then | 1 | 0 | 7 | 1 |
| order | 19 | 0 | 5 | 5 |
| **ALL** | **56 (48%)** | **2 (1.7%)** | **40 (34%)** | **19 (16%)** |

Method: every non-correct answer replayed against the copy store; holder found by
key-token fuzz over all 24,328 live nodes unioned with embedding top-200, then its
d20 cosine rank and presence in the injected top-8 recorded.

## Reasons extraction missed (56 not_in_store)

- 41 extraction granularity — a related node stored a generic/summary fact and dropped
  the specifics (e.g. "User likes making jewelry" vs. corvid necklace/harness question).
- 12 question/hypothetical framing — the source is a question or conditional, not an
  actual fact, so the extractor did not store it.
- 3 never ingested — no related node at all.
- 2 question-generation errors — the key is not in the source message.

## Keyword arm decision

`outside_cos_top40` is 2/117 (1.7%), so the pre-registered trigger ("large") is NOT met.
No BM25 arm was built; the dominant retrieval loss is rerank/RRF cutting a holder that
cosine already ranked <= 40 (40 cases), which keyword recall does not address.

## Recent / namespace finding

Ingest writes the user's session facts into **transcript**, not **memory**
(memory 19,945 nodes; transcript 2,122; ruminate 2,062). `recall()` defaults to
`namespaces=('memory',)`, so those facts are invisible to prefetch.

Fresh probe, 12 questions written from the user's messages after 2026-11-03:

- memory-only context: **acc 0.000**, mean key-token hit 0.01
- memory+transcript context: **acc 0.833**, mean key-token hit 0.73

Proposal (not applied): either add `transcript` to the recall namespaces or make the
turn ingester write durable facts into `memory`. The former is a config-only change;
the latter needs a namespace policy decision.

## d20_k 20→40

No win: CORE +1.36 pts, McNemar p=1.0, below the 5.4-pt noise floor. No HOLDOUT.