# Round 70–73 evidence (aggregates only)

No questions, answers, facts or conversation text. Aggregates, judge prompt and
checksums only. Personal evaluation data stays out of this repository.

## Head-to-head: rebuilt store vs previous store (129 frozen questions)

| set | n | rebuilt | previous | McNemar b/c | p |
|---|---|---|---|---|---|
| history | 60 | 33.3% | 8.3% | 2/18 | 0.0004 |
| now | 20 | 30.0% | 5.0% | 0/5 | 0.0625 |
| then | 19 | 57.9% | 10.5% | — | — |
| order | 15 | 0.0% | 0.0% | 0/0 | 1.0 |
| recent | 15 | 0.0% | 6.7% | 1/0 | 1.0 |
| **all** | **129** | **27.9%** | **5.4%** | **4/33** | **<0.0001** |

Correct / wrong / declined, production-faithful render (3,000-char block cap,
`<thinking-context>` appended), 3 judges per answer, each answer judged alone.

## Render ablation (same 129, on the rebuilt store)

| config | correct | mean score |
|---|---|---|
| 3,000 cap + thinking | 27.9% | 0.322 |
| 3,000 cap, no thinking | 30.2% | 0.333 |
| 6,000 cap + thinking | 29.5% | 0.333 |
| 6,000 cap, no thinking | 30.2% | 0.329 |

No block ever exceeded 3,000 chars (max 1,707 across all 129 questions), so the
cap half of the ablation is a null result; the thinking block costs 2.3 points
(p≈0.51, not significant).

## Rerank clamp (`rerank_clamp_chars`)

| setting | correct | rerank median | p90 |
|---|---|---|---|
| off (0) | 30.2% | 244 ms | 365 ms |
| 400 chars | 27.1% | 247 ms | 354 ms |

McNemar 6/2, p=0.289. Clamp buys no measurable speed on this corpus (candidate
texts are already short) and is not significantly worse; default stays 0.

## Rerank cost vs candidate length (CPU, 20 candidates)

| chars | 200 | 600 | 1200 | 3000 |
|---|---|---|---|---|
| ms (2 threads) | 151 | 421 | 758 | 2122 |
| ms (4 threads) | 1116 | 1218 | 1104 | 2898 |

Threads make it worse on a 4-core box; the cost driver is candidate text length.

## Merge reconciliation

| item | count |
|---|---|
| rebuilt store nodes | 16,540 |
| previous store nodes | 7,258 |
| duplicates merged | 177 |
| imported | 7,081 |
| merged total | 23,621 (later 24,296 after the delta import) |
| must-survive probes | 20/20 |
| duplicate recall hits after supersede pass | 59 pairs marked, 0 returned |

## Store checksums

| store | md5 | bytes | nodes |
|---|---|---|---|
| rebuilt v3 | `b315eee14bf3b032163837b841894b38` | 78,673,805 | 16,540 |
| rebuilt v3.1 | `06f0368f09a1b3bacf76674dc1a9f3be` | 83,052,091 | 16,540 |
| rebuilt v3.2 (re-tagged) | `119e02ace818d6ec13948221e1f1851d` | 83,221,391 | 16,540 |
| merged | `1203bb28ab293bac7830261328f263ae` | 120,787,569 | 23,621 |
