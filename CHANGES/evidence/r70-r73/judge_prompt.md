# Judge prompt (verbatim, rounds 70–73)

Every answer was judged **alone** — a judge never saw two systems' answers in one
call, and never saw a system name. Three judges per answer; the majority verdict
was used, ties broken toward the stricter verdict.

```
Judge a single answer produced by a memory assistant. You do not know which system
produced it; judge it on its merits.

QUESTION: {question}
ANSWER KEY (ground truth): {key}
SOURCE MESSAGE / EVIDENCE: {source message}
ANSWER UNDER TEST: {answer}

Verdicts:
- "declined": the answer says DECLINED or refuses/returns nothing
- "correct": it conveys the key's substance
- "partly": related but misses or blurs a specific
- "wrong": conflicts with the key or answers something else
If the question asks how something changed over time, also judge the chronology
("seq_ok": "yes"/"no"; otherwise "na").
Return JSON {"verdict":"correct|partly|wrong|declined","seq_ok":"yes|no|na",
             "why":"<=12 words"}
```

Models used: `deepseek/deepseek-v4.1-flash`, `google/gemini-2.5-flash`,
`openai/gpt-4.1-mini`. Temperature 0, `response_format=json_object`,
`reasoning.enabled=false`.

Scoring: correct = 1.0, partly = 0.5, wrong = 0.0, declined = 0.0.
McNemar exact test on the paired majority verdicts (correct vs not-correct).
