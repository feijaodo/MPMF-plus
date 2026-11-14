"""b0 judges. Prompt texts are the official ones (sources named per prompt); each judge returns
{"verdict": 0|1|None, "score": float|None, "cost": float, "raw": str}. ``None`` = the judge call or its
parse failed; such a (question, arm) is left out of the statistics, never counted as wrong.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from . import textmetrics as T
from .datasets import Question

# mem0 paper, appendix A (LLM-as-a-judge prompt, adapted by them from the MemGPT prompt).
MEM0_JUDGE = """Your task is to label an answer to a question as "CORRECT" or "WRONG". You will be given the following data: (1) a question (posed by one user to another user), (2) a 'gold' (ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG.
The point of the question is to ask about something one user should know about the other user based on their prior conversations. The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.
For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like 'last Tuesday' or 'next month'), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.
Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}
First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.
Just return the label CORRECT or WRONG in a json format with the key as "label"."""

# MemGPT paper appendix (DMR judge prompt, memgpt_raw.txt ~1190-1225); personas/old-chat preamble omitted.
MEMGPT_JUDGE = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data: (1) a question (posed by one user to another user), (2) a 'gold' (ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG.
The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.
For example, the following answers would be considered CORRECT:
Generated answer (CORRECT): Oh yeah, that was so fun! I got so much stuff there, including that shell necklace.
Generated answer (CORRECT): I got a ton of stuff... that surfboard, the mug, the necklace, those coasters too..
Generated answer (CORRECT): That cute necklace
The following answers would be considered WRONG:
Generated answer (WRONG): Oh yeah, that was so fun! I got so much stuff there, including that mug.
Generated answer (WRONG): I got a ton of stuff... that surfboard, the mug, those coasters too..
Generated answer (WRONG): I'm sorry, I don't remember what you're talking about.
Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}
First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script."""

# LongMemEval src/evaluation/evaluate_qa.py get_anscheck_prompt, templates verbatim.
_LME_BASE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
             "contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or "
             "contains all the intermediate steps to get the correct answer, you should also answer yes. If the response "
             "only contains a subset of the information required by the answer, answer no. ")
_LME_TAIL = " \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
LME_TEMPLATES: Dict[str, str] = {
    "default": _LME_BASE + _LME_TAIL,
    "temporal-reasoning": _LME_BASE + "In addition, do not penalize off-by-one errors for the number of days. If the question "
        "asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days "
        "when the answer is 18), the model's response is still correct. " + _LME_TAIL,
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
        "response contains the correct answer. Otherwise, answer no. If the response contains some previous information "
        "along with an updated answer, the response should be considered as correct as long as the updated answer is the "
        "required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? "
        "Answer yes or no only.",
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a response from a "
        "model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not "
        "need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's "
        "personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? "
        "Answer yes or no only.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the "
        "model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or "
        "some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel "
        "Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.",
}
_LME_DEFAULT_TYPES = ("single-session-user", "single-session-assistant", "multi-session")


def lme_prompt(qtype: str, question: str, answer: str, response: str, abstention: bool) -> str:
    if abstention:
        t = LME_TEMPLATES["abstention"]
    elif qtype in _LME_DEFAULT_TYPES:
        t = LME_TEMPLATES["default"]
    elif qtype in LME_TEMPLATES:
        t = LME_TEMPLATES[qtype]
    else:
        raise NotImplementedError("LongMemEval question_type %r" % (qtype,))
    return t.format(question, answer, response)


# BEAM src/prompts.py unified_llm_judge_base_prompt (verbatim; <rubric_item> / <llm_response> placeholders).
BEAM_JUDGE = """
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate. 

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT
A compliant response must be **on-topic** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., "$68,000", "68k", "68,000 USD", or "sixty-eight thousand dollars"). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., "itemized list", "no citations", "one sentence").
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.
  
- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.
  
- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)
   
3. **Check Compliance**: 
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}

NOTE: ONLY output the json object, without any explanation before or after that
"""

Chat = Callable[..., Dict[str, Any]]


def parse_label(text: str) -> Optional[int]:
    """CORRECT/WRONG from a json {"label": ...} reply or, failing that, the last bare keyword."""
    m = re.search(r'"label"\s*:\s*"?\s*(CORRECT|WRONG)', text or "", re.I)
    if m:
        return 1 if m.group(1).upper() == "CORRECT" else 0
    hits = re.findall(r"\b(CORRECT|WRONG)\b", (text or "").upper())
    if hits:
        return 1 if hits[-1] == "CORRECT" else 0
    return None


def parse_beam_score(text: str) -> Optional[float]:
    m = re.search(r'"score"\s*:\s*"?([01](?:\.\d+)?)', text or "")
    if not m:
        return None
    v = float(m.group(1))
    return v if v in (0.0, 0.5, 1.0) else None


def _res(verdict, score, cost, raw) -> Dict[str, Any]:
    return {"verdict": verdict, "score": score, "cost": cost, "raw": raw}


def judge(q: Question, answer: str, chat: Chat, model: str) -> Dict[str, Any]:
    """Score one answer with the benchmark's judge. Never raises: failures give verdict None."""
    cost = 0.0
    try:
        if q.bench in ("locomo", "dmr"):
            tpl = MEM0_JUDGE if q.bench == "locomo" else MEMGPT_JUDGE
            r = chat(tpl.format(question=q.question, gold_answer=q.gold, generated_answer=answer), model, max_tokens=200)
            cost += r["cost"]
            v = parse_label(r["text"])
            if q.bench == "locomo":
                sc = T.locomo_f1(int(q.extra.get("cat", 4)), answer, q.gold)
            else:
                sc = T.rouge_l(answer, q.gold)["r"]
            return _res(v, sc, cost, r["text"])
        if q.bench == "longmemeval":
            p = lme_prompt(q.category, q.question, q.gold, answer, bool(q.extra.get("abstention")))
            r = chat(p, model, max_tokens=10)
            cost += r["cost"]
            return _res(1 if "yes" in r["text"].lower() else 0, None, cost, r["text"])
        if q.bench == "beam":
            items: List[str] = q.extra.get("rubric") or []
            if not items:
                return _res(None, None, cost, "no rubric")
            scs: List[float] = []
            for it in items:
                r = chat(BEAM_JUDGE.replace("<rubric_item>", it).replace("<llm_response>", answer), model, max_tokens=400)
                cost += r["cost"]
                s = parse_beam_score(r["text"])
                if s is None:
                    return _res(None, None, cost, "unparseable rubric score: " + r["text"][:120])
                scs.append(s)
            mean = sum(scs) / len(scs)
            return _res(1 if mean >= 0.5 else 0, mean, cost, json.dumps(scs))
    except Exception as e:  # noqa: BLE001
        return _res(None, None, cost, "judge error: %s: %s" % (type(e).__name__, str(e)[:200]))
    raise ValueError(q.bench)
