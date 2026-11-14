"""b0 reader prompts (one per benchmark, identical for every arm) and the per-benchmark documentation of
how this harness differs from the published setups. Prompt texts that are quoted from an official source
say so; the reader sees the same text for BM25, MPMF history-off and MPMF history-on.
"""
from __future__ import annotations

from typing import Dict

from .datasets import Question

# LoCoMo official QA prompts: task_eval/gpt_utils.py (QA_PROMPT, QA_PROMPT_CAT_5).
_LOCOMO_QA = ("\nBased on the above context, write an answer in the form of a short phrase for the following "
              "question. Answer with exact words from the context whenever possible.\n\nQuestion: {} Short answer:\n")
_LOCOMO_QA5 = "\nBased on the above context, answer the following question.\n\nQuestion: {} Short answer:\n"

# BEAM official RAG answer prompt: src/prompts.py answer_generation_for_rag.
_BEAM_RAG = """
You are an assistant that MUST answer questions using ONLY the information provided in the context below. 

STRICT INSTRUCTIONS:
1. Answer ONLY based on the provided context
2. Do NOT use your internal knowledge

CONTEXT:
<context>

QUESTION:
<question>

ANSWER REQUIREMENTS:
- Be direct and concise
- Only output the answer to the question without any explanation 

RESPONSE:
"""

# LongMemEval official non-CoT answer prompt: src/generation/run_generation.py.
_LME = ("I will give you several history chats between you and a user. Please answer the question based on the "
        "relevant chat history.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:")

NO_CONTEXT = "(nothing was retrieved)"


def reader_prompt(q: Question, context: str) -> str:
    ctx = context.strip() or NO_CONTEXT
    if q.bench == "locomo":
        head = ("Below is retrieved context from a long conversation between two people. The date of each item "
                "is written with it.\n\n" + ctx + "\n")
        return head + (_LOCOMO_QA5 if q.extra.get("cat") == 5 else _LOCOMO_QA).format(q.question)
    if q.bench == "dmr":
        return ("Below is retrieved context from earlier conversations between two people. The question is "
                "asked by one of them.\n\n" + ctx + "\n\nBased on the above context, answer the question "
                "briefly.\n\nQuestion: " + q.question + " Answer:")
    if q.bench == "longmemeval":
        return _LME.format(ctx, q.extra.get("question_date", ""), q.question)
    if q.bench == "beam":
        return _BEAM_RAG.replace("<context>", ctx).replace("<question>", q.question)
    raise ValueError(q.bench)


JUDGE_MODELS: Dict[str, str] = {
    "locomo": "openai/gpt-4o-mini",
    "dmr": "openai/gpt-4o",
    "longmemeval": "openai/gpt-4o-2024-08-06",
    "beam": "openai/gpt-4.1-mini",
}

PRIMARY: Dict[str, str] = {
    "locomo": "LLM-judge accuracy (CORRECT/WRONG, mem0 J prompt); secondary: official token F1",
    "dmr": "LLM-judge accuracy (CORRECT/WRONG, MemGPT prompt); secondary: ROUGE-L recall",
    "longmemeval": "LLM-judge accuracy (official per-type yes/no prompts)",
    "beam": "mean rubric score (0 / 0.5 / 1 per rubric item, official unified judge prompt); secondary: share of questions with score >= 0.5",
}

#: What was verified in the downloaded official sources (paths relative to the bench source dir), and what was not.
DIFFERENCES: Dict[str, str] = {
    "locomo": (
        "- Reader: published LoCoMo runs use their own readers (task_eval/gpt_utils.py names gpt-3.5-turbo / gpt-4 "
        "style models, temperature 0, 32 new tokens, whole-conversation or RAG context; evaluate_qa.py --top-k "
        "default 5). Ours: deepseek/deepseek-v4.1-flash, temperature 0, 400 new tokens, 8 retrieved items.\n"
        "- Prompt: the official QA_PROMPT / QA_PROMPT_CAT_5 text, behind a one-line context header.\n"
        "- Judge: the LoCoMo repo has no LLM judge (token F1 only). The CORRECT/WRONG judge is the Mem0 paper's "
        "(appendix A; its LLM is stated as gpt-4o-mini, mem0.txt lines 325/344) and its prompt is used verbatim. Ours "
        "runs that prompt on openai/gpt-4o-mini through OpenRouter; the exact snapshot Mem0 used is not stated in "
        "what was downloaded (not verified).\n"
        "- Categories: Mem0 excluded adversarial (category 5, mem0.txt lines 256-265) and so do we; categories 1-4 "
        "are scored (multi-hop, temporal, open-domain, single-hop, in the numbering of locomo10.json).\n"
        "- Token F1 is the official formula (evaluation.py: normalise, split sub-answers for category 1, first ';' part "
        "for category 3) but WITHOUT the Porter stemmer (nltk is not installed): scores are slightly lower than the "
        "official script's would be.\n"
        "- Ingest is per turn through the production pipeline with the speaker name prefixed; LoCoMo image captions "
        "(blip_caption) are appended to the turn as text, as the official RAG setup also uses captions."),
    "dmr": (
        "- Reader: the MemGPT paper reports GPT-3.5 Turbo / GPT-4 style readers with MemGPT paging or a fixed "
        "context; ours is deepseek/deepseek-v4.1-flash on retrieved context (temperature 0).\n"
        "- Judge: the paper's prompt (appendix, memgpt_raw.txt lines ~1190-1225) with its worked CORRECT/WRONG "
        "examples; the paper says GPT-4 judged. We run it on openai/gpt-4o (closest available slug; exact GPT-4 "
        "snapshot not verified). The paper's prompt also lists personas and the old chat as inputs; the template "
        "text itself only shows question / gold / generated, which is what we pass.\n"
        "- ROUGE-L: LCS-based recall on lower-cased alphanumeric tokens (the paper reports ROUGE-L recall); not the "
        "rouge package (not installed), so values can differ slightly.\n"
        "- Data: msc_self_instruct.jsonl rows (500), question = self_instruct.B, gold = self_instruct.A, history = the "
        "four previous_dialogs plus the row's own dialog; DMR carries only relative times, so sessions are dated from "
        "time_back relative to a fixed anchor date (2024-01-01)."),
    "longmemeval": (
        "- Reader: the official generation script defaults to top-k 50 with the json history format; ours is "
        "deepseek/deepseek-v4.1-flash with 8 retrieved items; prompt = the official non-CoT template.\n"
        "- Judge: official evaluate_qa.py per-type prompts (copied verbatim), gpt-4o-2024-08-06, temperature 0, 10 "
        "tokens; ours uses the same slug through OpenRouter.\n"
        "- Haystack: only the file passed with --lme-file (oracle = evidence sessions only unless S/M is given); the "
        "oracle haystack is far easier than S/M and numbers must not be compared with published S numbers."),
    "beam": (
        "- Reader: official RAG answer prompt (prompts.py answer_generation_for_rag); the official baseline uses "
        "dense retrieval k=5 (answer_generation.py retrieval_config_default); ours uses 8 items.\n"
        "- Judge: official unified_llm_judge_base_prompt per rubric item, score 0 / 0.5 / 1, mean over items; the "
        "official judge model in src/llm.py is gpt-4.1-mini, ours is openai/gpt-4.1-mini.\n"
        "- NOT reproduced: the official event_ordering metric (Kendall-tau x F1 with embedding / LLM alignment) and "
        "the fact-level metrics in compute_metrics.py; every ability, including event_ordering, is scored with the "
        "rubric judge here, so event_ordering numbers are not comparable with published ones.\n"
        "- Only the 100K tier is loaded."),
}
