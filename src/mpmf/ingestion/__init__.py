"""
mpmf.ingestion — Self-evolving, question-driven auto-ingestion for MPMF.

See MPMF_AUTO_INGESTION_SPEC.md. Entry point:

    from mpmf.ingestion import AutoIngest
    ai = AutoIngest(store_path="~/.mpmf/athena.mpmf",
                    template_path="~/.mpmf/question_template.json")
    stats = ai.ingest(conversation_text)
    hits = ai.recall("what did we decide about X?")
"""
from .template import QuestionTemplate, Question
from .extract import Extractor, Answer
from .ingest import AutoIngest
from . import decay

__all__ = ["AutoIngest", "QuestionTemplate", "Question", "Extractor", "Answer", "decay"]
