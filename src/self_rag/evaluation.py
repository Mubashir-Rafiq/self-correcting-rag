"""Offline RAG evaluation module and RAGAS-compatible dataset scoring.

Provides:
- Structured data models for offline evaluation (`EvalSample`, `EvalResult`, `EvalReport`).
- LLM-based evaluation metrics: Faithfulness (groundedness), Answer Relevance, Context Precision.
- Evaluation engine `RAGEvaluator` runnable directly or via CLI (`self-rag eval`).
- Export to RAGAS / HuggingFace Dataset format (`to_ragas_dataset`).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from self_rag.config import Settings, get_settings
from self_rag.llm.factory import build_llm
from self_rag.system import RAGSystem

logger = logging.getLogger(__name__)


class EvalSample(BaseModel):
    """A single evaluation benchmark record."""

    question: str
    ground_truth: str = ""
    contexts: list[str] = Field(default_factory=list)
    answer: str = ""


class EvalMetricScore(BaseModel):
    """Evaluated metric scores for a single sample."""

    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    explanation: str = ""


class EvalResult(BaseModel):
    """Complete evaluation outcome for an evaluation sample."""

    sample: EvalSample
    metrics: EvalMetricScore


class EvalReport(BaseModel):
    """Aggregated evaluation metrics across an entire evaluation run."""

    results: list[EvalResult]
    mean_faithfulness: float
    mean_answer_relevance: float
    mean_context_precision: float

    def to_ragas_dataset(self) -> dict[str, list[Any]]:
        """Export samples to a dictionary compatible with RAGAS / datasets.Dataset."""
        return {
            "question": [r.sample.question for r in self.results],
            "contexts": [r.sample.contexts for r in self.results],
            "answer": [r.sample.answer for r in self.results],
            "ground_truth": [r.sample.ground_truth for r in self.results],
        }


EVAL_SYSTEM_PROMPT = """You are an impartial RAG evaluation judge.
Score the quality of a Retrieval-Augmented Generation (RAG) system output.

You must output valid JSON only, conforming to this schema:
{
  "faithfulness": <float 0.0-1.0: is every claim in answer supported by contexts?>,
  "answer_relevance": <float 0.0-1.0: does answer directly and completely address the question?>,
  "context_precision": <float 0.0-1.0: do retrieved contexts contain the ground truth facts?>,
  "explanation": "<brief 1-2 sentence rationale>"
}
"""


class RAGEvaluator:
    """Evaluates RAG outputs either standalone or by running against a live RAGSystem."""

    def __init__(
        self,
        judge_llm: BaseChatModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.judge_llm: BaseChatModel | None
        if judge_llm is not None:
            self.judge_llm = judge_llm
        else:
            try:
                if self.settings.judge_model:
                    self.judge_llm = build_llm(self.settings, model=self.settings.judge_model)
                else:
                    self.judge_llm = build_llm(self.settings)
            except Exception as err:
                logger.warning(
                    "Could not initialize judge LLM: %s. Using heuristic evaluation.", err
                )
                self.judge_llm = None

    def evaluate_sample(self, sample: EvalSample) -> EvalResult:
        """Score an individual evaluation sample using the judge LLM."""
        context_str = (
            "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(sample.contexts))
            if sample.contexts
            else "None"
        )
        if self.judge_llm is None:
            f_score = 1.0 if any(word in context_str for word in sample.answer.split()[:5]) else 0.5
            score = EvalMetricScore(
                faithfulness=f_score,
                answer_relevance=0.7 if sample.answer else 0.0,
                context_precision=0.7 if sample.contexts else 0.0,
                explanation="Heuristic evaluation (no LLM judge configured).",
            )
            return EvalResult(sample=sample, metrics=score)
        user_prompt = (
            f"Question: {sample.question}\n\n"
            f"Ground Truth: {sample.ground_truth or 'N/A'}\n\n"
            f"Retrieved Contexts:\n{context_str}\n\n"
            f"Generated Answer:\n{sample.answer}\n\n"
            "Evaluate and return JSON."
        )

        try:
            response = self.judge_llm.invoke(
                [
                    SystemMessage(content=EVAL_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
            raw = response.content
            if not isinstance(raw, str):
                raw = json.dumps(raw)

            # Strip markdown fence if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()

            parsed = json.loads(cleaned)
            score = EvalMetricScore(
                faithfulness=float(parsed.get("faithfulness", 0.0)),
                answer_relevance=float(parsed.get("answer_relevance", 0.0)),
                context_precision=float(parsed.get("context_precision", 0.0)),
                explanation=str(parsed.get("explanation", "")),
            )
        except Exception as err:
            logger.warning("Judge LLM evaluation failed: %s; falling back to heuristic", err)
            # Heuristic fallback: check context overlap
            f_score = 1.0 if any(word in context_str for word in sample.answer.split()[:5]) else 0.5
            score = EvalMetricScore(
                faithfulness=f_score,
                answer_relevance=0.7 if sample.answer else 0.0,
                context_precision=0.7 if sample.contexts else 0.0,
                explanation=f"Heuristic fallback due to: {err}",
            )

        return EvalResult(sample=sample, metrics=score)

    def evaluate_dataset(
        self,
        samples: Sequence[EvalSample],
        rag_system: RAGSystem | None = None,
    ) -> EvalReport:
        """Evaluate a collection of samples, optionally running rag_system to generate answers."""
        results: list[EvalResult] = []

        for sample in samples:
            current_sample = sample
            if rag_system is not None and not sample.answer:
                # Run query through RAGSystem
                thread_id = rag_system.create_thread_id()
                chat_res = rag_system.chat(sample.question, thread_id)
                ans = str(chat_res.get("answer", ""))
                current_sample = EvalSample(
                    question=sample.question,
                    ground_truth=sample.ground_truth,
                    contexts=sample.contexts,
                    answer=ans,
                )

            res = self.evaluate_sample(current_sample)
            results.append(res)

        if not results:
            return EvalReport(
                results=[],
                mean_faithfulness=0.0,
                mean_answer_relevance=0.0,
                mean_context_precision=0.0,
            )

        mean_f = sum(r.metrics.faithfulness for r in results) / len(results)
        mean_r = sum(r.metrics.answer_relevance for r in results) / len(results)
        mean_p = sum(r.metrics.context_precision for r in results) / len(results)

        return EvalReport(
            results=results,
            mean_faithfulness=round(mean_f, 4),
            mean_answer_relevance=round(mean_r, 4),
            mean_context_precision=round(mean_p, 4),
        )

    @classmethod
    def load_dataset_from_file(cls, path: Path) -> list[EvalSample]:
        """Load evaluation samples from a JSON or JSONL file."""
        text = path.read_text(encoding="utf-8").strip()
        samples: list[EvalSample] = []
        if path.suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    data = json.loads(line)
                    samples.append(EvalSample(**data))
        else:
            data = json.loads(text)
            if isinstance(data, list):
                samples.extend(EvalSample(**item) for item in data)
            elif isinstance(data, dict) and "samples" in data:
                samples.extend(EvalSample(**item) for item in data["samples"])
            else:
                samples.append(EvalSample(**data))
        return samples
