"""Tests for self_rag.evaluation — Offline evaluation and RAGAS dataset scoring."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from self_rag.evaluation import (
    EvalMetricScore,
    EvalReport,
    EvalResult,
    EvalSample,
    RAGEvaluator,
)


class TestEvaluationModels:
    def test_eval_sample_and_report_ragas_export(self) -> None:
        sample = EvalSample(
            question="What is Qdrant?",
            ground_truth="A vector database.",
            contexts=["Qdrant is a high-performance vector search engine."],
            answer="Qdrant is a vector search engine.",
        )
        score = EvalMetricScore(
            faithfulness=0.95,
            answer_relevance=0.9,
            context_precision=1.0,
            explanation="Well supported.",
        )
        result = EvalResult(sample=sample, metrics=score)
        report = EvalReport(
            results=[result],
            mean_faithfulness=0.95,
            mean_answer_relevance=0.9,
            mean_context_precision=1.0,
        )

        ragas_data = report.to_ragas_dataset()
        assert "question" in ragas_data
        assert "contexts" in ragas_data
        assert "answer" in ragas_data
        assert "ground_truth" in ragas_data
        assert ragas_data["question"] == ["What is Qdrant?"]
        assert ragas_data["ground_truth"] == ["A vector database."]

    def test_load_dataset_from_json_and_jsonl(self, tmp_path: Path) -> None:
        json_file = tmp_path / "dataset.json"
        data = [
            {"question": "Q1", "ground_truth": "A1"},
            {"question": "Q2", "ground_truth": "A2"},
        ]
        json_file.write_text(json.dumps(data), encoding="utf-8")
        samples = RAGEvaluator.load_dataset_from_file(json_file)
        assert len(samples) == 2
        assert samples[0].question == "Q1"

        jsonl_file = tmp_path / "dataset.jsonl"
        jsonl_file.write_text(
            '{"question": "QL1", "ground_truth": "AL1"}\n'
            '{"question": "QL2", "ground_truth": "AL2"}\n',
            encoding="utf-8",
        )
        samples_l = RAGEvaluator.load_dataset_from_file(jsonl_file)
        assert len(samples_l) == 2
        assert samples_l[1].question == "QL2"


class TestRAGEvaluator:
    def test_evaluator_with_mock_llm(self) -> None:
        judge_output = json.dumps(
            {
                "faithfulness": 0.9,
                "answer_relevance": 0.85,
                "context_precision": 0.8,
                "explanation": "High factual alignment.",
            }
        )
        fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content=judge_output)]))
        evaluator = RAGEvaluator(judge_llm=fake_llm)

        sample = EvalSample(
            question="What is LangGraph?",
            ground_truth="A graph orchestration framework.",
            contexts=["LangGraph is a library for building stateful multi-actor applications."],
            answer="LangGraph is for building stateful graph applications.",
        )
        res = evaluator.evaluate_sample(sample)
        assert res.metrics.faithfulness == 0.9
        assert res.metrics.answer_relevance == 0.85
        assert res.metrics.context_precision == 0.8
        assert res.metrics.explanation == "High factual alignment."

    def test_evaluator_dataset_averaging(self) -> None:
        resp1 = json.dumps({"faithfulness": 1.0, "answer_relevance": 0.8, "context_precision": 0.6})
        resp2 = json.dumps({"faithfulness": 0.6, "answer_relevance": 0.8, "context_precision": 1.0})
        fake_llm = GenericFakeChatModel(
            messages=iter([AIMessage(content=resp1), AIMessage(content=resp2)])
        )
        evaluator = RAGEvaluator(judge_llm=fake_llm)

        samples = [
            EvalSample(question="Q1", answer="A1"),
            EvalSample(question="Q2", answer="A2"),
        ]
        report = evaluator.evaluate_dataset(samples)
        assert len(report.results) == 2
        assert report.mean_faithfulness == 0.8
        assert report.mean_answer_relevance == 0.8
        assert report.mean_context_precision == 0.8

    def test_evaluator_fallback_on_invalid_output(self) -> None:
        # LLM returns invalid json string
        fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content="Not a json output")]))
        evaluator = RAGEvaluator(judge_llm=fake_llm)

        sample = EvalSample(
            question="What is RAG?",
            contexts=["Retrieval-Augmented Generation combines retrieval with generative models."],
            answer="RAG is Retrieval-Augmented Generation.",
        )
        res = evaluator.evaluate_sample(sample)
        # Should not raise exception, but use heuristic
        assert res.metrics.faithfulness > 0.0
        assert "Heuristic fallback" in res.metrics.explanation
