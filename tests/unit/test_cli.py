"""Tests for the ``self-rag chunk`` CLI subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from self_rag.cli import main


class TestChunkCommand:
    def test_nonexistent_file_returns_error(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        bogus = tmp_path / "nope.md"
        code = main(["chunk", str(bogus)])

        assert code == 1

    def test_chunk_markdown_file(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Chunking a real Markdown file should succeed and print a summary."""
        md = tmp_path / "test.md"
        md.write_text("# Title\n\n" + "word " * 400 + "\n\n## Section\n\n" + "more " * 400)

        # Point data_dir to tmp so we don't pollute the real project.
        monkeypatch.setenv("SELF_RAG_DATA_DIR", str(tmp_path / "data"))

        from self_rag.config import get_settings

        get_settings.cache_clear()

        code = main(["chunk", str(md)])

        assert code == 0


class TestSearchCommand:
    def test_search_empty_collection_prints_error(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SELF_RAG_DATA_DIR", str(tmp_path / "data"))
        from self_rag.config import get_settings

        get_settings.cache_clear()

        code = main(["search", "test query"])
        captured = capsys.readouterr()

        assert code == 0
        assert "collection is empty or does not exist" in captured.out

    def test_search_finds_indexed_content(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        data_dir = tmp_path / "data"
        monkeypatch.setenv("SELF_RAG_DATA_DIR", str(data_dir))
        from langchain_core.documents import Document

        from self_rag.config import get_settings
        from self_rag.storage.embeddings import get_dense_embeddings, get_sparse_embeddings
        from self_rag.storage.vector_store import VectorStoreManager

        get_settings.cache_clear()
        settings = get_settings()

        with VectorStoreManager(settings=settings) as vm:
            dense = get_dense_embeddings(settings)
            sparse = get_sparse_embeddings(settings)
            store = vm.get_vector_store(embedding=dense, sparse_embedding=sparse)

            docs = [
                Document(
                    page_content="Vector databases index embeddings for similarity search.",
                    metadata={"source": "vector.pdf", "parent_id": "v_p0", "H1": "Vector DBs"},
                ),
                Document(
                    page_content="Classical recipes for French onion soup with gruyere.",
                    metadata={"source": "cooking.pdf", "parent_id": "c_p0", "H1": "Soups"},
                ),
            ]
            store.add_documents(docs)

        code = main(["search", "similarity search", "-k", "1"])
        captured = capsys.readouterr()

        assert code == 0
        assert "found 1 results" in captured.out
        assert "vector.pdf" in captured.out
        assert "score=" in captured.out


class TestIngestAndListCommands:
    def test_ingest_and_list_flow(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        data_dir = tmp_path / "data"
        monkeypatch.setenv("SELF_RAG_DATA_DIR", str(data_dir))
        from self_rag.config import get_settings

        get_settings.cache_clear()

        # Initially list should be empty
        code = main(["list"])
        captured = capsys.readouterr()
        assert code == 0
        assert "No documents indexed" in captured.out

        # Ingest a real file
        doc_path = tmp_path / "manual.md"
        doc_path.write_text("# Manual\n\nInstructions here.", encoding="utf-8")

        code = main(["ingest", str(doc_path)])
        captured = capsys.readouterr()
        assert code == 0
        assert "ingested: manual.md" in captured.out

        # Ingest again without force should skip
        code = main(["ingest", str(doc_path)])
        captured = capsys.readouterr()
        assert code == 0
        assert "skipped: manual.md is unchanged" in captured.out

        # List should now show the ingested file
        code = main(["list"])
        captured = capsys.readouterr()
        assert code == 0
        assert "manual.md" in captured.out

    def test_ingest_nonexistent_file(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        code = main(["ingest", str(tmp_path / "ghost.md")])
        captured = capsys.readouterr()
        assert code == 1
        assert "error: file not found" in captured.out


class TestLLMCheckCommand:
    def test_llm_check_missing_key(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SELF_RAG_GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("SELF_RAG_GROQ_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from self_rag.config import get_settings

        get_settings.cache_clear()

        code = main(["llm-check"])
        captured = capsys.readouterr()

        assert code == 1
        assert "No API key configured" in captured.out or "missing" in captured.out

    def test_llm_check_success(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SELF_RAG_GOOGLE_API_KEY", "dummy_val")
        from self_rag.config import get_settings

        get_settings.cache_clear()

        class FakeResponse:
            content = "LLM connection successful"

        class FakeStructured:
            def __init__(self, schema: type) -> None:
                self.schema = schema

            def invoke(self, prompt: str) -> object:
                return self.schema(is_clear=True, questions=["What is RAG?"])

        class FakeModel:
            def invoke(self, prompt: str) -> FakeResponse:
                return FakeResponse()

            def with_structured_output(self, schema: type) -> FakeStructured:
                return FakeStructured(schema)

        monkeypatch.setattr("self_rag.llm.factory.build_llm", lambda s: FakeModel())

        code = main(["llm-check"])
        captured = capsys.readouterr()

        assert code == 0
        assert "status: OK" in captured.out


class TestAskCommand:
    def test_ask_empty_question(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        code = main(["ask", "   "])
        captured = capsys.readouterr()
        assert code == 1
        assert "error: question cannot be empty" in captured.out

    def test_ask_collection_missing_or_empty(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SELF_RAG_DATA_DIR", str(tmp_path / "data"))
        from self_rag.config import get_settings

        get_settings.cache_clear()

        code = main(["ask", "What is RAG?"])
        captured = capsys.readouterr()
        assert code == 1
        assert "error:" in captured.out

    def test_ask_success(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from self_rag.config import get_settings

        get_settings.cache_clear()

        fake_result = {
            "final_answer": (
                "RAG retrieves relevant chunks and augments the prompt.\n\nSources:\n- rag.pdf"
            ),
            "agent_answers": [],
        }
        monkeypatch.setattr(
            "self_rag.agent.graph.ask_question",
            lambda q, settings=None: fake_result,
        )

        code = main(["ask", "What is RAG?"])
        captured = capsys.readouterr()
        assert code == 0
        assert "RAG retrieves relevant chunks" in captured.out
        assert "Sources:" in captured.out


class TestEvalCommand:
    def test_eval_missing_file_returns_error(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        missing = tmp_path / "not_there.json"
        code = main(["eval", str(missing)])
        captured = capsys.readouterr()
        assert code == 1
        assert "error: evaluation dataset not found" in captured.out

    def test_eval_success(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from self_rag.evaluation import EvalMetricScore, EvalReport, EvalResult, EvalSample

        dataset_path = tmp_path / "dataset.json"
        dataset_path.write_text(
            json.dumps([{"question": "What is self-RAG?", "answer": "A self-correcting RAG."}]),
            encoding="utf-8",
        )
        output_path = tmp_path / "report.json"

        mock_sample = EvalSample(
            question="What is self-RAG?",
            answer="A self-correcting RAG.",
        )
        mock_score = EvalMetricScore(
            faithfulness=1.0,
            answer_relevance=1.0,
            context_precision=0.8,
            explanation="Great.",
        )
        mock_report = EvalReport(
            results=[EvalResult(sample=mock_sample, metrics=mock_score)],
            mean_faithfulness=1.0,
            mean_answer_relevance=1.0,
            mean_context_precision=0.8,
        )

        monkeypatch.setattr(
            "self_rag.evaluation.RAGEvaluator.evaluate_dataset",
            lambda self, samples, rag_system=None: mock_report,
        )

        code = main(["eval", str(dataset_path), "-o", str(output_path)])
        captured = capsys.readouterr()

        assert code == 0
        assert "EVALUATION RESULTS SUMMARY" in captured.out
        assert "Mean Faithfulness:       1.0000" in captured.out
        assert output_path.exists()
