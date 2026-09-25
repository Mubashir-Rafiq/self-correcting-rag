"""Unit tests for Stage 12: Gradio UI, session isolation, streaming, and document management."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import gradio as gr
import pytest

from self_rag.config import Settings
from self_rag.ingestion.pipeline import IngestionResult
from self_rag.ui.app import (
    create_app,
    format_sources_markdown,
    format_system_status_markdown,
)
from self_rag.ui.css import CUSTOM_CSS


class TestUICustomCSS:
    """Tests verifying custom CSS theme tokens and layout styling."""

    def test_custom_css_contains_theme_tokens(self) -> None:
        assert "--primary-gradient" in CUSTOM_CSS
        assert "--card-bg" in CUSTOM_CSS
        assert "Plus Jakarta Sans" in CUSTOM_CSS
        assert ".chatbot-container" in CUSTOM_CSS
        assert ".thread-badge" in CUSTOM_CSS


class TestUIFormatters:
    """Tests for UI markdown formatters."""

    def test_format_sources_empty(self) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline.list_documents.return_value = []

        res = format_sources_markdown(mock_pipeline)
        assert "No documents indexed yet" in res

    def test_format_sources_with_data(self) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline.list_documents.return_value = [
            {"source": "guide.pdf", "parents": 4},
            {"source": "readme.md", "parents": 2},
        ]

        res = format_sources_markdown(mock_pipeline)
        assert "| Source Document |" in res
        assert "`guide.pdf`" in res
        assert "**4**" in res
        assert "`readme.md`" in res
        assert "**2**" in res

    def test_format_sources_error_handling(self) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline.list_documents.side_effect = RuntimeError("DB error")

        res = format_sources_markdown(mock_pipeline)
        assert "Error reading indexed sources" in res

    def test_format_system_status(self) -> None:
        mock_system = MagicMock()
        mock_system.settings = Settings(
            llm_provider="google_genai",
            llm_model="gemini-3.6-flash",
            self_correction_enabled=True,
            max_correction_retries=2,
            langfuse_enabled=False,
        )
        mock_system.vector_store_manager.collection_exists.return_value = True
        mock_system.vector_store_manager.count.return_value = 42
        mock_system.parent_store.list_parent_ids.return_value = ["p1", "p2"]

        status = format_system_status_markdown(mock_system)
        assert "`google_genai`" in status
        assert "`gemini-3.6-flash`" in status
        assert "42 vectors" in status
        assert "parent documents" in status
        assert "Enabled (max 2 retries)" in status


class TestCreateApp:
    """Tests for the create_app assembly function."""

    def test_create_app_returns_blocks_instance(self) -> None:
        mock_system = MagicMock()
        mock_pipeline = MagicMock()
        mock_system.settings = Settings()
        mock_system.create_thread_id.return_value = "test-thread-id-1234"
        mock_system.vector_store_manager.collection_exists.return_value = False
        mock_system.parent_store.list_parent_ids.return_value = []
        mock_pipeline.list_documents.return_value = []

        demo = create_app(rag_system=mock_system, pipeline=mock_pipeline)
        assert isinstance(demo, gr.Blocks)


class TestUIDocumentHandlers:
    """Tests for document upload, refresh, and clear operations in UI."""

    def test_handle_add_documents_empty(self) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline.list_documents.return_value = []

        mock_system = MagicMock()
        mock_system.settings = Settings()
        mock_system.create_thread_id.return_value = "thread-1"

        demo = create_app(rag_system=mock_system, pipeline=mock_pipeline)
        # Find the add_btn click function from the blocks graph
        add_fn = None
        for fn in demo.fns.values():
            if fn.name == "handle_add_documents":
                add_fn = fn.fn
                break

        assert add_fn is not None
        msg, _ = add_fn(None)
        assert "No files selected" in msg

    def test_handle_add_documents_success(self, tmp_path: Path) -> None:
        doc1 = tmp_path / "manual.md"
        doc1.write_text("# Manual", encoding="utf-8")

        mock_pipeline = MagicMock()
        mock_pipeline.ingest_file.return_value = IngestionResult(
            source_path=doc1,
            status="added",
            parent_count=3,
            child_count=8,
            message="Ingested",
        )
        mock_pipeline.list_documents.return_value = [{"source": "manual.md", "parents": 3}]

        mock_system = MagicMock()
        mock_system.settings = Settings()
        mock_system.create_thread_id.return_value = "thread-1"

        demo = create_app(rag_system=mock_system, pipeline=mock_pipeline)
        add_fn = None
        for fn in demo.fns.values():
            if fn.name == "handle_add_documents":
                add_fn = fn.fn
                break

        assert add_fn is not None
        mock_file = MagicMock()
        mock_file.name = str(doc1)

        msg, sources = add_fn([mock_file])
        assert "1 added" in msg
        assert "3 parents" in msg
        assert "8 child vectors" in msg
        assert "manual.md" in sources

    def test_handle_clear_all_documents(self) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline.list_documents.return_value = []

        mock_system = MagicMock()
        mock_system.settings = Settings()
        mock_system.create_thread_id.return_value = "thread-1"

        demo = create_app(rag_system=mock_system, pipeline=mock_pipeline)
        clear_fn = None
        for fn in demo.fns.values():
            if fn.name == "handle_clear_all":
                clear_fn = fn.fn
                break

        assert clear_fn is not None
        msg, _ = clear_fn()
        mock_pipeline.clear_all.assert_called_once()
        assert "All document vectors" in msg


class TestUIChatAndSessionIsolation:
    """Tests verifying per-browser-session conversation isolation and streaming."""

    def test_chat_stream_handler_forwards_to_rag_system(self) -> None:
        mock_system = MagicMock()
        mock_system.settings = Settings()
        mock_system.create_thread_id.return_value = "thread-xyz"

        def fake_stream(msg: str, tid: str) -> Generator[str, None, None]:
            yield "Analyzing..."
            yield "Final answer."

        mock_system.chat_stream.side_effect = fake_stream

        from self_rag.ui.app import chat_stream_handler

        chunks = list(chat_stream_handler("What is Qdrant?", [], "thread-123", mock_system))
        assert chunks == ["Analyzing...", "Final answer."]
        mock_system.chat_stream.assert_called_once_with("What is Qdrant?", "thread-123")

    def test_chat_clear_mints_fresh_thread_id(self) -> None:
        mock_system = MagicMock()
        mock_system.settings = Settings()
        mock_system.create_thread_id.side_effect = ["initial-thread", "cleared-thread"]

        mock_pipeline = MagicMock()
        mock_pipeline.list_documents.return_value = []

        demo = create_app(rag_system=mock_system, pipeline=mock_pipeline)
        clear_chat_fn = None
        for fn in demo.fns.values():
            if fn.name == "handle_clear_chat":
                clear_chat_fn = fn.fn
                break

        assert clear_chat_fn is not None
        new_id, badge = clear_chat_fn()
        assert new_id == "initial-thread"
        assert "initial-" in badge


class TestCLIUiCommand:
    """Tests for the CLI ui subcommand."""

    def test_cli_ui_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        from self_rag.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["ui", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "--host" in captured.out
        assert "--port" in captured.out
        assert "--share" in captured.out

    def test_cli_ui_invokes_app_launch(self) -> None:
        from self_rag.cli import main

        mock_app = MagicMock()
        with (
            patch("self_rag.ui.app.create_app", return_value=mock_app) as mock_create,
            patch("builtins.print"),
        ):
            code = main(["ui", "--port", "8080", "--host", "0.0.0.0"])

            assert code == 0
            mock_create.assert_called_once()
            mock_app.launch.assert_called_once_with(
                server_name="0.0.0.0",
                server_port=8080,
                share=False,
                css=CUSTOM_CSS,
            )
