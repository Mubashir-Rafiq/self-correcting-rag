"""Tests for page-level citations across chunking, tools, and context formatting."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.documents import Document

from self_rag.agent.tools import (
    execute_retrieve_parent_chunks,
    execute_search_child_chunks,
    format_retrieval_contexts,
)
from self_rag.config import Settings
from self_rag.ingestion.chunker import DocumentChunker


class TestPageIntervalsAndChunking:
    def test_compute_page_intervals_standard_markers(self) -> None:
        text = (
            "Content of page 1\n"
            "--- end of page.page_number=1 ---\n"
            "Content of page 2\n"
            "--- end of page.page_number=2 ---\n"
            "Trailing page 3 content"
        )
        intervals = DocumentChunker._compute_page_intervals(text)
        assert len(intervals) == 3
        assert intervals[0][2] == 1
        assert intervals[1][2] == 2
        assert intervals[2][2] == 3

    def test_compute_page_intervals_empty(self) -> None:
        assert DocumentChunker._compute_page_intervals("No page markers here.") == []

    def test_get_page_for_offset_single_and_range(self) -> None:
        intervals = [(0, 50, 1), (50, 100, 2), (100, 150, 3)]
        # Offset fully within page 1
        assert DocumentChunker._get_page_for_offset(10, 30, intervals) == "1"
        # Offset crossing page 1 and page 2
        assert DocumentChunker._get_page_for_offset(40, 60, intervals) == "1-2"
        # Offset crossing pages 1, 2, and 3
        assert DocumentChunker._get_page_for_offset(10, 120, intervals) == "1-3"
        # Out of bounds
        assert DocumentChunker._get_page_for_offset(200, 250, intervals) is None

    def test_chunk_file_attaches_page_metadata(self, tmp_path: Path) -> None:
        text = (
            "# Document Title\n\n"
            "## Section 1\n"
            "This is page one information.\n\n"
            "--- end of page.page_number=1 ---\n\n"
            "## Section 2\n"
            "This is page two information.\n\n"
            "--- end of page.page_number=2 ---\n"
        )
        md_file = tmp_path / "sample.md"
        md_file.write_text(text, encoding="utf-8")

        settings = Settings(
            child_chunk_size=100,
            child_chunk_overlap=20,
            min_parent_size=50,
            max_parent_size=2000,
        )
        chunker = DocumentChunker(settings)
        parents, children = chunker.chunk_file(md_file, source_name="sample.pdf")

        assert len(parents) > 0
        assert len(children) > 0
        # Parents and children should have page metadata
        pages_present = [p[1].metadata.get("page") for p in parents if "page" in p[1].metadata]
        assert len(pages_present) > 0
        child_pages_present = [c.metadata.get("page") for c in children if "page" in c.metadata]
        assert len(child_pages_present) > 0


class TestCitationFormattingInTools:
    def test_search_child_chunks_includes_page(self) -> None:
        mock_collection = MagicMock()
        mock_doc = Document(
            page_content="Target child chunk",
            metadata={"parent_id": "p1", "source": "guide.pdf", "page": "4"},
        )
        mock_collection.similarity_search.return_value = [mock_doc]

        result_json = execute_search_child_chunks(mock_collection, "query")
        data = json.loads(result_json)
        assert len(data) == 1
        assert data[0]["source"] == "guide.pdf"
        assert data[0]["page"] == "4"

    def test_retrieve_parent_chunks_includes_page(self) -> None:
        mock_store = MagicMock()
        mock_parent = Document(
            page_content="Full parent chunk",
            metadata={"source": "manual.pdf", "page": "12-13"},
        )
        mock_store.load.return_value = mock_parent

        result_json = execute_retrieve_parent_chunks(mock_store, "pid_1")
        data = json.loads(result_json)
        assert data["source"] == "manual.pdf"
        assert data["page"] == "12-13"

    def test_format_retrieval_contexts_page_formatting(self) -> None:
        # With page number
        item_with_page = {
            "parent_id": "p1",
            "source": "paper.pdf",
            "page": "5",
            "content": "Result 1",
        }
        # Without page number
        item_without_page = {
            "parent_id": "p2",
            "source": "notes.md",
            "content": "Result 2",
        }

        contexts = format_retrieval_contexts([item_with_page, item_without_page])
        assert len(contexts) == 2
        assert "File Name: paper.pdf, p. 5" in contexts[0]
        assert "File Name: notes.md\n" in contexts[1]
