"""Unit tests for agent retrieval tools, sentinels, ToolFactory, and context extraction."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.messages import AIMessage, ToolMessage
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings, SparseVector
from qdrant_client import QdrantClient

from self_rag.agent.tools import (
    NO_PARENT_DOCUMENT,
    NO_RELEVANT_CHUNKS,
    PARENT_RETRIEVAL_ERROR_PREFIX,
    RETRIEVAL_ERROR_PREFIX,
    SENTINEL_PREFIXES,
    ToolFactory,
    _retrieval_contexts,
    execute_retrieve_parent_chunks,
    execute_search_child_chunks,
    format_retrieval_contexts,
)
from self_rag.storage.parent_store import ParentStore
from self_rag.storage.vector_store import VectorStoreManager


class FakeSparseEmbeddings(SparseEmbeddings):
    """Stub sparse embeddings for fast, deterministic unit testing."""

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1, 2], values=[0.5, 0.8]) for _ in texts]

    def embed_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[1, 2], values=[0.5, 0.8])


@pytest.fixture()
def in_memory_client() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture()
def fake_vector_store(in_memory_client: QdrantClient) -> QdrantVectorStore:
    vm = VectorStoreManager(
        client=in_memory_client,
        collection_name="test_children",
        dense_dimension=4,
    )
    store = vm.get_vector_store(
        embedding=FakeEmbeddings(size=4),
        sparse_embedding=FakeSparseEmbeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
    )
    return store


@pytest.fixture()
def parent_store(tmp_path: Path) -> ParentStore:
    return ParentStore(directory=tmp_path / "parents")


class TestSearchChildChunksTool:
    """Tests for execute_search_child_chunks and search_child_chunks tool wrapper."""

    def test_search_returns_no_relevant_chunks_for_empty_query(
        self, fake_vector_store: QdrantVectorStore
    ) -> None:
        result = execute_search_child_chunks(fake_vector_store, "   ")
        assert result == NO_RELEVANT_CHUNKS

    def test_search_returns_no_relevant_chunks_when_store_is_empty(
        self, fake_vector_store: QdrantVectorStore
    ) -> None:
        result = execute_search_child_chunks(fake_vector_store, "python programming")
        assert result == NO_RELEVANT_CHUNKS

    def test_search_returns_json_array_of_matching_chunks(
        self, fake_vector_store: QdrantVectorStore
    ) -> None:
        fake_vector_store.add_documents(
            [
                Document(
                    page_content="Child chunk content about agentic workflows.",
                    metadata={"parent_id": "doc1_p0", "source": "workflow.pdf"},
                ),
                Document(
                    page_content="Child chunk content about vector databases.",
                    metadata={"parent_id": "doc2_p1", "source": "vector.pdf"},
                ),
            ]
        )

        raw_result = execute_search_child_chunks(fake_vector_store, "agentic workflows", limit=2)
        assert not raw_result.startswith(RETRIEVAL_ERROR_PREFIX)
        assert raw_result != NO_RELEVANT_CHUNKS

        parsed = json.loads(raw_result)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        first = parsed[0]
        assert "parent_id" in first
        assert "source" in first
        assert "content" in first

    def test_search_handles_exceptions_cleanly(self) -> None:
        mock_store = MagicMock(spec=QdrantVectorStore)
        mock_store.similarity_search.side_effect = RuntimeError("Database offline")

        result = execute_search_child_chunks(mock_store, "query")
        assert result.startswith(RETRIEVAL_ERROR_PREFIX)
        assert "Database offline" in result


class TestRetrieveParentChunksTool:
    """Tests for execute_retrieve_parent_chunks and retrieve_parent_chunks tool wrapper."""

    def test_retrieve_returns_no_parent_document_for_empty_id(
        self, parent_store: ParentStore
    ) -> None:
        result = execute_retrieve_parent_chunks(parent_store, "  ")
        assert result == NO_PARENT_DOCUMENT

    def test_retrieve_returns_no_parent_document_for_missing_chunk(
        self, parent_store: ParentStore
    ) -> None:
        result = execute_retrieve_parent_chunks(parent_store, "nonexistent_p0")
        assert result == NO_PARENT_DOCUMENT

    def test_retrieve_returns_json_object_for_existing_chunk(
        self, parent_store: ParentStore
    ) -> None:
        parent_store.save(
            "doc1_p0",
            Document(
                page_content="# Architecture Overview\nDetailed system explanation.",
                metadata={"source": "doc1.pdf", "parent_id": "doc1_p0"},
            ),
        )

        result = execute_retrieve_parent_chunks(parent_store, "doc1_p0")
        assert not result.startswith(PARENT_RETRIEVAL_ERROR_PREFIX)
        assert result != NO_PARENT_DOCUMENT

        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert parsed["parent_id"] == "doc1_p0"
        assert parsed["source"] == "doc1.pdf"
        assert parsed["content"] == "# Architecture Overview\nDetailed system explanation."

    def test_retrieve_handles_exceptions_cleanly(self) -> None:
        mock_store = MagicMock(spec=ParentStore)
        mock_store.load.side_effect = PermissionError("Access denied")

        result = execute_retrieve_parent_chunks(mock_store, "doc1_p0")
        assert result.startswith(PARENT_RETRIEVAL_ERROR_PREFIX)
        assert "Access denied" in result


class TestToolFactory:
    """Tests for ToolFactory tool generation and LangChain invocation contracts."""

    def test_tool_factory_creates_both_tools(
        self, fake_vector_store: QdrantVectorStore, parent_store: ParentStore
    ) -> None:
        factory = ToolFactory(fake_vector_store, parent_store, default_limit=5)
        tools = factory.create_tools()

        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert tool_names == {"search_child_chunks", "retrieve_parent_chunks"}

    def test_tool_factory_tools_can_be_invoked_via_langchain(
        self, fake_vector_store: QdrantVectorStore, parent_store: ParentStore
    ) -> None:
        parent_store.save(
            "manual_p0",
            Document(
                page_content="Full user manual text.",
                metadata={"source": "manual.pdf", "parent_id": "manual_p0"},
            ),
        )
        fake_vector_store.add_documents(
            [
                Document(
                    page_content="Manual section on installation.",
                    metadata={"parent_id": "manual_p0", "source": "manual.pdf"},
                ),
            ]
        )

        factory = ToolFactory(fake_vector_store, parent_store, default_limit=3)
        search_tool = factory.create_search_tool()
        retrieve_tool = factory.create_retrieve_tool()

        search_output = search_tool.invoke({"query": "installation"})
        assert search_output != NO_RELEVANT_CHUNKS
        search_parsed = json.loads(search_output)
        assert isinstance(search_parsed, list)
        assert search_parsed[0]["parent_id"] == "manual_p0"

        retrieve_output = retrieve_tool.invoke({"parent_id": "manual_p0"})
        retrieve_parsed = json.loads(retrieve_output)
        assert isinstance(retrieve_parsed, dict)
        assert retrieve_parsed["parent_id"] == "manual_p0"
        assert retrieve_parsed["content"] == "Full user manual text."


class TestRetrievalContextExtraction:
    """Tests for format_retrieval_contexts and _retrieval_contexts helper."""

    def test_sentinels_are_filtered_out(self) -> None:
        messages = [
            ToolMessage(content=NO_RELEVANT_CHUNKS, tool_call_id="call_1"),
            ToolMessage(content=NO_PARENT_DOCUMENT, tool_call_id="call_2"),
            ToolMessage(content=f"{RETRIEVAL_ERROR_PREFIX} timeout", tool_call_id="call_3"),
            ToolMessage(content=f"{PARENT_RETRIEVAL_ERROR_PREFIX} IO error", tool_call_id="call_4"),
        ]
        contexts = format_retrieval_contexts(messages)
        assert contexts == []

    def test_formats_search_json_array(self) -> None:
        payload = json.dumps(
            [
                {"parent_id": "doc_p0", "source": "guide.pdf", "content": "Chunk one."},
                {"parent_id": "doc_p1", "source": "guide.pdf", "content": "Chunk two."},
            ]
        )
        messages = [
            ToolMessage(content=payload, tool_call_id="call_1"),
            AIMessage(content="Not a tool message"),
        ]
        contexts = _retrieval_contexts(messages)
        assert len(contexts) == 2
        assert contexts[0] == "Parent ID: doc_p0\nFile Name: guide.pdf\nContent: Chunk one."
        assert contexts[1] == "Parent ID: doc_p1\nFile Name: guide.pdf\nContent: Chunk two."

    def test_formats_retrieve_json_object(self) -> None:
        payload = json.dumps(
            {
                "parent_id": "doc_p3",
                "source": "manual.pdf",
                "content": "Full section text.",
            }
        )
        contexts = _retrieval_contexts([payload])
        assert len(contexts) == 1
        expected = "Parent ID: doc_p3\nFile Name: manual.pdf\nContent: Full section text."
        assert contexts[0] == expected

    def test_raw_string_fallback_on_unparseable_non_sentinel(self) -> None:
        contexts = format_retrieval_contexts(["Arbitrary plain-text snippet from external source."])
        assert contexts == ["Arbitrary plain-text snippet from external source."]

    def test_sentinel_prefixes_tuple_completeness(self) -> None:
        assert NO_RELEVANT_CHUNKS in SENTINEL_PREFIXES
        assert NO_PARENT_DOCUMENT in SENTINEL_PREFIXES
        assert RETRIEVAL_ERROR_PREFIX in SENTINEL_PREFIXES
        assert PARENT_RETRIEVAL_ERROR_PREFIX in SENTINEL_PREFIXES
