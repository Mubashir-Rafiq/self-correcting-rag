"""Tests for self_rag.storage.reranker — Cross-encoder reranking."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document

from self_rag.storage.reranker import FastEmbedReranker, PassthroughReranker


class TestRerankers:
    def test_passthrough_reranker(self) -> None:
        docs = [
            Document(page_content="doc1"),
            Document(page_content="doc2"),
            Document(page_content="doc3"),
        ]
        reranker = PassthroughReranker()
        assert reranker.rerank("query", docs) == docs
        assert reranker.rerank("query", docs, top_n=2) == docs[:2]

    def test_fastembed_reranker_empty(self) -> None:
        reranker = FastEmbedReranker()
        assert reranker.rerank("test query", []) == []

    def test_fastembed_reranker_blank_query(self) -> None:
        docs = [Document(page_content="text1"), Document(page_content="text2")]
        reranker = FastEmbedReranker()
        res = reranker.rerank("   ", docs, top_n=1)
        assert len(res) == 1
        assert res[0].page_content == "text1"

    def test_fastembed_reranker_scoring_and_ordering(self) -> None:
        docs = [
            Document(page_content="Low relevance content", metadata={"id": 1}),
            Document(page_content="High relevance content", metadata={"id": 2}),
            Document(page_content="Medium relevance content", metadata={"id": 3}),
        ]
        reranker = FastEmbedReranker()
        mock_encoder = MagicMock()
        # Mock scores: doc1 -> 0.1, doc2 -> 0.9, doc3 -> 0.5
        mock_encoder.rerank.return_value = [0.1, 0.9, 0.5]
        reranker._encoder = mock_encoder

        reranked = reranker.rerank("What is high relevance?", docs, top_n=2)
        assert len(reranked) == 2
        # Doc 2 should be first
        assert reranked[0].metadata["id"] == 2
        assert reranked[0].metadata["rerank_score"] == 0.9
        # Doc 3 should be second
        assert reranked[1].metadata["id"] == 3
        assert reranked[1].metadata["rerank_score"] == 0.5

    def test_fastembed_reranker_exception_fallback(self) -> None:
        docs = [Document(page_content="text1"), Document(page_content="text2")]
        reranker = FastEmbedReranker()
        mock_encoder = MagicMock()
        mock_encoder.rerank.side_effect = RuntimeError("ONNX inference failed")
        reranker._encoder = mock_encoder

        # Should log a warning and fall back to original candidates without raising
        result = reranker.rerank("query", docs, top_n=1)
        assert len(result) == 1
        assert result[0].page_content == "text1"


class TestRerankerIntegration:
    def test_tool_factory_with_reranker(self) -> None:
        mock_collection = MagicMock()
        docs = [
            Document(page_content="c1", metadata={"parent_id": "p1"}),
            Document(page_content="c2", metadata={"parent_id": "p2"}),
        ]
        mock_collection.similarity_search.return_value = docs

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [docs[1]]

        from self_rag.agent.tools import ToolFactory

        mock_parent_store = MagicMock()

        factory = ToolFactory(
            collection=mock_collection,
            parent_store=mock_parent_store,
            default_limit=1,
            reranker=mock_reranker,
        )
        tool = factory.create_search_tool()
        output = tool.invoke({"query": "find stuff"})

        mock_reranker.rerank.assert_called_once_with(query="find stuff", documents=docs, top_n=1)
        assert "p2" in output

    def test_rag_system_respects_reranker_injection(self) -> None:
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage

        from self_rag.config import Settings
        from self_rag.system import RAGSystem

        mock_reranker = MagicMock()
        fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content="answer")]))
        system = RAGSystem(
            settings=Settings(self_correction_enabled=False),
            llm=fake_llm,
            reranker=mock_reranker,
            tools=[],
        )
        assert system.reranker is mock_reranker
