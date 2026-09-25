"""Unit tests for Stage 7: minimal agent loop, orchestrator node, routing, and answer collection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings, SparseVector
from qdrant_client import QdrantClient

from self_rag.agent.edges import route_after_orchestrator_call
from self_rag.agent.graph import ask_question, build_agent_subgraph
from self_rag.agent.nodes import collect_answer, orchestrator
from self_rag.agent.state import AgentState
from self_rag.config import Settings
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
    return vm.get_vector_store(
        embedding=FakeEmbeddings(size=4),
        sparse_embedding=FakeSparseEmbeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
    )


@pytest.fixture()
def parent_store(tmp_path: Path) -> ParentStore:
    return ParentStore(directory=tmp_path / "parents")


class TestOrchestratorNode:
    """Tests for the orchestrator graph node."""

    def test_orchestrator_first_call_injects_mandatory_retrieval_prompt(self) -> None:
        mock_llm = MagicMock()
        mock_response = AIMessage(
            content="",
            tool_calls=[{"name": "search_child_chunks", "args": {"query": "RAG"}, "id": "1"}],
        )
        mock_llm.invoke.return_value = mock_response

        state: AgentState = {
            "messages": [],
            "question": "What is self-correcting RAG?",
            "question_index": 0,
        }

        update = orchestrator(state, llm_with_tools=mock_llm)

        assert mock_llm.invoke.called
        invoked_messages = mock_llm.invoke.call_args[0][0]
        # Must include orchestrator system prompt and the initial user message
        expected_phrase = "YOU MUST CALL 'search_child_chunks'"
        assert any(expected_phrase in str(m.content) for m in invoked_messages)

        assert update["tool_call_count"] == 1
        assert update["iteration_count"] == 1
        assert len(update["messages"]) == 2
        assert isinstance(update["messages"][0], HumanMessage)
        assert isinstance(update["messages"][1], AIMessage)
        assert update["messages"][1].name == "agent_response"

    def test_orchestrator_subsequent_call_replays_history(self) -> None:
        mock_llm = MagicMock()
        mock_response = AIMessage(content="Final synthesized answer.")
        mock_llm.invoke.return_value = mock_response

        existing_history: list[AnyMessage] = [
            HumanMessage(content="User question"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search_child_chunks", "args": {"query": "q"}, "id": "c1"}],
            ),
            ToolMessage(content='[{"parent_id": "p0", "content": "text"}]', tool_call_id="c1"),
        ]

        state: AgentState = {
            "messages": existing_history,
            "question": "User question",
            "context_summary": "Prior compressed knowledge.",
        }

        update = orchestrator(state, llm_with_tools=mock_llm)

        assert mock_llm.invoke.called
        invoked_messages = mock_llm.invoke.call_args[0][0]
        # Should include compressed context
        assert any("Prior compressed knowledge." in str(m.content) for m in invoked_messages)

        assert update["tool_call_count"] == 0
        assert update["iteration_count"] == 1
        assert len(update["messages"]) == 1
        assert update["messages"][0].content == "Final synthesized answer."


class TestRoutingEdges:
    """Tests for route_after_orchestrator_call."""

    def test_routes_to_tools_when_tool_calls_requested_and_under_budget(self) -> None:
        state: AgentState = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_child_chunks", "args": {}, "id": "1"}],
                )
            ],
            "iteration_count": 1,
            "tool_call_count": 1,
        }
        dest = route_after_orchestrator_call(state, max_iterations=4, max_tool_calls=4)
        assert dest == "tools"

    def test_routes_to_collect_answer_when_no_tool_calls(self) -> None:
        state: AgentState = {
            "messages": [AIMessage(content="I found the answer: 42.")],
            "iteration_count": 1,
            "tool_call_count": 1,
        }
        dest = route_after_orchestrator_call(state, max_iterations=4, max_tool_calls=4)
        assert dest == "collect_answer"

    def test_routes_to_fallback_response_when_budget_exceeded(self) -> None:
        state: AgentState = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_child_chunks", "args": {}, "id": "1"}],
                )
            ],
            "iteration_count": 4,  # Reached max_iterations
            "tool_call_count": 2,
        }
        dest = route_after_orchestrator_call(state, max_iterations=4, max_tool_calls=4)
        assert dest == "fallback_response"


class TestCollectAnswerNode:
    """Tests for collect_answer node."""

    def test_collect_answer_extracts_last_ai_content(self) -> None:
        payload = '[{"parent_id": "p0", "source": "doc.pdf", "content": "Context text"}]'
        state: AgentState = {
            "messages": [
                HumanMessage(content="Question?"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_child_chunks", "args": {}, "id": "1"}],
                ),
                ToolMessage(content=payload, tool_call_id="1"),
                AIMessage(content="Direct grounded answer.\n\nSources:\n- doc.pdf"),
            ],
            "question": "What is the answer?",
            "question_index": 0,
        }

        result = collect_answer(state)
        assert result["final_answer"] == "Direct grounded answer.\n\nSources:\n- doc.pdf"
        assert len(result["agent_answers"]) == 1
        entry = result["agent_answers"][0]
        assert entry["index"] == 0
        assert entry["question"] == "What is the answer?"
        assert entry["answer"] == result["final_answer"]
        assert len(entry["contexts"]) == 1
        assert "doc.pdf" in entry["contexts"][0]

    def test_collect_answer_substitutes_fallback_if_no_content(self) -> None:
        state: AgentState = {
            "messages": [HumanMessage(content="Hello")],
            "question": "Hello",
        }
        result = collect_answer(state)
        assert result["final_answer"] == "Unable to generate an answer."


class TestAgentLoopIntegration:
    """End-to-end integration tests for the minimal agent loop graph."""

    def test_agent_loop_runs_tool_and_answers(self) -> None:
        @tool
        def search_child_chunks(query: str, limit: int = 7) -> str:
            """Search for relevant chunks."""
            return (
                '[{"parent_id": "doc1_p0", "source": "manual.pdf", '
                '"content": "FastEmbed uses ONNX."}]'
            )

        fake_llm = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_child_chunks",
                                "args": {"query": "FastEmbed ONNX"},
                                "id": "call_1",
                            }
                        ],
                    ),
                    AIMessage(
                        content=(
                            "FastEmbed uses the ONNX runtime for CPU embeddings.\n\n"
                            "Sources:\n- manual.pdf"
                        )
                    ),
                ]
            )
        )

        graph = build_agent_subgraph(llm=fake_llm, tools=[search_child_chunks])
        input_state: AgentState = {
            "question": "What runtime does FastEmbed use?",
            "messages": [],
        }
        result = graph.invoke(input_state)

        assert (
            result["final_answer"]
            == "FastEmbed uses the ONNX runtime for CPU embeddings.\n\nSources:\n- manual.pdf"
        )
        assert result["tool_call_count"] == 1
        assert result["iteration_count"] == 2
        assert len(result["agent_answers"]) == 1
        assert result["agent_answers"][0]["answer"] == result["final_answer"]
        assert len(result["retrieved_contexts"]) == 1
        assert "manual.pdf" in result["retrieved_contexts"][0]


class TestAskQuestionHelper:
    """Tests for ask_question function and error conditions."""

    def test_ask_question_raises_if_collection_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = Settings(
            data_dir=tmp_path / "data",
            child_collection="missing_collection",
        )
        with pytest.raises(ValueError, match="does not exist"):
            ask_question("test query", settings=settings)

    def test_ask_question_raises_if_collection_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = Settings(
            data_dir=tmp_path / "data",
            child_collection="empty_collection",
        )
        vm = VectorStoreManager(settings=settings)
        vm.ensure_collection()
        vm.close()

        with pytest.raises(ValueError, match="is empty"):
            ask_question("test query", settings=settings)

    def test_ask_question_executes_successfully(
        self,
        tmp_path: Path,
        fake_vector_store: QdrantVectorStore,
        parent_store: ParentStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = Settings(
            data_dir=tmp_path / "data",
            child_collection="test_children",
            llm_provider="google_genai",
            google_api_key="test-api-key",
        )

        # Seed child document into collection
        fake_vector_store.add_documents(
            [
                Document(
                    page_content="Architecture of self-correcting RAG.",
                    metadata={"parent_id": "arch_p0", "source": "arch.pdf"},
                )
            ]
        )

        expected_ans = "Self-correcting RAG uses agentic reflection.\n\nSources:\n- arch.pdf"
        fake_llm = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_child_chunks",
                                "args": {"query": "architecture"},
                                "id": "c1",
                            }
                        ],
                    ),
                    AIMessage(content=expected_ans),
                ]
            )
        )

        with (
            patch("self_rag.agent.graph.VectorStoreManager") as mock_vsm_cls,
            patch("self_rag.agent.graph.build_llm", return_value=fake_llm),
        ):
            mock_vsm = MagicMock()
            mock_vsm.collection_exists.return_value = True
            mock_vsm.count.return_value = 1
            mock_vsm.get_vector_store.return_value = fake_vector_store
            mock_vsm_cls.return_value = mock_vsm

            output = ask_question("What is the architecture?", settings=settings)
            assert output["final_answer"] == expected_ans
