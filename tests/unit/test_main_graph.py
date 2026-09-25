"""Unit tests for Stage 9: main graph, memory, rewriting, clarification, and checkpointer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Send

from self_rag.agent.edges import route_after_rewrite
from self_rag.agent.graph import build_main_graph, create_sqlite_checkpointer
from self_rag.agent.nodes import (
    aggregate_answers,
    request_clarification,
    rewrite_query,
    summarize_history,
)
from self_rag.agent.schemas import QueryAnalysis
from self_rag.agent.state import State


class TestSummarizeHistoryNode:
    """Tests for summarize_history node."""

    def test_skips_summarization_when_under_keep_threshold(self) -> None:
        mock_llm = MagicMock()
        messages: list[AnyMessage] = [
            HumanMessage(content="Hello", id="1"),
            AIMessage(content="Hi there!", id="2"),
        ]
        state: State = {"messages": messages, "conversation_summary": ""}

        # keep_count = 4 -> keeps up to 3 plain messages verbatim
        update = summarize_history(state, llm=mock_llm, keep_count=4)

        assert not mock_llm.invoke.called
        assert update["agent_answers"] == [{"__reset__": True}]
        assert "conversation_summary" not in update
        assert "messages" not in update

    def test_summarizes_older_messages_and_emits_removals(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="User asked about system architecture.")

        messages: list[AnyMessage] = [
            HumanMessage(content="Old query 1", id="m1"),
            AIMessage(content="Old answer 1", id="m2"),
            HumanMessage(content="Recent query 2", id="m3"),
            AIMessage(content="Recent answer 2", id="m4"),
        ]
        state: State = {
            "messages": messages,
            "conversation_summary": "Prior summary.",
        }

        # keep_count = 3 -> keep_verbatim = 2 (m3 and m4). Older (m1, m2) summarized and removed.
        update = summarize_history(state, llm=mock_llm, keep_count=3)

        assert mock_llm.invoke.called
        prompt_args = mock_llm.invoke.call_args[0][0]
        assert isinstance(prompt_args[0], SystemMessage)
        assert isinstance(prompt_args[1], HumanMessage)
        prompt_text = str(prompt_args[1].content)
        assert "Existing Summary:\nPrior summary." in prompt_text
        assert "Old query 1" in prompt_text
        assert "Old answer 1" in prompt_text

        assert update["conversation_summary"] == "User asked about system architecture."
        assert update["agent_answers"] == [{"__reset__": True}]
        removals = update["messages"]
        assert len(removals) == 2
        assert all(isinstance(r, RemoveMessage) for r in removals)
        assert {r.id for r in removals} == {"m1", "m2"}


class TestRewriteQueryNode:
    """Tests for rewrite_query node."""

    def test_clear_query_sets_rewritten_questions_and_clears_pending(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = QueryAnalysis(
            is_clear=True,
            questions=["What is hybrid search in Qdrant?", "How does reciprocal rank fusion work?"],
            clarification_needed="",
        )

        messages: list[AnyMessage] = [HumanMessage(content="Tell me about hybrid search and RRF.")]
        state: State = {
            "messages": messages,
            "conversation_summary": "",
            "pendingQuery": "",
        }

        update = rewrite_query(state, llm=mock_llm, max_subquestions=2)

        assert update["questionIsClear"] is True
        assert len(update["rewrittenQuestions"]) == 2
        assert update["rewrittenQuestions"][0] == "What is hybrid search in Qdrant?"
        assert update["pendingQuery"] == ""
        assert update["pendingClarifications"] == []

    def test_clear_query_respects_max_subquestions_limit(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = QueryAnalysis(
            is_clear=True,
            questions=["Q1", "Q2", "Q3", "Q4"],
            clarification_needed="",
        )

        state: State = {
            "messages": [HumanMessage(content="Lots of questions")],
            "conversation_summary": "",
        }

        update = rewrite_query(state, llm=mock_llm, max_subquestions=2)
        assert len(update["rewrittenQuestions"]) == 2
        assert update["rewrittenQuestions"] == ["Q1", "Q2"]

    def test_unclear_query_requests_clarification(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = QueryAnalysis(
            is_clear=False,
            questions=[],
            clarification_needed="Could you clarify what 'it' refers to?",
        )

        messages: list[AnyMessage] = [HumanMessage(content="How does it work?")]
        state: State = {
            "messages": messages,
            "conversation_summary": "",
        }

        update = rewrite_query(state, llm=mock_llm, max_subquestions=2)

        assert update["questionIsClear"] is False
        assert update["pendingQuery"] == "How does it work?"
        assert len(update["messages"]) == 1
        clarification_msg = update["messages"][0]
        assert isinstance(clarification_msg, AIMessage)
        assert clarification_msg.name == "clarification"
        assert "Could you clarify" in str(clarification_msg.content)

    def test_clarification_round_trip_resolves_query(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = QueryAnalysis(
            is_clear=True,
            questions=["How does FastEmbed ONNX runtime work?"],
            clarification_needed="",
        )

        messages: list[AnyMessage] = [
            HumanMessage(content="How does it work?"),
            AIMessage(content="Could you clarify what 'it' refers to?", name="clarification"),
            HumanMessage(content="FastEmbed ONNX runtime"),
        ]
        state: State = {
            "messages": messages,
            "conversation_summary": "",
            "pendingQuery": "How does it work?",
            "pendingClarifications": [],
        }

        update = rewrite_query(state, llm=mock_llm, max_subquestions=2)

        assert update["questionIsClear"] is True
        assert update["rewrittenQuestions"] == ["How does FastEmbed ONNX runtime work?"]
        assert update["pendingQuery"] == ""
        assert update["pendingClarifications"] == []

        # Verify LLM was passed unresolved query + clarification
        invoked_args = mock_structured.invoke.call_args[0][0]
        human_text = str(invoked_args[1].content)
        assert "Unresolved Query:\nHow does it work?" in human_text
        assert "FastEmbed ONNX runtime" in human_text


class TestRouteAfterRewrite:
    """Tests for route_after_rewrite edge."""

    def test_routes_to_request_clarification_when_unclear(self) -> None:
        state: State = {"messages": [], "questionIsClear": False}
        dest = route_after_rewrite(state)
        assert dest == "request_clarification"

    def test_fans_out_with_send_when_clear(self) -> None:
        state: State = {
            "messages": [],
            "questionIsClear": True,
            "rewrittenQuestions": ["Sub-question 1", "Sub-question 2"],
        }
        dest = route_after_rewrite(state)
        assert isinstance(dest, list)
        assert len(dest) == 2
        assert all(isinstance(s, Send) for s in dest)
        assert dest[0].node == "agent"
        assert dest[0].arg["question"] == "Sub-question 1"
        assert dest[0].arg["question_index"] == 0
        assert dest[1].node == "agent"
        assert dest[1].arg["question"] == "Sub-question 2"
        assert dest[1].arg["question_index"] == 1


class TestRequestClarificationNode:
    """Tests for request_clarification no-op node."""

    def test_returns_empty_dict(self) -> None:
        state: State = {"messages": []}
        assert request_clarification(state) == {}


class TestAggregateAnswersNode:
    """Tests for aggregate_answers node."""

    def test_fallback_when_no_agent_answers(self) -> None:
        mock_llm = MagicMock()
        state: State = {"agent_answers": [], "messages": []}
        update = aggregate_answers(state, llm=mock_llm)

        assert not mock_llm.invoke.called
        assert len(update["messages"]) == 1
        assert "couldn't find any information" in str(update["messages"][0].content)

    def test_synthesizes_multiple_answers_sorted_by_index(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Final synthesized aggregate answer.")

        state: State = {
            "originalQuery": "Explain A and B.",
            "agent_answers": [
                {"index": 1, "answer": "Answer B."},
                {"index": 0, "answer": "Answer A."},
            ],
            "messages": [HumanMessage(content="Explain A and B.")],
        }

        update = aggregate_answers(state, llm=mock_llm)

        assert mock_llm.invoke.called
        invoked_args = mock_llm.invoke.call_args[0][0]
        prompt_content = str(invoked_args[1].content)
        assert "Question: Explain A and B." in prompt_content
        # Must be sorted by index 0 then index 1
        pos_a = prompt_content.find("Retrieved response 1:\nAnswer A.")
        pos_b = prompt_content.find("Retrieved response 2:\nAnswer B.")
        assert pos_a != -1 and pos_b != -1
        assert pos_a < pos_b

        assert len(update["messages"]) == 1
        assert update["messages"][0].content == "Final synthesized aggregate answer."


class TestMainGraphIntegration:
    """Integration tests for build_main_graph, SqliteSaver, and interrupts."""

    def test_clear_query_end_to_end_with_sqlite_checkpointer(self, tmp_path: Path) -> None:
        @tool
        def search_child_chunks(query: str) -> str:
            """Search tool."""
            payload = [{"parent_id": "p0", "source": "doc.md", "content": "FastEmbed is fast."}]
            return json.dumps(payload)

        # Mock LLM sequence:
        # 1. rewrite_query structured output -> QueryAnalysis(clear, ["What is FastEmbed?"])
        # 2. agent subgraph orchestrator -> requests tool call
        # 3. agent subgraph orchestrator -> synthesizes answer
        # 4. aggregate_answers -> synthesizes final response
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = QueryAnalysis(
            is_clear=True,
            questions=["What is FastEmbed?"],
            clarification_needed="",
        )

        agent_orchestrator_call = AIMessage(
            content="",
            tool_calls=[
                {"name": "search_child_chunks", "args": {"query": "FastEmbed"}, "id": "c1"}
            ],
        )
        agent_subgraph_answer = AIMessage(
            content="FastEmbed uses ONNX runtime.\n\nSources:\n- doc.md"
        )
        final_aggregated_answer = AIMessage(
            content=(
                "FastEmbed is a lightweight Python library for embeddings.\n\nSources:\n- doc.md"
            )
        )

        mock_llm.invoke.side_effect = [
            agent_orchestrator_call,
            agent_subgraph_answer,
            final_aggregated_answer,
        ]

        db_path = tmp_path / "checkpoints.db"
        checkpointer = create_sqlite_checkpointer(db_path)

        graph = build_main_graph(
            llm=mock_llm,
            tools=[search_child_chunks],
            checkpointer=checkpointer,
            max_subquestions=2,
        )

        config: RunnableConfig = {"configurable": {"thread_id": "session_1"}}
        input_state: State = {"messages": [HumanMessage(content="What is FastEmbed?")]}

        result = graph.invoke(input_state, config=config)

        assert len(result["messages"]) >= 2
        last_msg = result["messages"][-1]
        assert isinstance(last_msg, AIMessage)
        assert "FastEmbed is a lightweight Python library" in str(last_msg.content)
        assert len(result["agent_answers"]) == 1

        # Verify checkpoint persisted in sqlite database
        assert db_path.exists()
        reloaded_checkpointer = create_sqlite_checkpointer(db_path)
        reloaded_graph = build_main_graph(
            llm=mock_llm,
            tools=[search_child_chunks],
            checkpointer=reloaded_checkpointer,
        )
        saved_state = reloaded_graph.get_state(config)
        assert saved_state.values["messages"][-1].content == last_msg.content

    def test_clarification_interrupt_and_resume_flow(self, tmp_path: Path) -> None:
        @tool
        def search_child_chunks(query: str) -> str:
            """Search tool."""
            payload = [
                {"parent_id": "p0", "source": "guide.txt", "content": "RAG architecture guide."}
            ]
            return json.dumps(payload)

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        # 1st call to rewrite_query: query is unclear -> requests clarification
        # 2nd call to rewrite_query: query is now clear -> returns rewritten question
        mock_structured.invoke.side_effect = [
            QueryAnalysis(
                is_clear=False,
                questions=[],
                clarification_needed="Which guide do you mean?",
            ),
            QueryAnalysis(
                is_clear=True,
                questions=["Where is the RAG architecture guide?"],
                clarification_needed="",
            ),
        ]

        agent_call = AIMessage(
            content="",
            tool_calls=[
                {"name": "search_child_chunks", "args": {"query": "RAG guide"}, "id": "c1"}
            ],
        )
        agent_answer = AIMessage(content="Guide is in guide.txt.")
        final_answer = AIMessage(content="The RAG architecture guide is located in guide.txt.")

        mock_llm.invoke.side_effect = [agent_call, agent_answer, final_answer]

        checkpointer = create_sqlite_checkpointer(tmp_path / "checkpoints.db")
        graph = build_main_graph(
            llm=mock_llm,
            tools=[search_child_chunks],
            checkpointer=checkpointer,
        )

        config: RunnableConfig = {"configurable": {"thread_id": "clarify_session"}}

        # 1st turn: user asks ambiguous question
        input_state: State = {"messages": [HumanMessage(content="Where is the guide?")]}
        graph.invoke(input_state, config=config)

        # Graph should pause at request_clarification interrupt point
        state_pause = graph.get_state(config)
        assert state_pause.next == ("request_clarification",)
        last_pause_msg = state_pause.values["messages"][-1]
        assert isinstance(last_pause_msg, AIMessage)
        assert last_pause_msg.name == "clarification"
        assert "Which guide do you mean?" in str(last_pause_msg.content)

        # User provides clarification and resumes graph
        graph.update_state(
            config,
            {"messages": [HumanMessage(content="The RAG architecture one.")]},
            as_node="request_clarification",
        )
        graph.invoke(None, config=config)

        # Completed execution
        state_done = graph.get_state(config)
        assert state_done.next == ()
        last_resume_msg = state_done.values["messages"][-1]
        assert isinstance(last_resume_msg, AIMessage)
        assert "The RAG architecture guide is located in guide.txt." in str(last_resume_msg.content)
