"""Unit tests for LangGraph state schemas and reducers."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from self_rag.agent.state import (
    AgentState,
    State,
    accumulate_or_reset,
    append_unique,
    set_union,
)


class TestStateReducers:
    """Tests for channel reducers defined in agent.state."""

    def test_accumulate_or_reset_appends_items(self) -> None:
        existing = [{"answer": "A", "sub_question": "Q1"}]
        new = [{"answer": "B", "sub_question": "Q2"}]
        result = accumulate_or_reset(existing, new)
        assert result == [
            {"answer": "A", "sub_question": "Q1"},
            {"answer": "B", "sub_question": "Q2"},
        ]

    def test_accumulate_or_reset_handles_none_existing(self) -> None:
        new = [{"answer": "A", "sub_question": "Q1"}]
        result = accumulate_or_reset(None, new)
        assert result == [{"answer": "A", "sub_question": "Q1"}]

    def test_accumulate_or_reset_handles_empty_new(self) -> None:
        existing = [{"answer": "A"}]
        assert accumulate_or_reset(existing, []) == [{"answer": "A"}]

    def test_accumulate_or_reset_resets_on_reset_flag(self) -> None:
        existing = [{"answer": "Old A"}, {"answer": "Old B"}]
        new: list[dict[str, Any]] = [{"__reset__": True}]
        result = accumulate_or_reset(existing, new)
        assert result == []

    def test_accumulate_or_reset_resets_when_reset_flag_among_items(self) -> None:
        existing = [{"answer": "Old"}]
        new: list[dict[str, Any]] = [{"answer": "New"}, {"__reset__": True}]
        result = accumulate_or_reset(existing, new)
        assert result == []

    def test_set_union_combines_sets(self) -> None:
        a = {"search::python", "parent::p1"}
        b = {"parent::p1", "search::rag"}
        assert set_union(a, b) == {"search::python", "parent::p1", "search::rag"}

    def test_set_union_handles_none_initial(self) -> None:
        b = {"search::agent"}
        assert set_union(None, b) == {"search::agent"}

    def test_append_unique_preserves_order_and_deduplicates(self) -> None:
        existing = ["ctx1", "ctx2"]
        new = ["ctx2", "ctx3", "ctx1", "ctx4"]
        assert append_unique(existing, new) == ["ctx1", "ctx2", "ctx3", "ctx4"]

    def test_append_unique_handles_none_initial(self) -> None:
        assert append_unique(None, ["ctx1", "ctx1", "ctx2"]) == ["ctx1", "ctx2"]


class TestStateSchemas:
    """Tests validating TypedDict state definitions and StateGraph runtime integration."""

    def test_state_creation_typed_dict(self) -> None:
        state: State = {
            "messages": [HumanMessage(content="Hello")],
            "questionIsClear": True,
            "originalQuery": "Explain RAG",
            "rewrittenQuestions": ["What is RAG?"],
        }
        assert state["questionIsClear"] is True
        assert len(state["messages"]) == 1
        assert state["originalQuery"] == "Explain RAG"

    def test_agent_state_creation_typed_dict(self) -> None:
        agent_state: AgentState = {
            "messages": [HumanMessage(content="Sub-question")],
            "question": "What is RAG?",
            "question_index": 0,
            "tool_call_count": 2,
            "iteration_count": 1,
            "retrieval_keys": {"search::rag"},
        }
        assert agent_state["question"] == "What is RAG?"
        assert agent_state["tool_call_count"] == 2
        assert "search::rag" in agent_state["retrieval_keys"]

    def test_state_graph_runtime_with_agent_state(self) -> None:
        """Verify AgentState channels and reducers work in a compiled LangGraph."""
        builder = StateGraph(AgentState)

        def step_one(s: AgentState) -> dict[str, Any]:
            return {
                "tool_call_count": 1,
                "iteration_count": 1,
                "retrieval_keys": {"search::one"},
                "retrieved_contexts": ["chunk1"],
                "messages": [AIMessage(content="Step 1 done")],
            }

        def step_two(s: AgentState) -> dict[str, Any]:
            return {
                "tool_call_count": 2,
                "iteration_count": 1,
                "retrieval_keys": {"search::one", "parent::p1"},
                "retrieved_contexts": ["chunk1", "chunk2"],
                "messages": [AIMessage(content="Step 2 done")],
            }

        builder.add_node("step_one", step_one)  # type: ignore[call-overload]
        builder.add_node("step_two", step_two)  # type: ignore[call-overload]
        builder.add_edge(START, "step_one")
        builder.add_edge("step_one", "step_two")
        builder.add_edge("step_two", END)

        graph = builder.compile()
        result = graph.invoke({"messages": [HumanMessage(content="Start")]})

        # Reducers check:
        # tool_call_count: 1 + 2 = 3
        # iteration_count: 1 + 1 = 2
        # retrieval_keys: {'search::one'} | {'search::one', 'parent::p1'}
        # retrieved_contexts: ['chunk1'] + ['chunk1', 'chunk2'] deduplicated -> ['chunk1', 'chunk2']
        # messages: Start + Step 1 + Step 2 = 3 messages
        assert result["tool_call_count"] == 3
        assert result["iteration_count"] == 2
        assert result["retrieval_keys"] == {"search::one", "parent::p1"}
        assert result["retrieved_contexts"] == ["chunk1", "chunk2"]
        assert len(result["messages"]) == 3

    def test_state_graph_runtime_with_main_state_reset(self) -> None:
        """Verify State graph channels accumulate answers and reset on __reset__."""
        builder = StateGraph(State)

        def node_add(s: State) -> dict[str, Any]:
            return {"agent_answers": [{"answer": "A1"}]}

        def node_reset(s: State) -> dict[str, Any]:
            return {"agent_answers": [{"__reset__": True}]}

        def node_fresh(s: State) -> dict[str, Any]:
            return {"agent_answers": [{"answer": "Fresh A"}]}

        builder.add_node("add", node_add)  # type: ignore[call-overload]
        builder.add_node("reset", node_reset)  # type: ignore[call-overload]
        builder.add_node("fresh", node_fresh)  # type: ignore[call-overload]

        builder.add_edge(START, "add")
        builder.add_edge("add", "reset")
        builder.add_edge("reset", "fresh")
        builder.add_edge("fresh", END)

        graph = builder.compile()
        result = graph.invoke({"messages": []})
        assert result["agent_answers"] == [{"answer": "Fresh A"}]
