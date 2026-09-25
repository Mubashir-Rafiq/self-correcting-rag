"""Unit tests for Stage 8: research budget, context compression, and fallback response."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.types import Command

from self_rag.agent.graph import build_agent_subgraph
from self_rag.agent.nodes import (
    compress_context,
    fallback_response,
    should_compress_context,
)
from self_rag.agent.state import AgentState


class TestShouldCompressContextNode:
    """Tests for should_compress_context node and threshold gating."""

    def test_routes_to_orchestrator_when_under_threshold(self) -> None:
        state: AgentState = {
            "messages": [
                HumanMessage(content="Hello", id="1"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search_child_chunks", "args": {"query": "rag"}, "id": "tc1"}
                    ],
                    id="2",
                ),
                ToolMessage(
                    content='[{"parent_id": "p1", "source": "s.txt", "content": "short content"}]',
                    tool_call_id="tc1",
                    id="3",
                ),
            ],
            "context_summary": "",
        }

        cmd = should_compress_context(
            state,
            base_token_threshold=1000,
            token_growth_factor=0.9,
        )

        assert isinstance(cmd, Command)
        assert cmd.goto == "orchestrator"
        update = cmd.update
        assert isinstance(update, dict)
        assert "search::rag" in update["retrieval_keys"]
        assert len(update["retrieved_contexts"]) == 1
        assert "short content" in update["retrieved_contexts"][0]

    def test_routes_to_compress_context_when_tokens_exceed_threshold(self) -> None:
        large_content = "large text word " * 300
        state: AgentState = {
            "messages": [
                HumanMessage(content="Question", id="1"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "retrieve_parent_chunks",
                            "args": {"parent_id": "parent_99"},
                            "id": "tc1",
                        }
                    ],
                    id="2",
                ),
                ToolMessage(
                    content=(
                        f'[{{"parent_id": "parent_99", "source": "s.txt", '
                        f'"content": "{large_content}"}}]'
                    ),
                    tool_call_id="tc1",
                    id="3",
                ),
            ],
            "context_summary": "",
        }

        cmd = should_compress_context(
            state,
            base_token_threshold=50,  # Very low threshold to trigger compression
            token_growth_factor=0.9,
        )

        assert isinstance(cmd, Command)
        assert cmd.goto == "compress_context"
        update = cmd.update
        assert isinstance(update, dict)
        assert "parent::parent_99" in update["retrieval_keys"]

    def test_considers_existing_summary_in_threshold_calculation(self) -> None:
        state: AgentState = {
            "messages": [
                HumanMessage(content="Short question", id="1"),
                ToolMessage(content="Short result", tool_call_id="tc1", id="2"),
            ],
            "context_summary": "Extensive prior summary " * 10,  # ~40 tokens
        }

        # Base 100 + int(40 * 0.5 = 20) = 120 allowed.
        # Total tokens = 40 (summary) + ~5 (messages) = ~45 <= 120 -> orchestrator
        cmd_under = should_compress_context(
            state,
            base_token_threshold=100,
            token_growth_factor=0.5,
        )
        assert cmd_under.goto == "orchestrator"

        # Base 10 + int(40 * 0.1 = 4) = 14 allowed.
        # Total tokens = ~45 > 14 -> compress_context
        cmd_over = should_compress_context(
            state,
            base_token_threshold=10,
            token_growth_factor=0.1,
        )
        assert cmd_over.goto == "compress_context"


class TestCompressContextNode:
    """Tests for compress_context node and message history cleanup."""

    def test_compresses_and_emits_remove_messages(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="### Compressed Markdown Summary")

        messages: list[AnyMessage] = [
            HumanMessage(content="Find details on topic", id="m1"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_child_chunks", "args": {"query": "topic"}, "id": "tc1"}
                ],
                id="m2",
            ),
            ToolMessage(
                content="Found chunk data",
                tool_call_id="tc1",
                name="search_child_chunks",
                id="m3",
            ),
        ]

        state: AgentState = {
            "messages": messages,
            "context_summary": "Prior summary content.",
            "retrieval_keys": {"search::topic", "parent::p123"},
        }

        update = compress_context(state, llm=mock_llm)

        assert mock_llm.invoke.called
        invoked_args = mock_llm.invoke.call_args[0][0]
        assert isinstance(invoked_args[0], SystemMessage)
        assert isinstance(invoked_args[1], HumanMessage)
        human_content = str(invoked_args[1].content)
        assert "Prior Compressed Context:" in human_content
        assert "Tool Result (search_child_chunks): Found chunk data" in human_content

        summary = update["context_summary"]
        assert "### Compressed Markdown Summary" in summary
        assert "### Already executed (do NOT repeat):" in summary
        assert "- parent::p123" in summary
        assert "- search::topic" in summary

        # Must emit RemoveMessage for every message with an id
        removals = update["messages"]
        assert len(removals) == 3
        assert all(isinstance(r, RemoveMessage) for r in removals)
        assert {r.id for r in removals} == {"m1", "m2", "m3"}


class TestFallbackResponseNode:
    """Tests for fallback_response node on budget exhaustion."""

    def test_synthesizes_response_from_deduplicated_tool_messages_and_summary(self) -> None:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="Fallback answer based on available evidence."
        )

        raw_context = json.dumps(
            [{"parent_id": "p1", "source": "f.txt", "content": "Evidence piece 1"}]
        )
        messages: list[AnyMessage] = [
            HumanMessage(content="Complex query?", id="1"),
            AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "t1"}], id="2"),
            ToolMessage(content=raw_context, tool_call_id="t1", id="3"),
            # Duplicate tool message should be deduplicated
            ToolMessage(content=raw_context, tool_call_id="t1", id="4"),
            ToolMessage(content="Unique second tool data", tool_call_id="t2", id="5"),
        ]

        state: AgentState = {
            "question": "Complex query?",
            "messages": messages,
            "context_summary": "Summary of preliminary steps.",
        }

        update = fallback_response(state, llm=mock_llm)

        assert mock_llm.invoke.called
        prompt_msgs = mock_llm.invoke.call_args[0][0]
        human_text = str(prompt_msgs[1].content)

        assert "Compressed Research Context:\nSummary of preliminary steps." in human_text
        assert "Evidence piece 1" in human_text
        assert "Unique second tool data" in human_text
        assert human_text.count("Evidence piece 1") == 1  # Deduplicated!
        assert "Question: Complex query?" in human_text

        returned_messages = update["messages"]
        assert len(returned_messages) == 1
        assert returned_messages[0].name == "agent_response"
        assert returned_messages[0].content == "Fallback answer based on available evidence."


class TestSubgraphBudgetAndFallbackIntegration:
    """Integration tests for build_agent_subgraph with budget limits and context compression."""

    def test_subgraph_routes_to_fallback_when_iteration_limit_exceeded(self) -> None:
        @tool
        def dummy_search(query: str) -> str:
            """Dummy search tool."""
            payload = [{"parent_id": "p1", "source": "manual.md", "content": "Relevant info"}]
            return json.dumps(payload)

        orchestrator_call = AIMessage(
            content="",
            tool_calls=[{"name": "dummy_search", "args": {"query": "test"}, "id": "call_1"}],
        )
        fallback_msg = AIMessage(content="Final fallback answer from collected research.")

        # 1st call: orchestrator -> tool call
        # 2nd call: fallback_response -> final synthesis
        fake_llm = GenericFakeChatModel(messages=iter([orchestrator_call, fallback_msg]))

        graph = build_agent_subgraph(
            llm=fake_llm,
            tools=[dummy_search],
            max_iterations=1,  # Set budget to 1 iteration!
            max_tool_calls=4,
        )

        initial_state: AgentState = {
            "question": "What is the procedure?",
            "messages": [],
        }

        output = graph.invoke(initial_state)

        assert output["final_answer"] == "Final fallback answer from collected research."
        assert len(output["agent_answers"]) == 1
        assert (
            output["agent_answers"][0]["answer"] == "Final fallback answer from collected research."
        )

    def test_subgraph_triggers_compression_and_continues_loop(self) -> None:
        @tool
        def search_child_chunks(query: str) -> str:
            """Dummy search tool returning large payload."""
            payload = [{"parent_id": "p1", "source": "manual.md", "content": "large chunk " * 200}]
            return json.dumps(payload)

        tool_call_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "search_child_chunks", "args": {"query": "large query"}, "id": "c1"}
            ],
        )
        compression_summary = AIMessage(content="### Compressed Knowledge Summary")
        final_answer = AIMessage(content="Final answer synthesized using compressed knowledge.")

        # Sequence of LLM invocations:
        # 1. Orchestrator -> requests search_child_chunks
        # (tools -> should_compress_context detects token threshold -> compress_context)
        # 2. compress_context -> compresses history to summary & emits RemoveMessages
        # 3. Orchestrator -> receives state with compressed summary, produces final_answer
        fake_llm = GenericFakeChatModel(
            messages=iter([tool_call_msg, compression_summary, final_answer])
        )

        graph = build_agent_subgraph(
            llm=fake_llm,
            tools=[search_child_chunks],
            max_iterations=4,
            max_tool_calls=4,
            base_token_threshold=50,  # Very low threshold to trigger compression
            token_growth_factor=0.9,
        )

        initial_state: AgentState = {
            "question": "Tell me about the system.",
            "messages": [],
        }

        output = graph.invoke(initial_state)

        assert output["final_answer"] == "Final answer synthesized using compressed knowledge."
        assert "### Compressed Knowledge Summary" in output["context_summary"]
        assert "search::large query" in output["context_summary"]
