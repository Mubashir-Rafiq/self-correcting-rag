"""Snapshot and integrity tests for all 6 core system prompts."""

from __future__ import annotations

from self_rag.agent.prompts import (
    get_aggregation_prompt,
    get_context_compression_prompt,
    get_conversation_summary_prompt,
    get_fallback_response_prompt,
    get_orchestrator_prompt,
    get_rewrite_query_prompt,
)


class TestPrompts:
    def test_conversation_summary_prompt(self) -> None:
        prompt = get_conversation_summary_prompt()
        assert "compact memory manager" in prompt
        assert "30-70 words" in prompt
        assert "## Instructions" in prompt
        assert "## Output" in prompt

    def test_rewrite_query_prompt(self) -> None:
        prompt = get_rewrite_query_prompt()
        assert "query rewriting specialist" in prompt
        assert "## Clarification Boundary" in prompt
        assert "maximum of 3 rewritten questions" in prompt

    def test_orchestrator_prompt(self) -> None:
        prompt = get_orchestrator_prompt()
        assert "document-grounded research assistant" in prompt
        assert "search_child_chunks" in prompt
        assert "Sources:" in prompt
        assert "- filename.ext" in prompt

    def test_fallback_response_prompt(self) -> None:
        prompt = get_fallback_response_prompt()
        assert "constrained evidence synthesizer" in prompt
        assert "Prefer current Retrieved Data over compressed context" in prompt
        assert "Sources:" in prompt

    def test_context_compression_prompt(self) -> None:
        prompt = get_context_compression_prompt()
        assert "research context compressor" in prompt
        assert "# Research Context Summary" in prompt
        assert "## Structured Findings" in prompt
        assert "## Gaps" in prompt

    def test_aggregation_prompt(self) -> None:
        prompt = get_aggregation_prompt()
        assert "final-answer synthesizer" in prompt
        assert "Sources:" in prompt
        assert "I couldn't find any information to answer your question" in prompt
