"""Tests for self_rag.text.tokens — token estimation."""

from __future__ import annotations

from self_rag.text.tokens import estimate_tokens


class TestEstimateTokens:
    def test_empty_string_is_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_single_word(self) -> None:
        # "hello" is a single token in cl100k_base.
        assert estimate_tokens("hello") >= 1

    def test_longer_text_is_proportional(self) -> None:
        short = estimate_tokens("hello world")
        long = estimate_tokens("hello world " * 100)
        assert long > short

    def test_deterministic(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        assert estimate_tokens(text) == estimate_tokens(text)

    def test_known_value(self) -> None:
        # With cl100k_base, "hello world" is 2 tokens.
        assert estimate_tokens("hello world") == 2
