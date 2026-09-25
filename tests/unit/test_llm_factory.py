"""Tests for self_rag.llm.factory — model construction, rate limiting, and credential handling."""

from __future__ import annotations

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import SecretStr

from self_rag.config import Settings
from self_rag.llm.factory import (
    MissingApiKeyError,
    build_llm,
    build_rate_limiter,
)


class TestLLMFactory:
    def test_missing_api_key_raises_error(self) -> None:
        settings = Settings(
            llm_provider="google_genai",
            google_api_key=None,
            groq_api_key=None,
        )
        with pytest.raises(MissingApiKeyError, match="No API key configured"):
            build_llm(settings)

    def test_build_rate_limiter(self) -> None:
        settings = Settings(llm_requests_per_second=2.0, llm_max_concurrency=4)
        limiter = build_rate_limiter(settings)

        assert limiter.requests_per_second == 2.0
        assert limiter.max_bucket_size == 4.0

    def test_build_google_llm(self) -> None:
        settings = Settings(
            llm_provider="google_genai",
            llm_model="gemini-2.5-flash",
            google_api_key=SecretStr("test_google_key"),
            llm_temperature=0.3,
            llm_max_retries=2,
        )
        model = build_llm(settings)

        assert isinstance(model, ChatGoogleGenerativeAI)
        assert model.temperature == 0.3
        assert model.max_retries == 2
        assert model.rate_limiter is not None

    def test_build_groq_llm(self) -> None:
        settings = Settings(
            llm_provider="groq",
            llm_model="llama-3.3-70b-versatile",
            groq_api_key=SecretStr("test_groq_key"),
            llm_temperature=0.7,
            llm_max_retries=3,
        )
        model = build_llm(settings)

        assert isinstance(model, ChatGroq)
        assert model.temperature == 0.7
        assert model.max_retries == 3
        assert model.rate_limiter is not None
