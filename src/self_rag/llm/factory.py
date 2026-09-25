"""Provider-agnostic LLM initialization, rate limiting, and retry handling.

Constructs LangChain ``BaseChatModel`` instances using ``init_chat_model``, configuring:
- Provider credentials (Google GenAI, Groq)
- Global token/request rate limiter (InMemoryRateLimiter)
- Retries and concurrency settings
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter

from self_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM configuration and initialization failures."""


class MissingApiKeyError(LLMError):
    """Raised when the required API key for the chosen LLM provider is not found."""


class UnsupportedProviderError(LLMError):
    """Raised when an unknown LLM provider is specified."""


def build_rate_limiter(settings: Settings | None = None) -> InMemoryRateLimiter:
    """Build an ``InMemoryRateLimiter`` from settings to respect provider rate limits."""
    cfg = settings if settings is not None else get_settings()
    return InMemoryRateLimiter(
        requests_per_second=cfg.llm_requests_per_second,
        check_every_n_seconds=0.1,
        max_bucket_size=float(cfg.llm_max_concurrency),
    )


def build_llm(
    settings: Settings | None = None,
    *,
    model: str | None = None,
    temperature: float | None = None,
    rate_limiter: BaseRateLimiter | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Construct and return a configured chat model instance.

    Args:
        settings: Application settings. Defaults to ``get_settings()``.
        model: Optional model name override.
        temperature: Optional temperature override.
        rate_limiter: Optional custom rate limiter. If None, builds an ``InMemoryRateLimiter``.
        **kwargs: Extra parameters forwarded to ``init_chat_model``.

    Raises:
        MissingApiKeyError: If the configured provider's API key is not present.
        UnsupportedProviderError: If the provider is not supported.
    """
    cfg = settings if settings is not None else get_settings()
    provider = cfg.llm_provider
    model_name = model if model is not None else cfg.llm_model

    if provider not in ("google_genai", "groq"):
        raise UnsupportedProviderError(f"Unsupported LLM provider '{provider}'")

    api_key_secret = cfg.api_key_for_provider()
    if api_key_secret is None or not api_key_secret.get_secret_value().strip():
        var_name = f"{provider.upper()}_API_KEY"
        raise MissingApiKeyError(
            f"No API key configured for provider '{provider}'. "
            f"Set SELF_RAG_{var_name} or {var_name}."
        )

    api_key = api_key_secret.get_secret_value()
    temp = temperature if temperature is not None else cfg.llm_temperature
    limiter = rate_limiter if rate_limiter is not None else build_rate_limiter(cfg)

    logger.info(
        "Initializing chat model '%s' (provider: %s, temp: %.1f, retries: %d)",
        model_name,
        provider,
        temp,
        cfg.llm_max_retries,
    )

    llm = init_chat_model(
        model_name,
        model_provider=provider,
        api_key=api_key,
        temperature=temp,
        max_retries=cfg.llm_max_retries,
        rate_limiter=limiter,
        **kwargs,
    )

    if not isinstance(llm, BaseChatModel):
        raise LLMError(f"Expected BaseChatModel from init_chat_model, got {type(llm)}")

    return llm
