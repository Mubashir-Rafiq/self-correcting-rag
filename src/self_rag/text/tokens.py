"""Token-count estimation backed by tiktoken.

The main use is the context-compression trigger in later stages: when the accumulated retrieval
context approaches a threshold (measured in tokens, not characters), the agent compresses before
the next LLM call.  Having a fast, deterministic estimator here keeps that logic testable without
loading a model.

The ``cl100k_base`` encoding is used because it closely matches the tokenisation of GPT-4-class
models and is a reasonable proxy for Gemini and other providers — the estimate does not need to be
exact, only close enough to trigger compression at the right order of magnitude.
"""

from __future__ import annotations

from collections.abc import Sequence

import tiktoken
from langchain_core.messages import BaseMessage

# Lazy-initialised module-level encoder so the (small) startup cost is paid at most once.
_ENCODING: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def estimate_tokens(text: str) -> int:
    """Return the approximate token count for *text*.

    Uses the ``cl100k_base`` encoding (GPT-4 family).  The result is deterministic for a given
    input, so it is safe to cache or snapshot-test.
    """
    return len(_get_encoding().encode(text))


def estimate_messages_tokens(messages: Sequence[BaseMessage]) -> int:
    """Return the approximate total token count for a sequence of messages."""
    total = 0
    for m in messages:
        content = m.content
        if isinstance(content, str):
            total += estimate_tokens(content)
        else:
            parts = [str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content]
            total += estimate_tokens(" ".join(parts))
    return total
