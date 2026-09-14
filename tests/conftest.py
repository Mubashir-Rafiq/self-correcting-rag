"""Shared test fixtures.

Settings read from the ambient environment, so tests must not inherit whatever happens to be
exported in the developer's shell. The autouse fixture below strips anything that could leak in.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from self_rag.config import get_settings

_LEAKY_ENV_NAMES = frozenset({"GOOGLE_API_KEY", "GROQ_API_KEY"})


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in list(os.environ):
        if name.startswith("SELF_RAG_") or name in _LEAKY_ENV_NAMES:
            monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
