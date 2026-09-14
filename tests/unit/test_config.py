"""The configuration layer is the one place a typo can quietly corrupt everything downstream,
so every guard rail it claims to have is asserted here."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from self_rag.config import Settings, get_settings


class IsolatedSettings(Settings):
    """Settings that ignore a developer's real .env file, so tests stay reproducible."""

    model_config = SettingsConfigDict(env_file=None)


def make_settings(**overrides: Any) -> Settings:
    return IsolatedSettings(**overrides)


def test_defaults_are_internally_consistent() -> None:
    settings = make_settings()

    assert settings.child_chunk_overlap < settings.child_chunk_size
    assert settings.min_parent_size <= settings.max_parent_size
    assert settings.main_history_messages_to_keep >= 2


def test_derived_paths_all_follow_the_data_dir() -> None:
    settings = make_settings(data_dir=Path("/tmp/example"))

    assert settings.markdown_dir == Path("/tmp/example/markdown")
    assert settings.parent_store_dir == Path("/tmp/example/parent_store")
    assert settings.qdrant_path == Path("/tmp/example/qdrant")
    assert settings.checkpoint_db_path == Path("/tmp/example/checkpoints.db")


def test_settings_are_immutable() -> None:
    settings = make_settings()

    with pytest.raises(ValidationError):
        # mypy knows this is illegal; the assertion is that it also fails at runtime.
        settings.retrieval_k = 99  # type: ignore[misc]


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SELF_RAG_RETRIEVAL_K", "3")
    monkeypatch.setenv("SELF_RAG_LLM_PROVIDER", "groq")

    settings = make_settings()

    assert settings.retrieval_k == 3
    assert settings.llm_provider == "groq"


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


class TestChunkSizeValidation:
    def test_parent_range_must_not_be_inverted(self) -> None:
        with pytest.raises(ValidationError, match="max_parent_size"):
            make_settings(min_parent_size=4000, max_parent_size=2000)

    def test_child_overlap_must_be_smaller_than_child_size(self) -> None:
        with pytest.raises(ValidationError, match="child_chunk_overlap"):
            make_settings(child_chunk_size=500, child_chunk_overlap=500)

    def test_child_overlap_must_be_smaller_than_max_parent_size(self) -> None:
        with pytest.raises(ValidationError, match="child_chunk_overlap"):
            make_settings(
                child_chunk_size=6000,
                child_chunk_overlap=5000,
                min_parent_size=1000,
                max_parent_size=4000,
            )


class TestBudgetValidation:
    def test_history_window_below_two_is_rejected(self) -> None:
        # A keep-count of 1 leaves 0 messages verbatim, and `list[:-0]` silently means
        # "the whole list" rather than "nothing" — so the floor is enforced here instead.
        with pytest.raises(ValidationError):
            make_settings(main_history_messages_to_keep=1)

    @pytest.mark.parametrize(
        "field",
        ["retrieval_k", "max_tool_calls", "max_iterations", "max_subquestions"],
    )
    def test_budgets_must_be_positive(self, field: str) -> None:
        with pytest.raises(ValidationError):
            make_settings(**{field: 0})


class TestCredentials:
    def test_api_key_is_selected_by_provider(self) -> None:
        settings = make_settings(
            llm_provider="groq", google_api_key="google-key", groq_api_key="groq-key"
        )

        key = settings.api_key_for_provider()

        assert key is not None
        assert key.get_secret_value() == "groq-key"

    def test_missing_key_names_the_variable_to_set(self) -> None:
        settings = make_settings(llm_provider="google_genai", google_api_key=None)

        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            settings.require_llm_credentials()

    def test_blank_key_is_treated_as_missing(self) -> None:
        settings = make_settings(llm_provider="groq", groq_api_key="   ")

        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            settings.require_llm_credentials()

    def test_present_key_is_returned(self) -> None:
        settings = make_settings(llm_provider="groq", groq_api_key="groq-key")

        assert settings.require_llm_credentials().get_secret_value() == "groq-key"

    def test_secrets_are_not_exposed_by_repr(self) -> None:
        settings = make_settings(groq_api_key="super-secret-value")

        assert "super-secret-value" not in repr(settings)
