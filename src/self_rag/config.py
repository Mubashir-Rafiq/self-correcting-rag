"""Every tunable setting in the system, in one validated place.

Settings are read from environment variables (prefixed ``SELF_RAG_``) and from a ``.env`` file,
falling back to the defaults below. Values are validated at construction, so a bad configuration
fails immediately and loudly rather than surfacing as a confusing error deep inside a graph run.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LlmProvider = Literal["google_genai", "groq"]

DEFAULT_MARKDOWN_HEADERS: tuple[tuple[str, str], ...] = (
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
)


class Settings(BaseSettings):
    """Validated application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="SELF_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Storage locations -------------------------------------------------
    # One knob drives every runtime path, so they can never drift apart.
    data_dir: Path = Path("data")

    # --- Qdrant ------------------------------------------------------------
    child_collection: str = "document_child_chunks"
    sparse_vector_name: str = "sparse"

    # --- Embedding models --------------------------------------------------
    dense_model: str = "BAAI/bge-small-en-v1.5"
    # Declared rather than probed, so a model swap that changes dimensions is caught by an
    # explicit check instead of silently corrupting an existing collection.
    dense_dimension: int = Field(default=384, gt=0)
    sparse_model: str = "Qdrant/bm25"

    # --- Language model ----------------------------------------------------
    llm_provider: LlmProvider = "google_genai"
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_retries: int = Field(default=3, ge=0)
    # Free tiers are measured in requests per minute, and a fan-out fires several at once.
    llm_requests_per_second: float = Field(default=0.5, gt=0.0)
    llm_max_concurrency: int = Field(default=1, ge=1)

    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("SELF_RAG_GOOGLE_API_KEY", "GOOGLE_API_KEY"),
    )
    groq_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("SELF_RAG_GROQ_API_KEY", "GROQ_API_KEY"),
    )

    # --- Retrieval ---------------------------------------------------------
    retrieval_k: int = Field(default=7, gt=0)
    # NOTE: in hybrid mode Qdrant ranks by Reciprocal Rank Fusion, not cosine similarity. A fused
    # score is 1/(rank+2) summed across the dense and sparse branches, so it maxes out at 1.0 and
    # says nothing about semantic closeness. A floor of 0.4 silently drops results ranked worse
    # than 1st in both branches. Left disabled by default; see docs/stages/stage-00.
    hybrid_score_floor: float | None = Field(default=None, ge=0.0, le=2.0)

    # --- Agent research budget ---------------------------------------------
    max_tool_calls: int = Field(default=4, gt=0)
    max_iterations: int = Field(default=4, gt=0)
    max_subquestions: int = Field(default=2, gt=0)
    graph_recursion_limit: int = Field(default=50, gt=0)

    # --- Conversation memory -----------------------------------------------
    # Must be >= 2: the code keeps (n - 1) messages verbatim, and a keep-count of 0 would slice
    # the history list in a way that silently keeps everything instead of nothing.
    main_history_messages_to_keep: int = Field(default=4, ge=2)
    base_token_threshold: int = Field(default=2000, gt=0)
    token_growth_factor: float = Field(default=0.9, ge=0.0)

    # --- Chunking ----------------------------------------------------------
    child_chunk_size: int = Field(default=500, gt=0)
    child_chunk_overlap: int = Field(default=100, ge=0)
    min_parent_size: int = Field(default=2000, gt=0)
    max_parent_size: int = Field(default=4000, gt=0)
    markdown_headers: tuple[tuple[str, str], ...] = DEFAULT_MARKDOWN_HEADERS

    # --- Observability -----------------------------------------------------
    execution_logging_enabled: bool = False
    execution_log_max_chars: int = Field(default=1200, gt=0)
    langfuse_enabled: bool = False
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "http://localhost:3000"

    # --- Derived paths -----------------------------------------------------
    @property
    def markdown_dir(self) -> Path:
        return self.data_dir / "markdown"

    @property
    def parent_store_dir(self) -> Path:
        return self.data_dir / "parent_store"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def checkpoint_db_path(self) -> Path:
        return self.data_dir / "checkpoints.db"

    # --- Cross-field validation --------------------------------------------
    @model_validator(mode="after")
    def _check_chunk_sizes(self) -> Settings:
        if self.max_parent_size < self.min_parent_size:
            raise ValueError(
                f"max_parent_size ({self.max_parent_size}) must be >= "
                f"min_parent_size ({self.min_parent_size})"
            )
        if self.child_chunk_overlap >= self.child_chunk_size:
            raise ValueError(
                f"child_chunk_overlap ({self.child_chunk_overlap}) must be < "
                f"child_chunk_size ({self.child_chunk_size})"
            )
        if self.child_chunk_overlap >= self.max_parent_size:
            raise ValueError(
                f"child_chunk_overlap ({self.child_chunk_overlap}) must be < "
                f"max_parent_size ({self.max_parent_size})"
            )
        return self

    def api_key_for_provider(self) -> SecretStr | None:
        """Return the API key belonging to the selected provider, if one is configured."""
        keys: dict[LlmProvider, SecretStr | None] = {
            "google_genai": self.google_api_key,
            "groq": self.groq_api_key,
        }
        return keys[self.llm_provider]

    def require_llm_credentials(self) -> SecretStr:
        """Assert that the selected provider has a usable API key.

        Called at the point the language model is actually constructed, so that commands which do
        not talk to a model (inspecting config, chunking a file) still work without credentials.
        """
        key = self.api_key_for_provider()
        if key is None or not key.get_secret_value().strip():
            env_name = "GOOGLE_API_KEY" if self.llm_provider == "google_genai" else "GROQ_API_KEY"
            raise RuntimeError(
                f"llm_provider is '{self.llm_provider}' but no API key was found. "
                f"Set {env_name} in your environment or .env file."
            )
        return key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, constructed once."""
    return Settings()
