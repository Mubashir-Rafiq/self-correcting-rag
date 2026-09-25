"""Embedding models and factories for dense and sparse vector representations.

Provides:
- ``FastEmbedEmbeddings``: LangChain-compatible ``Embeddings`` adapter backed by FastEmbed's
  lightweight ONNX runtime (default: BAAI/bge-small-en-v1.5, 384 dimensions).
- Dense and sparse embedding factories wired to configuration defaults.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse

from self_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)


class FastEmbedEmbeddings(Embeddings):
    """LangChain ``Embeddings`` interface wrapping FastEmbed's ONNX runtime."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        batch_size: int = 256,
        cache_dir: str | None = None,
        threads: int | None = None,
        parallel: int | None = None,
        lazy_load: bool = True,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.threads = threads
        self.parallel = parallel
        self.kwargs = kwargs
        self._model: TextEmbedding | None = None

        if not lazy_load:
            self._ensure_model()

    def _ensure_model(self) -> TextEmbedding:
        if self._model is None:
            logger.info("Initializing FastEmbed TextEmbedding model '%s'", self.model_name)
            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                threads=self.threads,
                **self.kwargs,
            )
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document strings into dense vectors."""
        if not texts:
            return []
        model = self._ensure_model()
        results = model.embed(texts, batch_size=self.batch_size, parallel=self.parallel)
        return [cast(list[float], arr.tolist()) for arr in results]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string into a dense vector."""
        model = self._ensure_model()
        results = list(model.embed([text], batch_size=1, parallel=self.parallel))
        if not results:
            return []
        return cast(list[float], results[0].tolist())


def get_dense_embeddings(settings: Settings | None = None) -> FastEmbedEmbeddings:
    """Return a configured ``FastEmbedEmbeddings`` instance using application settings."""
    cfg = settings if settings is not None else get_settings()
    return FastEmbedEmbeddings(model_name=cfg.dense_model)


def get_sparse_embeddings(settings: Settings | None = None) -> FastEmbedSparse:
    """Return a configured ``FastEmbedSparse`` BM25 instance using application settings."""
    cfg = settings if settings is not None else get_settings()
    return FastEmbedSparse(model_name=cfg.sparse_model)
