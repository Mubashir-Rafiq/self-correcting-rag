"""Tests for self_rag.storage.embeddings — FastEmbed dense and sparse adapters."""

from __future__ import annotations

import pytest

from self_rag.config import Settings
from self_rag.storage.embeddings import (
    FastEmbedEmbeddings,
    get_dense_embeddings,
    get_sparse_embeddings,
)


@pytest.fixture()
def dense_embeddings() -> FastEmbedEmbeddings:
    """Return a FastEmbedEmbeddings instance with the default BAAI/bge-small-en-v1.5 model."""
    return FastEmbedEmbeddings()


class TestFastEmbedEmbeddings:
    def test_embed_documents(self, dense_embeddings: FastEmbedEmbeddings) -> None:
        texts = ["Machine learning in healthcare", "Distributed graph databases"]
        vectors = dense_embeddings.embed_documents(texts)

        assert len(vectors) == 2
        assert len(vectors[0]) == 384
        assert len(vectors[1]) == 384
        assert all(isinstance(v, float) for v in vectors[0])
        assert vectors[0] != vectors[1]

    def test_embed_empty_documents(self, dense_embeddings: FastEmbedEmbeddings) -> None:
        assert dense_embeddings.embed_documents([]) == []

    def test_embed_query(self, dense_embeddings: FastEmbedEmbeddings) -> None:
        query = "What is retrieval augmented generation?"
        vector = dense_embeddings.embed_query(query)

        assert len(vector) == 384
        assert all(isinstance(v, float) for v in vector)

    def test_factories(self) -> None:
        settings = Settings(dense_model="BAAI/bge-small-en-v1.5", sparse_model="Qdrant/bm25")
        dense = get_dense_embeddings(settings)
        sparse = get_sparse_embeddings(settings)

        assert isinstance(dense, FastEmbedEmbeddings)
        assert dense.model_name == "BAAI/bge-small-en-v1.5"
        assert getattr(sparse._model, "model_name", None) == "Qdrant/bm25"
