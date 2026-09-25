"""Tests for self_rag.storage.vector_store — Qdrant collection lifecycle and purge primitives.

All tests use in-memory Qdrant and fake embedding stubs so they execute instantly
without external services, GPU requirements, or network downloads.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_qdrant import RetrievalMode, SparseEmbeddings, SparseVector
from qdrant_client import QdrantClient

from self_rag.config import Settings
from self_rag.storage.vector_store import (
    CollectionNotFoundError,
    DimensionMismatchError,
    VectorStoreManager,
)


class FakeSparseEmbeddings(SparseEmbeddings):
    """Stub sparse embeddings for fast, deterministic unit testing."""

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1, 2], values=[0.5, 0.8]) for _ in texts]

    def embed_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[1, 2], values=[0.5, 0.8])


@pytest.fixture()
def client() -> QdrantClient:
    """Return an in-memory Qdrant client."""
    return QdrantClient(":memory:")


@pytest.fixture()
def vm(client: QdrantClient) -> VectorStoreManager:
    """Return a VectorStoreManager configured with in-memory client and 384 dim."""
    settings = Settings(
        child_collection="test_child_chunks",
        dense_dimension=384,
        sparse_vector_name="sparse",
    )
    return VectorStoreManager(settings=settings, client=client)


class TestCollectionLifecycle:
    def test_create_and_delete_collection(self, vm: VectorStoreManager) -> None:
        assert vm.collection_exists() is False
        assert vm.count() == 0

        vm.create_collection()
        assert vm.collection_exists() is True
        assert vm.count() == 0

        deleted = vm.delete_collection()
        assert deleted is True
        assert vm.collection_exists() is False

        # Deleting already deleted collection returns False
        assert vm.delete_collection() is False

    def test_ensure_collection(self, vm: VectorStoreManager) -> None:
        created = vm.ensure_collection()
        assert created is True
        assert vm.collection_exists() is True

        # Second call does not recreate
        created_again = vm.ensure_collection()
        assert created_again is False

    def test_recreate_collection(self, vm: VectorStoreManager) -> None:
        vm.ensure_collection()
        # Add a test vector directly
        store = vm.get_vector_store(embedding=FakeEmbeddings(size=384))
        store.add_documents([Document(page_content="hello", metadata={})])
        assert vm.count() == 1

        vm.recreate_collection()
        assert vm.count() == 0
        assert vm.collection_exists() is True


class TestDimensionGuard:
    def test_matching_dimension_passes(self, vm: VectorStoreManager) -> None:
        vm.create_collection(dense_dimension=384)
        # Should not raise
        vm.check_dimension(expected_dimension=384)

    def test_mismatched_dimension_raises_error(self, vm: VectorStoreManager) -> None:
        vm.create_collection(dense_dimension=384)
        with pytest.raises(DimensionMismatchError, match="has vector dimension 384"):
            vm.check_dimension(expected_dimension=768)

    def test_ensure_collection_enforces_dimension_guard(self, vm: VectorStoreManager) -> None:
        vm.create_collection(dense_dimension=384)
        with pytest.raises(DimensionMismatchError):
            vm.ensure_collection(dense_dimension=512)

    def test_check_dimension_missing_collection_raises(self, vm: VectorStoreManager) -> None:
        with pytest.raises(CollectionNotFoundError, match="does not exist"):
            vm.check_dimension(collection_name="nonexistent_collection")


class TestPurgePrimitives:
    def test_delete_by_source(self, vm: VectorStoreManager) -> None:
        store = vm.get_vector_store(embedding=FakeEmbeddings(size=384))
        docs = [
            Document(
                page_content="Chunk 1 from file A",
                metadata={"source": "file_a.pdf", "parent_id": "a_p0"},
            ),
            Document(
                page_content="Chunk 2 from file A",
                metadata={"source": "file_a.pdf", "parent_id": "a_p1"},
            ),
            Document(
                page_content="Chunk 1 from file B",
                metadata={"source": "file_b.pdf", "parent_id": "b_p0"},
            ),
        ]
        store.add_documents(docs)
        assert vm.count() == 3

        deleted = vm.delete_by_source("file_a.pdf")
        assert deleted == 2
        assert vm.count() == 1

        # Confirm remaining document is from file_b
        results = store.similarity_search("file", k=1)
        assert len(results) == 1
        assert results[0].metadata["source"] == "file_b.pdf"

    def test_delete_by_source_on_nonexistent_source(self, vm: VectorStoreManager) -> None:
        store = vm.get_vector_store(embedding=FakeEmbeddings(size=384))
        store.add_documents([Document(page_content="text", metadata={"source": "doc.pdf"})])

        deleted = vm.delete_by_source("nonexistent.pdf")
        assert deleted == 0
        assert vm.count() == 1

    def test_delete_by_source_on_nonexistent_collection(self, vm: VectorStoreManager) -> None:
        deleted = vm.delete_by_source("any.pdf", collection_name="not_a_collection")
        assert deleted == 0


class TestVectorStoreFactory:
    def test_get_vector_store_dense(self, vm: VectorStoreManager) -> None:
        embeddings = FakeEmbeddings(size=384)
        store = vm.get_vector_store(embedding=embeddings, retrieval_mode=RetrievalMode.DENSE)

        doc = Document(page_content="dense document", metadata={"source": "test.pdf"})
        store.add_documents([doc])
        results = store.similarity_search("dense", k=1)
        assert len(results) == 1
        assert results[0].page_content == "dense document"

    def test_get_vector_store_hybrid(self, vm: VectorStoreManager) -> None:
        embeddings = FakeEmbeddings(size=384)
        sparse_embeddings = FakeSparseEmbeddings()
        store = vm.get_vector_store(
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
        )

        doc = Document(page_content="hybrid document", metadata={"source": "test.pdf"})
        store.add_documents([doc])
        results = store.similarity_search("hybrid", k=1)
        assert len(results) == 1
        assert results[0].page_content == "hybrid document"

    def test_context_manager(self, client: QdrantClient) -> None:
        with VectorStoreManager(client=client) as manager:
            assert manager.collection_exists() is False
