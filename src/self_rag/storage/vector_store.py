"""Qdrant vector store manager, collection lifecycle, and purge primitives.

Manages:
- QdrantClient lifecycle (in-memory, local on-disk, or remote)
- Collection creation with dense + sparse BM25 vector configurations
- Dimension guard: prevents silent vector space corruption by validating dense dimensions
- Purge primitive: ``delete_by_source`` for clean document updates and orphan prevention
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import TracebackType
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from self_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Base exception for storage and vector database errors."""


class DimensionMismatchError(StorageError):
    """Raised when an existing collection's vector dimension doesn't match the configuration."""


class CollectionNotFoundError(StorageError):
    """Raised when an operation requires an existing collection that does not exist."""


class VectorStoreManager:
    """Manages Qdrant collections, client connections, and document purges."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: QdrantClient | None = None,
        path: Path | str | None = None,
        collection_name: str | None = None,
        dense_dimension: int | None = None,
        sparse_vector_name: str | None = None,
    ) -> None:
        self.settings: Settings = settings if settings is not None else get_settings()
        self.collection_name: str = collection_name or self.settings.child_collection
        self.dense_dimension: int = dense_dimension or self.settings.dense_dimension
        self.sparse_vector_name: str = sparse_vector_name or self.settings.sparse_vector_name

        if client is not None:
            self.client: QdrantClient = client
        elif path is not None:
            if str(path) == ":memory:":
                self.client = QdrantClient(":memory:")
            else:
                p = Path(path)
                p.mkdir(parents=True, exist_ok=True)
                self.client = QdrantClient(path=str(p))
        elif self.settings.qdrant_url:
            api_key_val = (
                self.settings.qdrant_api_key.get_secret_value()
                if self.settings.qdrant_api_key
                else None
            )
            self.client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=api_key_val,
            )
        else:
            p = self.settings.qdrant_path
            p.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(p))

    def collection_exists(self, collection_name: str | None = None) -> bool:
        """Check whether the collection exists in Qdrant."""
        name = collection_name or self.collection_name
        return bool(self.client.collection_exists(name))

    def create_collection(
        self,
        collection_name: str | None = None,
        dense_dimension: int | None = None,
    ) -> None:
        """Create a collection with dense cosine vector and named sparse vector configs."""
        name = collection_name or self.collection_name
        dim = dense_dimension or self.dense_dimension

        self.client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=dim,
                distance=qmodels.Distance.COSINE,
            ),
            sparse_vectors_config={
                self.sparse_vector_name: qmodels.SparseVectorParams(
                    modifier=qmodels.Modifier.IDF,
                ),
            },
        )
        logger.info("Created Qdrant collection '%s' with dimension %d", name, dim)

    def check_dimension(
        self,
        collection_name: str | None = None,
        expected_dimension: int | None = None,
    ) -> None:
        """Validate that an existing collection's dense vector dimension matches expectation.

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            DimensionMismatchError: If the collection dimension does not match expected_dimension.
        """
        name = collection_name or self.collection_name
        expected = expected_dimension or self.dense_dimension

        if not self.collection_exists(name):
            raise CollectionNotFoundError(f"Collection '{name}' does not exist")

        info = self.client.get_collection(name)
        vectors = info.config.params.vectors

        actual_dim: int | None = None
        if isinstance(vectors, qmodels.VectorParams):
            actual_dim = vectors.size
        elif isinstance(vectors, dict):
            # Check standard names or first available vector param
            for key in ("", "dense"):
                val = vectors.get(key)
                if isinstance(val, qmodels.VectorParams):
                    actual_dim = val.size
                    break
            if actual_dim is None:
                for val in vectors.values():
                    if isinstance(val, qmodels.VectorParams):
                        actual_dim = val.size
                        break

        if actual_dim is not None and actual_dim != expected:
            raise DimensionMismatchError(
                f"Collection '{name}' has vector dimension {actual_dim}, "
                f"but configured dense_dimension is {expected}. "
                "Recreate the collection or update configuration."
            )

    def ensure_collection(
        self,
        collection_name: str | None = None,
        dense_dimension: int | None = None,
    ) -> bool:
        """Ensure the collection exists and validates its vector dimension.

        Returns True if a new collection was created, False if it already existed.

        Raises:
            DimensionMismatchError: If an existing collection has conflicting dimensions.
        """
        name = collection_name or self.collection_name
        dim = dense_dimension or self.dense_dimension

        if self.collection_exists(name):
            self.check_dimension(name, expected_dimension=dim)
            return False

        self.create_collection(name, dense_dimension=dim)
        return True

    def delete_collection(self, collection_name: str | None = None) -> bool:
        """Delete a collection. Returns True if deleted, False if it did not exist."""
        name = collection_name or self.collection_name
        if self.collection_exists(name):
            self.client.delete_collection(name)
            logger.info("Deleted Qdrant collection '%s'", name)
            return True
        return False

    def recreate_collection(
        self,
        collection_name: str | None = None,
        dense_dimension: int | None = None,
    ) -> None:
        """Drop collection if it exists, then create a clean empty collection."""
        name = collection_name or self.collection_name
        self.delete_collection(name)
        self.create_collection(name, dense_dimension=dense_dimension)

    def count(self, collection_name: str | None = None) -> int:
        """Return the number of points in the collection, or 0 if it doesn't exist."""
        name = collection_name or self.collection_name
        if not self.collection_exists(name):
            return 0
        return int(self.client.count(name).count)

    def delete_by_source(
        self,
        source_name: str,
        collection_name: str | None = None,
    ) -> int:
        """Purge all child chunks indexed from *source_name*.

        Targets both ``metadata.source`` (LangChain Qdrant convention) and flat ``source`` keys.
        Returns the number of points deleted.
        """
        name = collection_name or self.collection_name
        if not self.collection_exists(name):
            return 0

        count_before = self.count(name)
        if count_before == 0:
            return 0

        self.client.delete(
            collection_name=name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    should=[
                        qmodels.FieldCondition(
                            key="metadata.source",
                            match=qmodels.MatchValue(value=source_name),
                        ),
                        qmodels.FieldCondition(
                            key="source",
                            match=qmodels.MatchValue(value=source_name),
                        ),
                    ]
                )
            ),
        )

        count_after = self.count(name)
        deleted = max(0, count_before - count_after)
        logger.info(
            "Purged %d points for source '%s' from collection '%s'",
            deleted,
            source_name,
            name,
        )
        return deleted

    def get_vector_store(
        self,
        embedding: Embeddings,
        *,
        sparse_embedding: SparseEmbeddings | None = None,
        retrieval_mode: RetrievalMode | None = None,
        collection_name: str | None = None,
        **kwargs: Any,
    ) -> QdrantVectorStore:
        """Return a configured ``QdrantVectorStore`` for the managed collection.

        Automatically ensures the collection exists and validates dimensions.
        Defaults to ``RetrievalMode.HYBRID`` if *sparse_embedding* is provided,
        or ``RetrievalMode.DENSE`` otherwise.
        """
        name = collection_name or self.collection_name
        self.ensure_collection(name)

        mode = retrieval_mode
        if mode is None:
            mode = RetrievalMode.HYBRID if sparse_embedding is not None else RetrievalMode.DENSE

        return QdrantVectorStore(
            client=self.client,
            collection_name=name,
            embedding=embedding,
            sparse_embedding=sparse_embedding,
            sparse_vector_name=self.sparse_vector_name,
            retrieval_mode=mode,
            **kwargs,
        )

    def close(self) -> None:
        """Close the underlying Qdrant client connection."""
        self.client.close()

    def __enter__(self) -> VectorStoreManager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
