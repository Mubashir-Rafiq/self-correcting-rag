"""Storage package for self-correcting-rag.

Provides:
- ``ParentStore``: on-disk JSON storage for full-text parent chunks.
- ``VectorStoreManager``: Qdrant client lifecycle, dimension validation, and purge primitives.
"""

from __future__ import annotations

from self_rag.storage.embeddings import (
    FastEmbedEmbeddings,
    get_dense_embeddings,
    get_sparse_embeddings,
)
from self_rag.storage.parent_store import ParentStore
from self_rag.storage.reranker import (
    BaseReranker,
    FastEmbedReranker,
    PassthroughReranker,
)
from self_rag.storage.vector_store import (
    CollectionNotFoundError,
    DimensionMismatchError,
    StorageError,
    VectorStoreManager,
)

__all__ = [
    "BaseReranker",
    "CollectionNotFoundError",
    "DimensionMismatchError",
    "FastEmbedEmbeddings",
    "FastEmbedReranker",
    "ParentStore",
    "PassthroughReranker",
    "StorageError",
    "VectorStoreManager",
    "get_dense_embeddings",
    "get_sparse_embeddings",
]
