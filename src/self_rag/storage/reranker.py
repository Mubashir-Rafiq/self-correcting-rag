"""Cross-encoder reranking for retrieval candidates.

Uses FastEmbed's ONNX-based TextCrossEncoder for fast, PyTorch-free cross-encoder
reranking between the candidate search results and the LLM response generator.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract interface for document rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        """Rerank a sequence of documents against *query* and return the top *top_n*."""


class FastEmbedReranker(BaseReranker):
    """Cross-encoder reranker powered by FastEmbed's ONNX runtime."""

    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._encoder: Any | None = None

    @property
    def encoder(self) -> Any:
        if self._encoder is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._encoder = TextCrossEncoder(model_name=self.model_name)
        return self._encoder

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        """Score each document against *query*, sort descending, and return top *top_n*."""
        if not documents:
            return []

        doc_list = list(documents)
        clean_query = query.strip()
        if not clean_query:
            return doc_list[:top_n] if top_n is not None else doc_list

        try:
            texts = [doc.page_content for doc in doc_list]
            scores = list(self.encoder.rerank(clean_query, texts))

            scored_docs: list[tuple[float, Document]] = []
            for score, doc in zip(scores, doc_list, strict=False):
                # Attach score to metadata for traceability
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "rerank_score": float(score)},
                )
                scored_docs.append((float(score), doc_copy))

            scored_docs.sort(key=lambda item: item[0], reverse=True)
            ranked = [doc for _, doc in scored_docs]
            return ranked[:top_n] if top_n is not None else ranked
        except Exception as err:
            logger.warning("Reranking failed (%s); returning original candidates", err)
            return doc_list[:top_n] if top_n is not None else doc_list


class PassthroughReranker(BaseReranker):
    """No-op reranker that preserves the input order."""

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        doc_list = list(documents)
        return doc_list[:top_n] if top_n is not None else doc_list
