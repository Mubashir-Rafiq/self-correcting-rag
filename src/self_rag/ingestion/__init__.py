"""Document ingestion — conversion, hashing, chunking, and the unified pipeline."""

from __future__ import annotations

from self_rag.ingestion.chunker import DocumentChunker
from self_rag.ingestion.converters import (
    convert_file,
    file_sha256,
    markdown_passthrough,
    pdf_to_markdown,
)
from self_rag.ingestion.pipeline import IngestionPipeline, IngestionResult

__all__ = [
    "DocumentChunker",
    "IngestionPipeline",
    "IngestionResult",
    "convert_file",
    "file_sha256",
    "markdown_passthrough",
    "pdf_to_markdown",
]
