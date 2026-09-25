"""End-to-end ingestion pipeline: convert, chunk, store parents, and index children.

Orchestrates the entire document ingestion workflow with:
- Content-hash change detection: skips unchanged documents.
- Stale-chunk purge: removes prior parent files and vector embeddings before re-indexing.
- Atomic rollback: leaves no orphaned parents, vectors, or partial markdown if an error occurs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings

from self_rag.config import Settings, get_settings
from self_rag.ingestion.chunker import DocumentChunker
from self_rag.ingestion.converters import convert_file, file_sha256
from self_rag.storage.embeddings import get_dense_embeddings, get_sparse_embeddings
from self_rag.storage.parent_store import ParentStore
from self_rag.storage.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    """Outcome of attempting to ingest a single document."""

    source_path: Path
    status: Literal["added", "skipped", "failed"]
    parent_count: int = 0
    child_count: int = 0
    message: str = ""
    error: str | None = None


class IngestionPipeline:
    """Unified ingestion pipeline coordinating conversion, chunking, and dual-store indexing."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        parent_store: ParentStore | None = None,
        vector_store_manager: VectorStoreManager | None = None,
        dense_embeddings: Embeddings | None = None,
        sparse_embeddings: SparseEmbeddings | None = None,
        chunker: DocumentChunker | None = None,
    ) -> None:
        self.settings: Settings = settings if settings is not None else get_settings()
        self.parent_store: ParentStore = (
            parent_store if parent_store is not None else ParentStore(settings=self.settings)
        )
        self.vector_store_manager: VectorStoreManager = (
            vector_store_manager
            if vector_store_manager is not None
            else VectorStoreManager(settings=self.settings)
        )
        self.dense_embeddings: Embeddings = (
            dense_embeddings
            if dense_embeddings is not None
            else get_dense_embeddings(self.settings)
        )
        self.sparse_embeddings: SparseEmbeddings = (
            sparse_embeddings
            if sparse_embeddings is not None
            else get_sparse_embeddings(self.settings)
        )
        self.chunker: DocumentChunker = (
            chunker if chunker is not None else DocumentChunker(self.settings)
        )

    def _hash_sidecar_path(self, md_path: Path) -> Path:
        return md_path.with_suffix(".md.sha256")

    def is_unchanged(self, source_path: Path, current_hash: str) -> bool:
        """Check whether *source_path* matches its stored markdown and hash sidecar."""
        md_path = (self.settings.markdown_dir / source_path.stem).with_suffix(".md")
        hash_path = self._hash_sidecar_path(md_path)

        if not md_path.is_file() or not hash_path.is_file():
            return False

        try:
            stored_hash = hash_path.read_text(encoding="utf-8").strip()
            return stored_hash == current_hash
        except OSError:
            return False

    def ingest_file(self, source_path: Path, *, force: bool = False) -> IngestionResult:
        """Ingest a single document file into the dual storage layer.

        1. Computes SHA-256 hash.
        2. Skips if unchanged (unless force=True).
        3. Purges stale parents and vectors for this source.
        4. Converts to Markdown and chunks into parents and children.
        5. Saves parents to disk and child vectors to Qdrant.
        6. Writes sidecar hash file on success.
        7. On exception, rolls back any partial writes and cleans up.
        """
        source_path = Path(source_path)
        if not source_path.exists():
            return IngestionResult(
                source_path=source_path,
                status="failed",
                error="File does not exist",
                message=f"File not found: {source_path}",
            )

        try:
            source_hash = file_sha256(source_path)
        except Exception as err:
            return IngestionResult(
                source_path=source_path,
                status="failed",
                error=str(err),
                message=f"Failed to compute file hash: {err}",
            )

        if not force and self.is_unchanged(source_path, source_hash):
            logger.info("File %s unchanged; skipping ingestion", source_path.name)
            return IngestionResult(
                source_path=source_path,
                status="skipped",
                message=f"File '{source_path.name}' is unchanged; skipped.",
            )

        # Track state for atomic rollback on failure
        source_name = source_path.name
        converted_md: Path | None = None
        hash_path: Path | None = None
        created_parent_ids: list[str] = []

        try:
            # 1. Purge stale artifacts first
            self.vector_store_manager.delete_by_source(source_name)
            self.parent_store.delete_by_source(source_name)

            # 2. Convert to markdown
            converted_md = convert_file(source_path, self.settings.markdown_dir)
            hash_path = self._hash_sidecar_path(converted_md)

            # 3. Chunk file
            parents, children = self.chunker.chunk_file(converted_md, source_name=source_name)

            # 4. Save parent chunks
            for pid, doc in parents:
                self.parent_store.save(pid, doc)
                created_parent_ids.append(pid)

            # 5. Index children in Qdrant
            if children:
                store = self.vector_store_manager.get_vector_store(
                    embedding=self.dense_embeddings,
                    sparse_embedding=self.sparse_embeddings,
                )
                store.add_documents(children)

            # 6. Write sidecar hash file to mark successful completion
            hash_path.write_text(source_hash, encoding="utf-8")

            logger.info(
                "Ingested '%s': %d parents, %d children",
                source_name,
                len(parents),
                len(children),
            )
            return IngestionResult(
                source_path=source_path,
                status="added",
                parent_count=len(parents),
                child_count=len(children),
                message=f"Indexed {len(parents)} parents and {len(children)} children.",
            )

        except Exception as err:
            logger.exception("Failed to ingest '%s'; rolling back partial artifacts", source_name)
            # Rollback: purge vectors, delete saved parents, remove markdown & hash files
            self.vector_store_manager.delete_by_source(source_name)
            if created_parent_ids:
                self.parent_store.delete_many(created_parent_ids)
            else:
                self.parent_store.delete_by_source(source_name)

            if hash_path is not None and hash_path.exists():
                hash_path.unlink(missing_ok=True)
            if converted_md is not None and converted_md.exists():
                converted_md.unlink(missing_ok=True)

            return IngestionResult(
                source_path=source_path,
                status="failed",
                error=str(err),
                message=f"Failed to ingest: {err}",
            )

    def ingest_files(
        self,
        paths: Sequence[Path],
        *,
        force: bool = False,
    ) -> list[IngestionResult]:
        """Ingest a batch of documents sequentially."""
        return [self.ingest_file(p, force=force) for p in paths]

    def list_sources(self) -> list[str]:
        """Return a sorted list of unique document source names currently in the store."""
        return self.parent_store.list_sources()

    def list_documents(self) -> list[dict[str, Any]]:
        """Return details for all indexed documents."""
        sources = self.list_sources()
        results: list[dict[str, Any]] = []

        for src in sources:
            stem = Path(src).stem
            md_path = (self.settings.markdown_dir / stem).with_suffix(".md")
            hash_path = self._hash_sidecar_path(md_path)

            sha = ""
            if hash_path.is_file():
                try:
                    sha = hash_path.read_text(encoding="utf-8").strip()
                except OSError:
                    sha = ""

            # Count parents
            parent_ids = [
                pid
                for pid in self.parent_store.list_parent_ids()
                if pid.startswith(f"{stem}_p") or pid.startswith(f"{src}_p")
            ]

            results.append(
                {
                    "source": src,
                    "parents": len(parent_ids),
                    "sha256": sha,
                    "markdown_path": str(md_path) if md_path.exists() else "",
                }
            )

        return results

    def delete_source(self, source_name: str) -> None:
        """Completely remove all artifacts (vectors, parents, markdown, hash) for *source_name*."""
        self.vector_store_manager.delete_by_source(source_name)
        self.parent_store.delete_by_source(source_name)

        stem = Path(source_name).stem
        md_path = (self.settings.markdown_dir / stem).with_suffix(".md")
        hash_path = self._hash_sidecar_path(md_path)

        if md_path.exists():
            md_path.unlink(missing_ok=True)
        if hash_path.exists():
            hash_path.unlink(missing_ok=True)
        logger.info("Purged all artifacts for '%s'", source_name)

    def clear_all(self) -> None:
        """Drop all indexed data: drops collection, clears parent store, wipes markdown dir."""
        self.vector_store_manager.recreate_collection()
        self.parent_store.clear()

        if self.settings.markdown_dir.exists():
            for p in self.settings.markdown_dir.iterdir():
                if p.is_file():
                    p.unlink(missing_ok=True)
        logger.info("Cleared all indexed data and storage directories")
