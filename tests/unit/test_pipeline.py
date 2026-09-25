"""Tests for self_rag.ingestion.pipeline — unified document ingestion workflow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.embeddings import FakeEmbeddings
from langchain_qdrant import SparseEmbeddings, SparseVector
from qdrant_client import QdrantClient

from self_rag.config import Settings
from self_rag.ingestion.chunker import DocumentChunker
from self_rag.ingestion.converters import file_sha256
from self_rag.ingestion.pipeline import IngestionPipeline
from self_rag.storage.parent_store import ParentStore
from self_rag.storage.vector_store import VectorStoreManager


class FakeSparseEmbeddings(SparseEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1, 2], values=[0.5, 0.8]) for _ in texts]

    def embed_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[1, 2], values=[0.5, 0.8])


@pytest.fixture()
def pipeline(tmp_path: Path) -> IngestionPipeline:
    """Return an IngestionPipeline backed by temporary directories and in-memory Qdrant."""
    data_dir = tmp_path / "data"
    settings = Settings(
        data_dir=data_dir,
        child_collection="test_chunks",
        dense_dimension=384,
        child_chunk_size=100,
        child_chunk_overlap=20,
        min_parent_size=50,
        max_parent_size=500,
    )

    client = QdrantClient(":memory:")
    vm = VectorStoreManager(settings=settings, client=client)
    parent_store = ParentStore(directory=settings.parent_store_dir)
    dense = FakeEmbeddings(size=384)
    sparse = FakeSparseEmbeddings()
    chunker = DocumentChunker(settings)

    return IngestionPipeline(
        settings=settings,
        parent_store=parent_store,
        vector_store_manager=vm,
        dense_embeddings=dense,
        sparse_embeddings=sparse,
        chunker=chunker,
    )


class TestIngestionWorkflow:
    def test_ingest_new_file(self, pipeline: IngestionPipeline, tmp_path: Path) -> None:
        doc_path = tmp_path / "guide.md"
        doc_path.write_text(
            "# Introduction\n\nWelcome to the system.\n\n# Details\n\n" + "detail " * 50,
            encoding="utf-8",
        )

        result = pipeline.ingest_file(doc_path)

        assert result.status == "added"
        assert result.parent_count >= 1
        assert result.child_count >= 1
        assert pipeline.parent_store.count() == result.parent_count
        assert pipeline.vector_store_manager.count() == result.child_count

        # Check sidecar hash file exists
        hash_file = pipeline.settings.markdown_dir / "guide.md.sha256"
        assert hash_file.is_file()
        assert hash_file.read_text(encoding="utf-8").strip() == file_sha256(doc_path)

    def test_ingest_unchanged_file_is_skipped(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        doc_path = tmp_path / "notes.md"
        doc_path.write_text("# Notes\n\nSome important notes here.", encoding="utf-8")

        first = pipeline.ingest_file(doc_path)
        assert first.status == "added"
        initial_points = pipeline.vector_store_manager.count()

        # Second ingestion without changes should skip
        second = pipeline.ingest_file(doc_path)
        assert second.status == "skipped"
        assert pipeline.vector_store_manager.count() == initial_points

    def test_ingest_modified_file_purges_stale_chunks(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        doc_path = tmp_path / "doc.md"
        doc_path.write_text("# V1\n\n" + "first version " * 30, encoding="utf-8")
        res1 = pipeline.ingest_file(doc_path)
        assert res1.status == "added"

        # Overwrite with V2 content
        doc_path.write_text("# V2\n\n" + "second version " * 30, encoding="utf-8")
        res2 = pipeline.ingest_file(doc_path)
        assert res2.status == "added"

        # Confirm old content is not present in parent store
        for pid in pipeline.parent_store.list_parent_ids():
            content = pipeline.parent_store.load_content(pid)
            assert content is not None
            assert "first version" not in content
            assert "second version" in content

    def test_force_flag_reindexes(self, pipeline: IngestionPipeline, tmp_path: Path) -> None:
        doc_path = tmp_path / "doc.md"
        doc_path.write_text("# Title\n\nUnchanged text.", encoding="utf-8")

        pipeline.ingest_file(doc_path)
        # Even though unchanged, force=True re-indexes
        res = pipeline.ingest_file(doc_path, force=True)
        assert res.status == "added"

    def test_ingest_nonexistent_file(self, pipeline: IngestionPipeline, tmp_path: Path) -> None:
        bogus = tmp_path / "does_not_exist.md"
        res = pipeline.ingest_file(bogus)
        assert res.status == "failed"
        assert "not found" in res.message


class TestRollbackAndCleanup:
    def test_rollback_on_indexing_failure(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        doc_path = tmp_path / "fail.md"
        doc_path.write_text("# Fail\n\nSome text.", encoding="utf-8")

        # Simulate exception during vector indexing
        with patch.object(
            pipeline.vector_store_manager,
            "get_vector_store",
            side_effect=RuntimeError("Qdrant error"),
        ):
            res = pipeline.ingest_file(doc_path)

        assert res.status == "failed"
        assert "Qdrant error" in str(res.error)

        # Verify rollback: no parents, no vectors, no hash sidecar left behind
        assert pipeline.parent_store.count() == 0
        assert pipeline.vector_store_manager.count() == 0
        hash_file = pipeline.settings.markdown_dir / "fail.md.sha256"
        assert not hash_file.exists()

    def test_list_documents(self, pipeline: IngestionPipeline, tmp_path: Path) -> None:
        d1 = tmp_path / "doc1.md"
        d1.write_text("# Doc 1\n\nText 1", encoding="utf-8")
        d2 = tmp_path / "doc2.md"
        d2.write_text("# Doc 2\n\nText 2", encoding="utf-8")

        pipeline.ingest_files([d1, d2])
        docs = pipeline.list_documents()

        assert len(docs) == 2
        sources = {d["source"] for d in docs}
        assert sources == {"doc1.md", "doc2.md"}

    def test_delete_source(self, pipeline: IngestionPipeline, tmp_path: Path) -> None:
        d1 = tmp_path / "doc1.md"
        d1.write_text("# Doc 1\n\nText 1", encoding="utf-8")
        pipeline.ingest_file(d1)
        assert pipeline.parent_store.count() > 0

        pipeline.delete_source("doc1.md")
        assert pipeline.parent_store.count() == 0
        assert pipeline.vector_store_manager.count() == 0

    def test_clear_all(self, pipeline: IngestionPipeline, tmp_path: Path) -> None:
        d1 = tmp_path / "doc1.md"
        d1.write_text("# Doc 1\n\nText 1", encoding="utf-8")
        pipeline.ingest_file(d1)

        pipeline.clear_all()
        assert pipeline.parent_store.count() == 0
        assert pipeline.vector_store_manager.count() == 0
        assert len(list(pipeline.settings.markdown_dir.glob("*"))) == 0
