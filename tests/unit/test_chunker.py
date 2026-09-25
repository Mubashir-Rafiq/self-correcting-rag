"""Tests for self_rag.ingestion.chunker — the parent/child algorithm.

All tests use synthetic Markdown so they run in CI without downloading models or reading real PDFs.
The fixture ``make_chunker`` builds a ``DocumentChunker`` with controllable size thresholds.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic_settings import SettingsConfigDict

from self_rag.config import Settings
from self_rag.ingestion.chunker import DocumentChunker

ChunkerFactory = Callable[..., tuple[DocumentChunker, Settings]]
MdFileFactory = Callable[..., Path]


class IsolatedSettings(Settings):
    """Settings that ignore a developer's real .env file, so tests stay reproducible."""

    model_config = SettingsConfigDict(env_file=None)


def _settings(**overrides: Any) -> Settings:
    return IsolatedSettings(**overrides)


@pytest.fixture()
def make_chunker() -> ChunkerFactory:
    """Factory fixture: returns a ``(chunker, settings)`` pair with merged overrides."""

    def _factory(**overrides: Any) -> tuple[DocumentChunker, Settings]:
        s = _settings(**overrides)
        return DocumentChunker(s), s

    return _factory


@pytest.fixture()
def md_file(tmp_path: Path) -> MdFileFactory:
    """Factory fixture: writes Markdown text into a temp file and returns its path."""

    def _factory(text: str, name: str = "doc") -> Path:
        p = tmp_path / f"{name}.md"
        p.write_text(text, encoding="utf-8")
        return p

    return _factory


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


class TestBasicChunking:
    def test_single_short_section_becomes_one_parent(
        self, make_chunker: ChunkerFactory, md_file: MdFileFactory
    ) -> None:
        chunker, _ = make_chunker(
            min_parent_size=10, max_parent_size=5000, child_chunk_size=50, child_chunk_overlap=10
        )
        path = md_file("# Intro\n\nThis is a short document with enough text to pass the min.")

        parents, children = chunker.chunk_file(path)

        assert len(parents) >= 1
        assert len(children) >= 1

    def test_parent_ids_are_sequential(
        self, make_chunker: ChunkerFactory, md_file: MdFileFactory
    ) -> None:
        text = "# Section A\n\n" + "word " * 500 + "\n\n# Section B\n\n" + "word " * 500
        chunker, _ = make_chunker(min_parent_size=100, max_parent_size=3000, child_chunk_size=200)
        path = md_file(text)

        parents, _ = chunker.chunk_file(path)

        ids = [pid for pid, _ in parents]
        for i, pid in enumerate(ids):
            assert pid == f"doc_p{i}"

    def test_source_defaults_to_stem_dot_pdf(
        self, make_chunker: ChunkerFactory, md_file: MdFileFactory
    ) -> None:
        chunker, _ = make_chunker(
            min_parent_size=10, max_parent_size=5000, child_chunk_size=50, child_chunk_overlap=10
        )
        path = md_file("# Doc\n\nSome content here to chunk.")

        parents, children = chunker.chunk_file(path)

        _, doc = parents[0]
        assert doc.metadata["source"] == "doc.pdf"
        assert children[0].metadata["source"] == "doc.pdf"

    def test_custom_source_name(self, make_chunker: ChunkerFactory, md_file: MdFileFactory) -> None:
        chunker, _ = make_chunker(
            min_parent_size=10, max_parent_size=5000, child_chunk_size=50, child_chunk_overlap=10
        )
        path = md_file("# Doc\n\nSome content here.")

        parents, _ = chunker.chunk_file(path, source_name="notes.md")

        _, doc = parents[0]
        assert doc.metadata["source"] == "notes.md"


# ---------------------------------------------------------------------------
# Merging small parents
# ---------------------------------------------------------------------------


class TestMergeSmall:
    def test_tiny_sections_are_merged(
        self, make_chunker: ChunkerFactory, md_file: MdFileFactory
    ) -> None:
        """Three tiny sections (each well below min_parent_size) should merge into one parent."""
        text = "# A\n\nshort\n\n# B\n\nalso short\n\n# C\n\nstill short"
        chunker, _ = make_chunker(
            min_parent_size=200,
            max_parent_size=5000,
            child_chunk_size=50,
            child_chunk_overlap=10,
        )
        path = md_file(text)

        parents, _ = chunker.chunk_file(path)

        # All three should have been merged into a single parent.
        assert len(parents) == 1
        content = parents[0][1].page_content
        assert "short" in content
        assert "also short" in content
        assert "still short" in content


# ---------------------------------------------------------------------------
# Splitting large parents
# ---------------------------------------------------------------------------


class TestSplitLarge:
    def test_oversized_section_is_split(
        self, make_chunker: ChunkerFactory, md_file: MdFileFactory
    ) -> None:
        """A single section larger than max_parent_size must be split."""
        # 2000 chars, with max_parent_size=500 → should produce multiple parents.
        text = "# Big\n\n" + "a " * 1000
        chunker, settings = make_chunker(
            min_parent_size=100,
            max_parent_size=500,
            child_chunk_size=100,
            child_chunk_overlap=20,
        )
        path = md_file(text)

        parents, _ = chunker.chunk_file(path)

        assert len(parents) > 1
        for _, doc in parents:
            assert len(doc.page_content) <= settings.max_parent_size


# ---------------------------------------------------------------------------
# Children
# ---------------------------------------------------------------------------


class TestChildChunks:
    def test_children_carry_parent_id(
        self, make_chunker: ChunkerFactory, md_file: MdFileFactory
    ) -> None:
        text = "# Section\n\n" + "word " * 300
        chunker, _ = make_chunker(
            min_parent_size=100,
            max_parent_size=5000,
            child_chunk_size=100,
            child_chunk_overlap=20,
        )
        path = md_file(text)

        parents, children = chunker.chunk_file(path)

        parent_ids = {pid for pid, _ in parents}
        for child in children:
            assert "parent_id" in child.metadata
            assert child.metadata["parent_id"] in parent_ids

    def test_child_sizes_respect_config(
        self, make_chunker: ChunkerFactory, md_file: MdFileFactory
    ) -> None:
        text = "# Section\n\n" + "word " * 500
        chunk_size = 200
        chunker, _ = make_chunker(
            min_parent_size=100,
            max_parent_size=5000,
            child_chunk_size=chunk_size,
            child_chunk_overlap=40,
        )
        path = md_file(text)

        _, children = chunker.chunk_file(path)

        # RecursiveCharacterTextSplitter may produce chunks slightly over chunk_size
        # when it can't find a good split point, but they should be in the right ballpark.
        assert len(children) > 1
        for child in children:
            # Allow a generous margin — the splitter doesn't guarantee exact sizes.
            assert len(child.page_content) <= chunk_size * 2


# ---------------------------------------------------------------------------
# Metadata merging
# ---------------------------------------------------------------------------


class TestMetadataMerging:
    def test_merge_creates_arrow_chain(self) -> None:
        target: dict[str, str] = {"H1": "Introduction"}
        source: dict[str, str] = {"H1": "Background"}
        DocumentChunker._merge_metadata(target, source)
        assert target["H1"] == "Introduction -> Background"

    def test_merge_deduplicates(self) -> None:
        target: dict[str, str] = {"H1": "Intro"}
        source: dict[str, str] = {"H1": "Intro"}
        DocumentChunker._merge_metadata(target, source)
        assert target["H1"] == "Intro"

    def test_merge_new_key(self) -> None:
        target: dict[str, str] = {"H1": "Title"}
        source: dict[str, str] = {"H2": "Subtitle"}
        DocumentChunker._merge_metadata(target, source)
        assert target["H1"] == "Title"
        assert target["H2"] == "Subtitle"

    def test_merge_prepend(self) -> None:
        target: dict[str, str] = {"H1": "Second"}
        source: dict[str, str] = {"H1": "First"}
        DocumentChunker._merge_metadata(target, source, prepend=True)
        assert target["H1"] == "First -> Second"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_document(self, make_chunker: ChunkerFactory, md_file: MdFileFactory) -> None:
        chunker, _ = make_chunker(
            min_parent_size=10, max_parent_size=5000, child_chunk_size=50, child_chunk_overlap=10
        )
        path = md_file("")

        parents, children = chunker.chunk_file(path)

        # An empty document produces no meaningful chunks.
        # The header splitter may or may not emit an empty Document.
        assert isinstance(parents, list)
        assert isinstance(children, list)

    def test_no_headers(self, make_chunker: ChunkerFactory, md_file: MdFileFactory) -> None:
        """A document with no Markdown headers should still produce chunks."""
        text = "Just a plain paragraph with no headers. " * 50
        chunker, _ = make_chunker(
            min_parent_size=100,
            max_parent_size=5000,
            child_chunk_size=200,
            child_chunk_overlap=40,
        )
        path = md_file(text)

        parents, children = chunker.chunk_file(path)

        assert len(parents) >= 1
        assert len(children) >= 1
