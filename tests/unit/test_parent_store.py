"""Tests for self_rag.storage.parent_store — JSON parent chunk storage on disk."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from self_rag.storage.parent_store import ParentStore


@pytest.fixture()
def store(tmp_path: Path) -> ParentStore:
    """Return a fresh ParentStore pointing to a temporary directory."""
    return ParentStore(directory=tmp_path / "parents")


class TestParentStoreBasics:
    def test_save_and_load(self, store: ParentStore) -> None:
        doc = Document(
            page_content="This is the parent chunk text.",
            metadata={"source": "report.pdf", "parent_id": "report_p0", "H1": "Introduction"},
        )
        saved_path = store.save("report_p0", doc)

        assert saved_path.is_file()
        loaded = store.load("report_p0")
        assert loaded is not None
        assert loaded.page_content == "This is the parent chunk text."
        assert loaded.metadata["source"] == "report.pdf"
        assert loaded.metadata["H1"] == "Introduction"

    def test_load_nonexistent_returns_none(self, store: ParentStore) -> None:
        """Loading a chunk that does not exist must return None rather than raising."""
        assert store.load("nonexistent_id") is None
        assert store.load_content("nonexistent_id") is None

    def test_save_empty_id_raises(self, store: ParentStore) -> None:
        doc = Document(page_content="text", metadata={})
        with pytest.raises(ValueError, match="parent_id cannot be empty"):
            store.save("", doc)

    def test_save_many(self, store: ParentStore) -> None:
        docs = [
            ("doc_p0", Document(page_content="p0 text", metadata={"source": "doc.pdf"})),
            ("doc_p1", Document(page_content="p1 text", metadata={"source": "doc.pdf"})),
        ]
        paths = store.save_many(docs)

        assert len(paths) == 2
        assert store.count() == 2
        assert store.load_content("doc_p0") == "p0 text"
        assert store.load_content("doc_p1") == "p1 text"


class TestContentOrdering:
    def test_load_content_many_sorts_by_trailing_index(self, store: ParentStore) -> None:
        """Ensure IDs sort numerically (p1, p2, p10) rather than lexicographically."""
        store.save("doc_p10", Document(page_content="tenth", metadata={}))
        store.save("doc_p1", Document(page_content="first", metadata={}))
        store.save("doc_p2", Document(page_content="second", metadata={}))

        # Query in scrambled order
        contents = store.load_content_many(["doc_p10", "doc_p2", "doc_p1"])
        assert contents == ["first", "second", "tenth"]

    def test_load_content_many_skips_missing(self, store: ParentStore) -> None:
        store.save("doc_p0", Document(page_content="present", metadata={}))

        contents = store.load_content_many(["doc_p0", "doc_p99"])
        assert contents == ["present"]

    def test_list_parent_ids_natural_order(self, store: ParentStore) -> None:
        store.save("a_p2", Document(page_content="2", metadata={}))
        store.save("a_p10", Document(page_content="10", metadata={}))
        store.save("a_p1", Document(page_content="1", metadata={}))

        assert store.list_parent_ids() == ["a_p1", "a_p2", "a_p10"]


class TestDeletionAndPurge:
    def test_delete(self, store: ParentStore) -> None:
        store.save("doc_p0", Document(page_content="data", metadata={}))
        assert store.delete("doc_p0") is True
        assert store.delete("doc_p0") is False
        assert store.load("doc_p0") is None

    def test_delete_many(self, store: ParentStore) -> None:
        store.save("doc_p0", Document(page_content="0", metadata={}))
        store.save("doc_p1", Document(page_content="1", metadata={}))
        store.save("doc_p2", Document(page_content="2", metadata={}))

        deleted = store.delete_many(["doc_p0", "doc_p2", "doc_p99"])
        assert deleted == 2
        assert store.count() == 1
        assert store.load("doc_p1") is not None

    def test_delete_by_source_stem(self, store: ParentStore) -> None:
        store.save("manual_p0", Document(page_content="m0", metadata={"source": "manual.pdf"}))
        store.save("manual_p1", Document(page_content="m1", metadata={"source": "manual.pdf"}))
        store.save("other_p0", Document(page_content="o0", metadata={"source": "other.pdf"}))

        deleted = store.delete_by_source("manual.pdf")
        assert deleted == 2
        assert store.count() == 1
        assert store.load("other_p0") is not None

    def test_delete_by_source_metadata(self, store: ParentStore) -> None:
        # Non-standard filename stem, but metadata matches
        store.save("custom_id_1", Document(page_content="c1", metadata={"source": "guide.md"}))
        store.save("custom_id_2", Document(page_content="c2", metadata={"source": "other.md"}))

        deleted = store.delete_by_source("guide.md")
        assert deleted == 1
        assert store.count() == 1
        assert store.load("custom_id_2") is not None

    def test_clear(self, store: ParentStore) -> None:
        store.save("p0", Document(page_content="a", metadata={}))
        store.save("p1", Document(page_content="b", metadata={}))

        assert store.clear() == 2
        assert store.count() == 0


class TestSourcesAndCorruptFiles:
    def test_list_sources(self, store: ParentStore) -> None:
        store.save("a_p0", Document(page_content="a", metadata={"source": "zebra.pdf"}))
        store.save("a_p1", Document(page_content="a", metadata={"source": "zebra.pdf"}))
        store.save("b_p0", Document(page_content="b", metadata={"source": "apple.pdf"}))

        sources = store.list_sources()
        assert sources == ["apple.pdf", "zebra.pdf"]

    def test_corrupt_file_returns_none(self, store: ParentStore) -> None:
        corrupt_file = store.directory / "corrupt_p0.json"
        corrupt_file.write_text("NOT VALID JSON {{{", encoding="utf-8")

        assert store.load("corrupt_p0") is None
        assert store.load_content("corrupt_p0") is None
