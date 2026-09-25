"""Tests for self_rag.ingestion.converters — conversion and hashing."""

from __future__ import annotations

from pathlib import Path

import pytest

from self_rag.ingestion.converters import convert_file, file_sha256, markdown_passthrough


class TestFileSha256:
    def test_known_digest(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("hello\n")
        # SHA-256 of b"hello\n" is well-known.
        assert file_sha256(f) == (
            "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
        )

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("alpha")
        b.write_text("bravo")
        assert file_sha256(a) != file_sha256(b)

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("identical")
        b.write_text("identical")
        assert file_sha256(a) == file_sha256(b)


class TestMarkdownPassthrough:
    def test_copies_file_unchanged(self, tmp_path: Path) -> None:
        source = tmp_path / "source.md"
        out_dir = tmp_path / "output"
        source.write_text("# Title\n\nBody text.")

        result = markdown_passthrough(source, out_dir)

        assert result.exists()
        assert result.suffix == ".md"
        assert result.read_text() == "# Title\n\nBody text."

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        source = tmp_path / "source.md"
        out_dir = tmp_path / "nested" / "deep"
        source.write_text("content")

        result = markdown_passthrough(source, out_dir)

        assert result.exists()
        assert out_dir.is_dir()


class TestConvertFile:
    def test_markdown_is_accepted(self, tmp_path: Path) -> None:
        source = tmp_path / "doc.md"
        out_dir = tmp_path / "out"
        source.write_text("# Hello\n\nWorld.")

        result = convert_file(source, out_dir)

        assert result.suffix == ".md"
        assert result.read_text() == "# Hello\n\nWorld."

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        source = tmp_path / "data.csv"
        source.write_text("a,b,c")

        with pytest.raises(ValueError, match="Unsupported file type"):
            convert_file(source, tmp_path / "out")
