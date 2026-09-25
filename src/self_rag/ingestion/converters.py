"""Convert source documents to Markdown and compute content hashes.

Supported formats
-----------------
* **PDF** — converted via *pymupdf4llm* with images stripped and page separators preserved.
* **Markdown** — passed through as-is (the file is simply copied into the output directory).

Every source file is also SHA-256 hashed so that later stages can detect content changes without
re-parsing.

Design notes
~~~~~~~~~~~~
* ``pymupdf4llm.to_markdown`` accepts a *glob pattern* internally, but paths containing ``[``,
  ``]``, ``*``, or ``?`` silently match nothing.  Single files are therefore opened directly via
  ``pymupdf.open`` to avoid that trap.
* Surrogate bytes occasionally appear in PDF text extraction; the encode/decode round-trip with
  ``errors='surrogatepass'`` / ``errors='ignore'`` strips them safely.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pymupdf
import pymupdf4llm


def file_sha256(path: Path) -> str:
    """Return the hex-encoded SHA-256 digest of the file at *path*."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdf_to_markdown(pdf_path: Path, output_dir: Path) -> Path:
    """Convert a single PDF to Markdown and write it into *output_dir*.

    Returns the path to the written ``.md`` file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path))  # type: ignore[no-untyped-call]
    md_text: str = pymupdf4llm.to_markdown(
        doc,
        header=False,
        footer=False,
        page_separators=True,
        ignore_images=True,
        write_images=False,
        image_path=None,
    )
    # Strip surrogate characters that occasionally leak through PDF text extraction.
    md_cleaned = md_text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")

    md_path = (output_dir / pdf_path.stem).with_suffix(".md")
    md_path.write_bytes(md_cleaned.encode("utf-8"))
    return md_path


def markdown_passthrough(md_source: Path, output_dir: Path) -> Path:
    """Copy a Markdown source file into *output_dir* unchanged.

    Returns the path to the copy.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = (output_dir / md_source.stem).with_suffix(".md")
    shutil.copy2(md_source, dest)
    return dest


def convert_file(source: Path, output_dir: Path) -> Path:
    """Convert *source* (PDF or Markdown) into the canonical markdown directory.

    Returns the path to the resulting ``.md`` file.

    Raises ``ValueError`` for unsupported file extensions.
    """
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return pdf_to_markdown(source, output_dir)
    if suffix == ".md":
        return markdown_passthrough(source, output_dir)
    raise ValueError(f"Unsupported file type: {suffix!r} (expected .pdf or .md)")
