"""Hierarchical parent/child chunking algorithm.

Pipeline order
--------------
1. **Header split** — ``MarkdownHeaderTextSplitter`` breaks on H1/H2/H3 boundaries.
2. **Merge small** — greedily concatenate consecutive header sections until each accumulated
   block reaches ``min_parent_size``.
3. **Split large** — anything still over ``max_parent_size`` is recursively character-split.
4. **Rebalance / clean** — undersized leftovers are absorbed into a neighbour or rebalanced
   across a boundary at a clean separator (paragraph > line > space).
5. **Child split** — each final parent is cut into fixed-size children for vector indexing.

This reproduces the reference implementation's algorithm *exactly* (see ``BLUEPRINT.md`` §7.2)
because the chunking boundaries are load-bearing: changing them would silently alter retrieval
quality without any test catching it.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from self_rag.config import Settings

PAGE_MARKER_PATTERN = re.compile(
    r"(?:---\s*end of page\.page_number=(\d+)\s*---|<!--\s*[Pp]age\s*(\d+)\s*-->)"
)

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

# A parent "chunk" is a ``(parent_id, Document)`` pair so the caller can persist
# parents by id without needing to know how ids are assigned.
type ParentPair = tuple[str, Document]


class DocumentChunker:
    """Split a Markdown document into hierarchical parent and child chunks.

    All size thresholds and splitter settings are drawn from the injected ``Settings``
    so that the chunker is fully testable with overridden values.
    """

    def __init__(self, settings: Settings) -> None:
        self._min_parent_size = settings.min_parent_size
        self._max_parent_size = settings.max_parent_size
        self._child_overlap = settings.child_chunk_overlap

        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=list(settings.markdown_headers),
            strip_headers=False,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.child_chunk_size,
            chunk_overlap=settings.child_chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_file(
        self,
        md_path: Path,
        source_name: str | None = None,
    ) -> tuple[list[ParentPair], list[Document]]:
        """Chunk a single Markdown file into parent/child pairs.

        Parameters
        ----------
        md_path:
            Path to a ``.md`` file (already converted from PDF if necessary).
        source_name:
            Human-readable origin for the ``source`` metadata field.  Defaults to
            ``<stem>.pdf`` to match the convention that most documents start as PDFs.

        Returns
        -------
        A ``(parents, children)`` tuple where *parents* is a list of
        ``(parent_id, Document)`` pairs and *children* is a flat list of child
        ``Document`` objects carrying inherited metadata (including ``parent_id``
        and ``source``).
        """
        source_name = source_name or f"{md_path.stem}.pdf"
        text = md_path.read_text(encoding="utf-8")

        header_chunks = self._header_splitter.split_text(text)
        merged = self._merge_small_parents(header_chunks)
        split = self._split_large_parents(merged)
        cleaned = self._clean_small_chunks(split)

        if any(len(c.page_content) > self._max_parent_size for c in cleaned):
            raise ValueError("Parent chunking produced a chunk larger than max_parent_size.")

        parents: list[ParentPair] = []
        children: list[Document] = []
        intervals = self._compute_page_intervals(text)
        self._create_child_chunks(
            parents,
            children,
            cleaned,
            md_path,
            source_name,
            full_text=text,
            page_intervals=intervals,
        )
        return parents, children

    # ------------------------------------------------------------------
    # Internal: merge / split / rebalance
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_metadata(
        target: dict[str, str],
        source: dict[str, str],
        *,
        prepend: bool = False,
    ) -> None:
        """Merge *source* metadata into *target*, joining on ``" -> "``."""
        for key, value in source.items():
            if key not in target:
                target[key] = value
            else:
                first, second = (value, target[key]) if prepend else (target[key], value)
                values = [
                    item.strip()
                    for raw in (first, second)
                    for item in str(raw).split(" -> ")
                    if item.strip()
                ]
                target[key] = " -> ".join(dict.fromkeys(values))

    def _merge_small_parents(self, chunks: list[Document]) -> list[Document]:
        """Greedily concatenate consecutive header-split chunks until *min_parent_size*."""
        if not chunks:
            return []

        merged: list[Document] = []
        current: Document | None = None

        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content
                self._merge_metadata(current.metadata, chunk.metadata)

            if len(current.page_content) >= self._min_parent_size:
                merged.append(current)
                current = None

        if current is not None:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                self._merge_metadata(merged[-1].metadata, current.metadata)
            else:
                merged.append(current)

        return merged

    def _split_large_parents(self, chunks: list[Document]) -> list[Document]:
        """Character-split anything still over *max_parent_size*."""
        result: list[Document] = []
        for chunk in chunks:
            if len(chunk.page_content) <= self._max_parent_size:
                result.append(chunk)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self._max_parent_size,
                    chunk_overlap=self._child_overlap,
                )
                result.extend(splitter.split_documents([chunk]))
        return result

    def _rebalance_pair(self, first: Document, second: Document) -> tuple[Document, Document]:
        """Re-split the boundary between two neighbours at a cleaner separator."""
        combined = first.page_content.rstrip() + "\n\n" + second.page_content.lstrip()

        lower = max(1, len(combined) - self._max_parent_size)
        upper = min(self._max_parent_size, len(combined) - 1)

        if len(combined) >= 2 * self._min_parent_size:
            lower = max(lower, self._min_parent_size)
            upper = min(upper, len(combined) - self._min_parent_size)

        preferred = min(max(len(combined) // 2, lower), upper)

        split_at = preferred
        for separator in ("\n\n", "\n", " "):
            before = combined.rfind(separator, lower, preferred + 1)
            after = combined.find(separator, preferred, upper + 1)
            if before >= lower:
                split_at = before
                break
            if after != -1:
                split_at = after
                break

        left_text = combined[:split_at].rstrip()
        right_text = combined[split_at:].lstrip()

        if len(combined) >= 2 * self._min_parent_size and (
            len(left_text) < self._min_parent_size or len(right_text) < self._min_parent_size
        ):
            split_at = preferred
            left_text, right_text = combined[:split_at], combined[split_at:]

        if not left_text or not right_text:
            return first, second

        metadata: dict[str, str] = dict(first.metadata)
        self._merge_metadata(metadata, second.metadata)
        first.page_content, first.metadata = left_text, dict(metadata)
        second.page_content, second.metadata = right_text, dict(metadata)
        return first, second

    def _clean_small_chunks(self, chunks: list[Document]) -> list[Document]:
        """Absorb or rebalance undersized chunks."""
        # Pass 1: absorb into a neighbour if it fits.
        cleaned: list[Document] = []
        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < self._min_parent_size:
                if (
                    cleaned
                    and len(cleaned[-1].page_content) + 2 + len(chunk.page_content)
                    <= self._max_parent_size
                ):
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    self._merge_metadata(cleaned[-1].metadata, chunk.metadata)
                elif (
                    i < len(chunks) - 1
                    and len(chunk.page_content) + 2 + len(chunks[i + 1].page_content)
                    <= self._max_parent_size
                ):
                    chunks[i + 1].page_content = (
                        chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    )
                    self._merge_metadata(chunks[i + 1].metadata, chunk.metadata, prepend=True)
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)

        # Pass 2: rebalance anything still undersized.
        for i, chunk in enumerate(cleaned):
            if len(chunk.page_content) >= self._min_parent_size or len(cleaned) == 1:
                continue
            if i < len(cleaned) - 1:
                cleaned[i], cleaned[i + 1] = self._rebalance_pair(chunk, cleaned[i + 1])
            else:
                cleaned[i - 1], cleaned[i] = self._rebalance_pair(cleaned[i - 1], chunk)

        return cleaned

    # ------------------------------------------------------------------
    # Internal: page detection & child generation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_page_intervals(text: str) -> list[tuple[int, int, int]]:
        """Compute (start_char, end_char, page_number) spans from page markers."""
        matches = list(PAGE_MARKER_PATTERN.finditer(text))
        if not matches:
            return []
        intervals: list[tuple[int, int, int]] = []
        curr = 0
        last_page = 1
        for m in matches:
            p_str = m.group(1) or m.group(2)
            p_num = int(p_str)
            intervals.append((curr, m.end(), p_num))
            curr = m.end()
            last_page = p_num
        if curr < len(text):
            intervals.append((curr, len(text), last_page + 1))
        return intervals

    @staticmethod
    def _get_page_for_offset(
        start: int,
        end: int,
        intervals: list[tuple[int, int, int]],
    ) -> str | None:
        """Find the page number or range for [start, end] char offsets."""
        pages: list[int] = []
        for p_start, p_end, p_num in intervals:
            if max(start, p_start) < min(end, p_end):
                pages.append(p_num)
        if not pages:
            return None
        if len(pages) == 1:
            return str(pages[0])
        return f"{pages[0]}-{pages[-1]}"

    @staticmethod
    def _find_content_offset(
        content: str,
        full_text: str,
        start_hint: int = 0,
    ) -> int:
        """Find the start offset of *content* within *full_text* resilient to header formatting."""
        content_strip = content.strip()
        if not content_strip:
            return -1

        probe = content_strip[:60]
        pos = full_text.find(probe, start_hint)
        if pos != -1:
            return pos
        pos = full_text.find(probe)
        if pos != -1:
            return pos

        for line in content.splitlines():
            line_str = line.strip()
            if len(line_str) >= 10:
                pos = full_text.find(line_str, start_hint)
                if pos != -1:
                    return pos
                pos = full_text.find(line_str)
                if pos != -1:
                    return pos
        return -1

    def _create_child_chunks(
        self,
        all_parents: list[ParentPair],
        all_children: list[Document],
        parent_chunks: list[Document],
        doc_path: Path,
        source_name: str,
        full_text: str = "",
        page_intervals: list[tuple[int, int, int]] | None = None,
    ) -> None:
        last_found_offset = 0
        for i, p_chunk in enumerate(parent_chunks):
            parent_id = f"{doc_path.stem}_p{i}"
            p_chunk.metadata.update({"source": source_name, "parent_id": parent_id})

            if page_intervals and full_text:
                pos = self._find_content_offset(
                    p_chunk.page_content, full_text, start_hint=last_found_offset
                )
                if pos != -1:
                    last_found_offset = pos
                    page_label = self._get_page_for_offset(
                        pos, pos + len(p_chunk.page_content.strip()), page_intervals
                    )
                    if page_label:
                        p_chunk.metadata["page"] = page_label

            all_parents.append((parent_id, p_chunk))
            child_docs = self._child_splitter.split_documents([p_chunk])
            if page_intervals and full_text and "page" in p_chunk.metadata:
                parent_page = str(p_chunk.metadata["page"])
                for child in child_docs:
                    c_pos = self._find_content_offset(
                        child.page_content, full_text, start_hint=last_found_offset
                    )
                    if c_pos != -1:
                        c_page = self._get_page_for_offset(
                            c_pos, c_pos + len(child.page_content.strip()), page_intervals
                        )
                        if c_page:
                            child.metadata["page"] = c_page
                    else:
                        child.metadata["page"] = parent_page

            all_children.extend(child_docs)
