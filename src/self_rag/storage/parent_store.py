"""On-disk JSON storage for full-text parent chunks.

Parent chunks are stored as individual JSON files on disk in ``data_dir / "parent_store"``:
    ``{data_dir}/parent_store/{parent_id}.json``

Each JSON file stores the chunk's text and metadata:
    {"page_content": "...", "metadata": {"source": "manual.pdf", "parent_id": "manual_p0", ...}}

Retrieving a nonexistent chunk returns ``None`` rather than raising, matching the contract
specified in the architecture plan.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from self_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _parent_sort_key(parent_id: str) -> tuple[str, int, str]:
    """Sort key that orders parent IDs naturally by prefix and trailing numeric index.

    For example, ``'doc_p2'`` sorts before ``'doc_p10'``.
    """
    match = re.search(r"^(.*)_p(\d+)$", parent_id)
    if match:
        return (match.group(1), int(match.group(2)), "")
    return (parent_id, -1, parent_id)


class ParentStore:
    """Manages full-text parent chunks as JSON files on disk."""

    def __init__(
        self,
        directory: Path | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        if directory is not None:
            self.directory = Path(directory)
        elif settings is not None:
            self.directory = settings.parent_store_dir
        else:
            self.directory = get_settings().parent_store_dir

        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, parent_id: str) -> Path:
        """Return the filesystem path for a given *parent_id*."""
        return self.directory / f"{parent_id}.json"

    def save(self, parent_id: str, document: Document) -> Path:
        """Save a single parent *document* under *parent_id*.

        Overwrites any existing file for this ID. Returns the path of the saved file.
        """
        if not parent_id.strip():
            raise ValueError("parent_id cannot be empty")

        path = self.path_for(parent_id)
        payload: dict[str, Any] = {
            "page_content": document.page_content,
            "metadata": document.metadata,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def save_many(self, items: Sequence[tuple[str, Document]]) -> list[Path]:
        """Save multiple ``(parent_id, document)`` pairs to disk."""
        return [self.save(parent_id, doc) for parent_id, doc in items]

    def load(self, parent_id: str) -> Document | None:
        """Load a parent chunk as a ``Document``.

        Returns ``None`` if the chunk does not exist on disk rather than raising.
        """
        path = self.path_for(parent_id)
        if not path.is_file():
            return None

        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                logger.warning("Invalid JSON structure in parent file: %s", path)
                return None
            content = data.get("page_content", "")
            metadata = data.get("metadata", {})
            meta_dict = metadata if isinstance(metadata, dict) else {}
            return Document(page_content=content, metadata=meta_dict)
        except (json.JSONDecodeError, OSError) as err:
            logger.warning("Failed to read parent file %s: %s", path, err)
            return None

    def load_content(self, parent_id: str) -> str | None:
        """Load just the text content for *parent_id*, or ``None`` if not found."""
        doc = self.load(parent_id)
        return doc.page_content if doc is not None else None

    def load_content_many(self, parent_ids: Sequence[str]) -> list[str]:
        """Load content for multiple parent IDs, sorted by document stem and numeric index.

        Any missing parent IDs are silently omitted from the returned list.
        """
        sorted_ids = sorted(parent_ids, key=_parent_sort_key)
        contents: list[str] = []
        for pid in sorted_ids:
            text = self.load_content(pid)
            if text is not None:
                contents.append(text)
        return contents

    def delete(self, parent_id: str) -> bool:
        """Delete a single parent chunk. Returns True if deleted, False if not found."""
        path = self.path_for(parent_id)
        if path.is_file():
            path.unlink()
            return True
        return False

    def delete_many(self, parent_ids: Sequence[str]) -> int:
        """Delete multiple parent chunks by ID. Returns the count of deleted files."""
        deleted = 0
        for pid in parent_ids:
            if self.delete(pid):
                deleted += 1
        return deleted

    def delete_by_source(self, source_name: str) -> int:
        """Delete all parent chunks belonging to *source_name*.

        Matches against ``metadata['source']``, as well as documents whose filename
        stem matches *source_name* (e.g. 'manual.pdf' matches 'manual_p*.json').

        Returns the count of deleted files.
        """
        source_stem = Path(source_name).stem
        deleted = 0

        for path in list(self.directory.glob("*.json")):
            should_delete = False
            # Quick check: stem-based match (e.g., manual_p0.json matches manual.pdf)
            name_stem = path.stem
            if re.match(rf"^{re.escape(source_stem)}_p\d+$", name_stem) or re.match(
                rf"^{re.escape(source_name)}_p\d+$", name_stem
            ):
                should_delete = True
            else:
                # Content metadata inspection
                doc = self.load(path.stem)
                if doc is not None and doc.metadata.get("source") == source_name:
                    should_delete = True

            if should_delete and self.delete(path.stem):
                deleted += 1

        return deleted

    def list_parent_ids(self) -> list[str]:
        """Return all parent IDs currently in the store, sorted by natural order."""
        ids = [p.stem for p in self.directory.glob("*.json") if p.is_file()]
        return sorted(ids, key=_parent_sort_key)

    def list_sources(self) -> list[str]:
        """Return a sorted list of unique document sources currently in the parent store."""
        sources: set[str] = set()
        for path in self.directory.glob("*.json"):
            doc = self.load(path.stem)
            if doc is not None:
                src = doc.metadata.get("source")
                if isinstance(src, str) and src.strip():
                    sources.add(src.strip())
        return sorted(sources)

    def count(self) -> int:
        """Return the number of parent chunk files currently stored."""
        return len(list(self.directory.glob("*.json")))

    def clear(self) -> int:
        """Delete all parent chunks in the store. Returns the number of deleted files."""
        deleted = 0
        for path in list(self.directory.glob("*.json")):
            if path.is_file():
                path.unlink()
                deleted += 1
        return deleted
