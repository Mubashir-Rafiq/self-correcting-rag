"""Retrieval tools and context formatters for agent graphs.

Provides:
- Centralized sentinel constants (`NO_RELEVANT_CHUNKS`, `NO_PARENT_DOCUMENT`, error prefixes).
- Direct retrieval functions (`execute_search_child_chunks`, `execute_retrieve_parent_chunks`).
- `ToolFactory` producing LangChain `BaseTool` instances bound to storage.
- Context extraction helper `format_retrieval_contexts` (and alias `_retrieval_contexts`).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_qdrant import QdrantVectorStore

from self_rag.storage.parent_store import ParentStore
from self_rag.storage.reranker import BaseReranker

logger = logging.getLogger(__name__)

# --- Sentinels ----------------------------------------------------------------
NO_RELEVANT_CHUNKS: str = "NO_RELEVANT_CHUNKS"
NO_PARENT_DOCUMENT: str = "NO_PARENT_DOCUMENT"
RETRIEVAL_ERROR_PREFIX: str = "RETRIEVAL_ERROR:"
PARENT_RETRIEVAL_ERROR_PREFIX: str = "PARENT_RETRIEVAL_ERROR:"

SENTINEL_PREFIXES: tuple[str, ...] = (
    NO_RELEVANT_CHUNKS,
    NO_PARENT_DOCUMENT,
    RETRIEVAL_ERROR_PREFIX,
    PARENT_RETRIEVAL_ERROR_PREFIX,
)


# --- Core tool execution functions -------------------------------------------
def execute_search_child_chunks(
    collection: QdrantVectorStore,
    query: str,
    limit: int = 7,
    score_threshold: float | None = None,
    reranker: BaseReranker | None = None,
) -> str:
    """Execute hybrid search on child chunks and return JSON or a sentinel.

    Returns:
    - Sentinel ``"NO_RELEVANT_CHUNKS"`` if query is blank or no documents found.
    - JSON array of objects: ``[{"parent_id": "...", "source": "...", "content": "..."}, ...]``
    - Sentinel ``"RETRIEVAL_ERROR: <err>"`` on exception.
    """
    clean_query = query.strip()
    if not clean_query:
        return NO_RELEVANT_CHUNKS

    k = max(1, limit)
    try:
        search_k = max(k * 2, 10) if reranker is not None else k
        results = collection.similarity_search(
            query=clean_query,
            k=search_k,
            score_threshold=score_threshold,
        )
        if not results:
            return NO_RELEVANT_CHUNKS

        if reranker is not None:
            results = reranker.rerank(query=clean_query, documents=results, top_n=k)

        items: list[dict[str, str]] = []
        for doc in results:
            parent_id = str(doc.metadata.get("parent_id", ""))
            source = str(doc.metadata.get("source", ""))
            page = doc.metadata.get("page")
            item = {
                "parent_id": parent_id,
                "source": source,
                "content": doc.page_content,
            }
            if page is not None and str(page).strip():
                item["page"] = str(page).strip()
            items.append(item)
        return json.dumps(items, ensure_ascii=False)
    except Exception as err:
        logger.warning("Error during search_child_chunks: %s", err)
        return f"{RETRIEVAL_ERROR_PREFIX} {err}"


def execute_retrieve_parent_chunks(
    parent_store: ParentStore,
    parent_id: str,
) -> str:
    """Retrieve full parent chunk by ID and return JSON or a sentinel.

    Returns:
    - Sentinel ``"NO_PARENT_DOCUMENT"`` if ID is blank or chunk does not exist.
    - JSON object: ``{"parent_id": "...", "source": "...", "content": "..."}``
    - Sentinel ``"PARENT_RETRIEVAL_ERROR: <err>"`` on exception.
    """
    clean_id = parent_id.strip()
    if not clean_id:
        return NO_PARENT_DOCUMENT

    try:
        parent_doc = parent_store.load(clean_id)
        if parent_doc is None:
            return NO_PARENT_DOCUMENT

        source = str(parent_doc.metadata.get("source", ""))
        page = parent_doc.metadata.get("page")
        payload: dict[str, str] = {
            "parent_id": clean_id,
            "source": source,
            "content": parent_doc.page_content,
        }
        if page is not None and str(page).strip():
            payload["page"] = str(page).strip()
        return json.dumps(payload, ensure_ascii=False)
    except Exception as err:
        logger.warning("Error during retrieve_parent_chunks for '%s': %s", clean_id, err)
        return f"{PARENT_RETRIEVAL_ERROR_PREFIX} {err}"


# --- Context formatting -------------------------------------------------------
def format_retrieval_contexts(
    items: Sequence[BaseMessage | str | dict[str, Any]],
) -> list[str]:
    """Extract and format readable context strings from tool messages or outputs.

    Excludes any sentinel or error messages (matching ``SENTINEL_PREFIXES``).
    Parses JSON array or object payloads into readable multi-line context blocks:
        Parent ID: {parent_id}
        File Name: {source}
        Content: {content}

    Falls back to raw string content if JSON parsing fails.
    """
    contexts: list[str] = []

    for item in items:
        # Extract raw text depending on whether input is Message, dict, or str
        text: str
        if isinstance(item, BaseMessage):
            # Only process ToolMessage outputs or messages carrying tool content
            if not isinstance(item, ToolMessage):
                continue
            content = item.content
            text = content if isinstance(content, str) else json.dumps(content)
        elif isinstance(item, dict):
            pid = str(item.get("parent_id", ""))
            source = str(item.get("source", ""))
            page = item.get("page")
            file_label = f"{source}, p. {page}" if page else source
            content_str = str(item.get("content", ""))
            contexts.append(f"Parent ID: {pid}\nFile Name: {file_label}\nContent: {content_str}")
            continue
        else:
            text = item

        clean_text = text.strip()
        if not clean_text:
            continue

        # Skip recognized sentinels and errors
        if any(clean_text.startswith(prefix) for prefix in SENTINEL_PREFIXES):
            continue

        # Attempt JSON decoding
        try:
            parsed = json.loads(clean_text)
            if isinstance(parsed, list):
                for entry in parsed:
                    if isinstance(entry, dict):
                        pid = str(entry.get("parent_id", ""))
                        source = str(entry.get("source", ""))
                        page = entry.get("page")
                        file_label = f"{source}, p. {page}" if page else source
                        body = str(entry.get("content", ""))
                        contexts.append(
                            f"Parent ID: {pid}\nFile Name: {file_label}\nContent: {body}"
                        )
                    else:
                        contexts.append(str(entry))
            elif isinstance(parsed, dict):
                pid = str(parsed.get("parent_id", ""))
                source = str(parsed.get("source", ""))
                page = parsed.get("page")
                file_label = f"{source}, p. {page}" if page else source
                body = str(parsed.get("content", ""))
                contexts.append(f"Parent ID: {pid}\nFile Name: {file_label}\nContent: {body}")
            else:
                contexts.append(clean_text)
        except (json.JSONDecodeError, TypeError):
            # Raw string fallback
            contexts.append(clean_text)

    return contexts


# Downstream naming alias per blueprint specification
_retrieval_contexts = format_retrieval_contexts


# --- Tool Factory -------------------------------------------------------------
class ToolFactory:
    """Factory producing LangChain ``BaseTool`` instances bound to live stores."""

    def __init__(
        self,
        collection: QdrantVectorStore,
        parent_store: ParentStore,
        *,
        default_limit: int = 7,
        score_threshold: float | None = None,
        reranker: BaseReranker | None = None,
    ) -> None:
        self.collection = collection
        self.parent_store = parent_store
        self.default_limit = max(1, default_limit)
        self.score_threshold = score_threshold
        self.reranker = reranker

    def create_search_tool(self) -> BaseTool:
        """Create the ``search_child_chunks`` tool."""
        collection = self.collection
        default_limit = self.default_limit
        score_threshold = self.score_threshold
        reranker = self.reranker

        @tool
        def search_child_chunks(query: str, limit: int = default_limit) -> str:
            """Search for relevant child chunks in the document corpus using hybrid search.

            Returns a JSON array of matching chunks with parent_id, source, and content,
            or NO_RELEVANT_CHUNKS if none match.
            """
            return execute_search_child_chunks(
                collection=collection,
                query=query,
                limit=limit,
                score_threshold=score_threshold,
                reranker=reranker,
            )

        return search_child_chunks

    def create_retrieve_tool(self) -> BaseTool:
        """Create the ``retrieve_parent_chunks`` tool."""
        parent_store = self.parent_store

        @tool
        def retrieve_parent_chunks(parent_id: str) -> str:
            """Retrieve the full parent document text for a given parent_id.

            Returns a JSON object with parent_id, source, and content,
            or NO_PARENT_DOCUMENT if not found.
            """
            return execute_retrieve_parent_chunks(
                parent_store=parent_store,
                parent_id=parent_id,
            )

        return retrieve_parent_chunks

    def create_tools(self) -> list[BaseTool]:
        """Create both retrieval tools ready for binding to an LLM."""
        return [self.create_search_tool(), self.create_retrieve_tool()]
