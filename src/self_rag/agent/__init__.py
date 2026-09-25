"""Agent package containing state, prompts, schemas, and tools."""

from __future__ import annotations

from self_rag.agent.edges import route_after_orchestrator_call, route_after_rewrite
from self_rag.agent.graph import (
    ask_question,
    build_agent_subgraph,
    build_main_graph,
    create_sqlite_checkpointer,
)
from self_rag.agent.nodes import (
    aggregate_answers,
    collect_answer,
    compress_context,
    fallback_response,
    orchestrator,
    request_clarification,
    rewrite_query,
    should_compress_context,
    summarize_history,
)
from self_rag.agent.prompts import (
    get_aggregation_prompt,
    get_context_compression_prompt,
    get_conversation_summary_prompt,
    get_fallback_response_prompt,
    get_orchestrator_prompt,
    get_rewrite_query_prompt,
)
from self_rag.agent.schemas import (
    GradeAnswer,
    GradeDocuments,
    QueryAnalysis,
    SearchResultItem,
)
from self_rag.agent.state import (
    AgentState,
    State,
    accumulate_or_reset,
    append_unique,
    set_union,
)
from self_rag.agent.tools import (
    NO_PARENT_DOCUMENT,
    NO_RELEVANT_CHUNKS,
    PARENT_RETRIEVAL_ERROR_PREFIX,
    RETRIEVAL_ERROR_PREFIX,
    SENTINEL_PREFIXES,
    ToolFactory,
    _retrieval_contexts,
    execute_retrieve_parent_chunks,
    execute_search_child_chunks,
    format_retrieval_contexts,
)

__all__ = [
    "NO_PARENT_DOCUMENT",
    "NO_RELEVANT_CHUNKS",
    "PARENT_RETRIEVAL_ERROR_PREFIX",
    "RETRIEVAL_ERROR_PREFIX",
    "SENTINEL_PREFIXES",
    "AgentState",
    "GradeAnswer",
    "GradeDocuments",
    "QueryAnalysis",
    "SearchResultItem",
    "State",
    "ToolFactory",
    "_retrieval_contexts",
    "accumulate_or_reset",
    "aggregate_answers",
    "append_unique",
    "ask_question",
    "build_agent_subgraph",
    "build_main_graph",
    "collect_answer",
    "compress_context",
    "create_sqlite_checkpointer",
    "execute_retrieve_parent_chunks",
    "execute_search_child_chunks",
    "fallback_response",
    "format_retrieval_contexts",
    "get_aggregation_prompt",
    "get_context_compression_prompt",
    "get_conversation_summary_prompt",
    "get_fallback_response_prompt",
    "get_orchestrator_prompt",
    "get_rewrite_query_prompt",
    "orchestrator",
    "request_clarification",
    "rewrite_query",
    "route_after_orchestrator_call",
    "route_after_rewrite",
    "set_union",
    "should_compress_context",
    "summarize_history",
]
