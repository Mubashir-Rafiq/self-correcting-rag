"""Graph assembly and execution for agent subgraphs and loops."""

from __future__ import annotations

import functools
import sqlite3
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from self_rag.agent.edges import (
    route_after_answer_grading,
    route_after_evidence_grading,
    route_after_orchestrator_call,
    route_after_rewrite,
)
from self_rag.agent.nodes import (
    aggregate_answers,
    collect_answer,
    compress_context,
    corrective_action,
    fallback_response,
    grade_answer,
    grade_evidence,
    orchestrator,
    request_clarification,
    rewrite_query,
    should_compress_context,
    summarize_history,
)
from self_rag.agent.state import AgentState, State
from self_rag.agent.tools import ToolFactory
from self_rag.config import Settings, get_settings
from self_rag.llm.factory import build_llm
from self_rag.storage.embeddings import get_dense_embeddings, get_sparse_embeddings
from self_rag.storage.parent_store import ParentStore
from self_rag.storage.vector_store import VectorStoreManager


def build_agent_subgraph(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    grader_llm: BaseChatModel | None = None,
    self_correction_enabled: bool = False,
    grade_evidence_enabled: bool = True,
    grade_answer_enabled: bool = True,
    max_correction_retries: int = 2,
    max_iterations: int = 4,
    max_tool_calls: int = 4,
    base_token_threshold: int = 2000,
    token_growth_factor: float = 0.9,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Build and compile the agent loop subgraph with budget, compression, and self-correction.

    Topology:
        START -> orchestrator
        orchestrator -> (conditional) -> 'tools' -> 'should_compress_context' (or 'grade_evidence')
                                      -> 'fallback_response' -> 'collect_answer'
                                      -> 'grade_answer' -> 'collect_answer' / 'corrective_action'
                                      -> 'collect_answer' -> END
        corrective_action -> orchestrator
        should_compress_context -> (Command goto) -> 'compress_context' -> orchestrator
                                                 -> 'orchestrator'
    """
    active_grader = grader_llm if grader_llm is not None else llm

    llm_with_tools: Any
    try:
        llm_with_tools = llm.bind_tools(list(tools))
    except NotImplementedError:
        llm_with_tools = llm

    builder = StateGraph(AgentState)
    builder.add_node(
        "orchestrator",
        functools.partial(orchestrator, llm_with_tools=llm_with_tools),
    )
    builder.add_node("tools", ToolNode(tools))
    builder.add_node(
        "should_compress_context",
        functools.partial(
            should_compress_context,
            base_token_threshold=base_token_threshold,
            token_growth_factor=token_growth_factor,
        ),
    )
    builder.add_node(
        "compress_context",
        functools.partial(compress_context, llm=llm),
    )
    builder.add_node(
        "fallback_response",
        functools.partial(fallback_response, llm=llm),
    )
    builder.add_node("collect_answer", collect_answer)

    if self_correction_enabled:
        builder.add_node(
            "corrective_action",
            functools.partial(corrective_action, llm=active_grader),
        )
        builder.add_edge("corrective_action", "orchestrator")

        if grade_evidence_enabled:
            builder.add_node(
                "grade_evidence",
                functools.partial(grade_evidence, grader_llm=active_grader),
            )
            builder.add_edge("tools", "grade_evidence")
            builder.add_conditional_edges(
                "grade_evidence",
                functools.partial(
                    route_after_evidence_grading,
                    max_correction_retries=max_correction_retries,
                ),
                {
                    "should_compress_context": "should_compress_context",
                    "corrective_action": "corrective_action",
                    "fallback_response": "fallback_response",
                },
            )
        else:
            builder.add_edge("tools", "should_compress_context")

        if grade_answer_enabled:
            builder.add_node(
                "grade_answer",
                functools.partial(grade_answer, grader_llm=active_grader),
            )
            builder.add_conditional_edges(
                "grade_answer",
                functools.partial(
                    route_after_answer_grading,
                    max_correction_retries=max_correction_retries,
                ),
                {
                    "collect_answer": "collect_answer",
                    "corrective_action": "corrective_action",
                    "fallback_response": "fallback_response",
                },
            )
    else:
        builder.add_edge("tools", "should_compress_context")

    builder.add_edge(START, "orchestrator")

    orchestrator_routing: dict[Hashable, str] = {
        "tools": "tools",
        "fallback_response": "fallback_response",
        "collect_answer": "collect_answer",
    }
    if self_correction_enabled and grade_answer_enabled:
        orchestrator_routing["grade_answer"] = "grade_answer"

    builder.add_conditional_edges(
        "orchestrator",
        functools.partial(
            route_after_orchestrator_call,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            self_correction_enabled=self_correction_enabled,
            grade_answer_enabled=grade_answer_enabled,
        ),
        orchestrator_routing,
    )

    builder.add_edge("compress_context", "orchestrator")
    builder.add_edge("fallback_response", "collect_answer")
    builder.add_edge("collect_answer", END)

    return builder.compile()


def ask_question(
    question: str,
    *,
    settings: Settings | None = None,
    self_correction_enabled: bool | None = None,
    grader_llm: BaseChatModel | None = None,
) -> dict[str, Any]:
    """Execute the agent research loop on a question against configured storage and LLM.

    Returns the graph output state containing ``final_answer`` and ``agent_answers``.
    Raises ValueError if the vector collection does not exist or contains no documents.
    """
    active_settings = settings or get_settings()
    active_correction = (
        active_settings.self_correction_enabled
        if self_correction_enabled is None
        else self_correction_enabled
    )
    vm = VectorStoreManager(settings=active_settings)

    try:
        if not vm.collection_exists():
            raise ValueError(
                f"Vector collection '{vm.collection_name}' does not exist. "
                "Ingest documents first using 'self-rag ingest <file>'."
            )
        if vm.count() == 0:
            raise ValueError(
                f"Vector collection '{vm.collection_name}' is empty. "
                "Ingest documents first using 'self-rag ingest <file>'."
            )

        dense = get_dense_embeddings(active_settings)
        sparse = get_sparse_embeddings(active_settings)
        vector_store = vm.get_vector_store(embedding=dense, sparse_embedding=sparse)
        parent_store = ParentStore(settings=active_settings)

        tool_factory = ToolFactory(
            collection=vector_store,
            parent_store=parent_store,
            default_limit=active_settings.retrieval_k,
            score_threshold=active_settings.hybrid_score_floor,
        )
        tools = tool_factory.create_tools()

        llm = build_llm(active_settings)
        graph = build_agent_subgraph(
            llm=llm,
            tools=tools,
            grader_llm=grader_llm,
            self_correction_enabled=active_correction,
            max_correction_retries=active_settings.max_correction_retries,
            max_iterations=active_settings.max_iterations,
            max_tool_calls=active_settings.max_tool_calls,
            base_token_threshold=active_settings.base_token_threshold,
            token_growth_factor=active_settings.token_growth_factor,
        )

        input_state: AgentState = {"question": question, "messages": []}
        output: dict[str, Any] = graph.invoke(input_state)
        return output
    finally:
        vm.close()


def create_sqlite_checkpointer(db_path: Path | str) -> SqliteSaver:
    """Create a persistent SQLite checkpointer at the specified database file path."""
    resolved_path = Path(db_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(resolved_path), check_same_thread=False)
    return SqliteSaver(conn)


def build_main_graph(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    grader_llm: BaseChatModel | None = None,
    self_correction_enabled: bool = False,
    grade_evidence_enabled: bool = True,
    grade_answer_enabled: bool = True,
    max_correction_retries: int = 2,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    max_subquestions: int = 2,
    main_history_messages_to_keep: int = 4,
    max_iterations: int = 4,
    max_tool_calls: int = 4,
    base_token_threshold: int = 2000,
    token_growth_factor: float = 0.9,
) -> CompiledStateGraph[State, None, State, State]:
    """Build and compile the main multi-turn conversation graph with memory and checkpointer.

    Topology:
        START -> summarize_history -> rewrite_query
        rewrite_query -> (conditional: route_after_rewrite)
            -> request_clarification (interrupted) -> rewrite_query
            -> [agent (parallel fan-out via Send)] -> aggregate_answers -> END
    """
    agent_subgraph = build_agent_subgraph(
        llm=llm,
        tools=tools,
        grader_llm=grader_llm,
        self_correction_enabled=self_correction_enabled,
        grade_evidence_enabled=grade_evidence_enabled,
        grade_answer_enabled=grade_answer_enabled,
        max_correction_retries=max_correction_retries,
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
        base_token_threshold=base_token_threshold,
        token_growth_factor=token_growth_factor,
    )

    builder = StateGraph(State)
    builder.add_node(
        "summarize_history",
        functools.partial(
            summarize_history,
            llm=llm,
            keep_count=main_history_messages_to_keep,
        ),
    )
    builder.add_node(
        "rewrite_query",
        functools.partial(
            rewrite_query,
            llm=llm,
            max_subquestions=max_subquestions,
        ),
    )
    builder.add_node("request_clarification", request_clarification)
    builder.add_node("agent", agent_subgraph)
    builder.add_node(
        "aggregate_answers",
        functools.partial(aggregate_answers, llm=llm),
    )

    builder.add_edge(START, "summarize_history")
    builder.add_edge("summarize_history", "rewrite_query")
    builder.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        ["request_clarification", "agent"],
    )
    builder.add_edge("request_clarification", "rewrite_query")
    builder.add_edge("agent", "aggregate_answers")
    builder.add_edge("aggregate_answers", END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["request_clarification"],
    )
