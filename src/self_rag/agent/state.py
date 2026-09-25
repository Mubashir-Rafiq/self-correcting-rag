"""LangGraph state schemas and reducers for agent graphs.

Defines:
- `State`: State schema for the main multi-turn conversation graph.
- `AgentState`: State schema for individual sub-question research subgraphs.
- Channel reducers: `accumulate_or_reset`, `set_union`, and `append_unique`.
"""

import operator
from typing import Annotated, Any, NotRequired

from langgraph.graph import MessagesState


def accumulate_or_reset(
    existing: list[dict[str, Any]] | None,
    new: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reducer that accumulates agent answer dicts or clears them on reset.

    If any item in ``new`` contains a truthy ``'__reset__'`` flag, the accumulated
    answers are wiped and an empty list is returned. Otherwise, ``new`` is appended
    to ``existing``.
    """
    if not existing:
        existing = []
    if new and any(item.get("__reset__") for item in new):
        return []
    return existing + new


def set_union(a: set[str] | None, b: set[str]) -> set[str]:
    """Reducer that merges two sets of retrieval keys via set union."""
    if a is None:
        a = set()
    return a | b


def append_unique(existing: list[str] | None, new: list[str]) -> list[str]:
    """Reducer that appends strings while preserving insertion order and deduplicating."""
    if not existing:
        existing = []
    return list(dict.fromkeys(existing + new))


class State(MessagesState):
    """State schema for the main multi-turn conversation graph.

    Inherits ``messages: Annotated[list[AnyMessage], add_messages]`` from ``MessagesState``.
    """

    questionIsClear: NotRequired[bool]
    conversation_summary: NotRequired[str]
    originalQuery: NotRequired[str]
    pendingQuery: NotRequired[str]
    pendingClarifications: NotRequired[list[str]]
    rewrittenQuestions: NotRequired[list[str]]
    agent_answers: NotRequired[Annotated[list[dict[str, Any]], accumulate_or_reset]]


class AgentState(MessagesState):
    """State schema for individual sub-question research subgraphs.

    Inherits ``messages: Annotated[list[AnyMessage], add_messages]`` from ``MessagesState``.
    """

    question: NotRequired[str]
    question_index: NotRequired[int]
    context_summary: NotRequired[str]
    retrieval_keys: NotRequired[Annotated[set[str], set_union]]
    retrieved_contexts: NotRequired[Annotated[list[str], append_unique]]
    final_answer: NotRequired[str]
    agent_answers: NotRequired[list[dict[str, Any]]]
    tool_call_count: NotRequired[Annotated[int, operator.add]]
    iteration_count: NotRequired[Annotated[int, operator.add]]
    correction_count: NotRequired[Annotated[int, operator.add]]
    evidence_score: NotRequired[str]
    evidence_explanation: NotRequired[str]
    answer_score: NotRequired[str]
    answer_explanation: NotRequired[str]
