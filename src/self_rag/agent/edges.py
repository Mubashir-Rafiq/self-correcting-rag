"""Routing conditional edges for agent graphs."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Send

from self_rag.agent.state import AgentState, State


def route_after_orchestrator_call(
    state: AgentState,
    max_iterations: int = 4,
    max_tool_calls: int = 4,
    self_correction_enabled: bool = False,
    grade_answer_enabled: bool = True,
) -> str:
    """Route execution after orchestrator node invocation.

    - If iteration or tool call limits are reached: returns 'fallback_response'.
    - If tool calls are requested and budget remains: returns 'tools'.
    - If the last AI message has no tool calls:
      - If self-correction and answer grading are enabled: returns 'grade_answer'.
      - Otherwise: returns 'collect_answer'.
    """
    messages = state.get("messages", [])
    if not messages:
        return (
            "grade_answer"
            if (self_correction_enabled and grade_answer_enabled)
            else "collect_answer"
        )

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        if self_correction_enabled and grade_answer_enabled:
            return "grade_answer"
        return "collect_answer"

    # Tool calls are requested: check research budgets
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)

    if iteration >= max_iterations or tool_count > max_tool_calls:
        return "fallback_response"

    return "tools"


def route_after_evidence_grading(
    state: AgentState,
    max_correction_retries: int = 2,
) -> str:
    """Route after evidence grading.

    - If evidence is relevant ('yes'): continues to 'should_compress_context'.
    - If evidence is not relevant ('no') and correction budget remains: returns 'corrective_action'.
    - If evidence is not relevant ('no') and budget exhausted: returns 'fallback_response'.
    """
    score = state.get("evidence_score", "yes").lower()
    if score == "yes":
        return "should_compress_context"

    corrections = state.get("correction_count", 0)
    if corrections < max_correction_retries:
        return "corrective_action"

    return "fallback_response"


def route_after_answer_grading(
    state: AgentState,
    max_correction_retries: int = 2,
) -> str:
    """Route after answer grading.

    - If answer is grounded ('yes'): returns 'collect_answer'.
    - If answer is ungrounded ('no') and correction budget remains: returns 'corrective_action'.
    - If answer is ungrounded ('no') and budget exhausted: returns 'fallback_response'.
    """
    score = state.get("answer_score", "yes").lower()
    if score == "yes":
        return "collect_answer"

    corrections = state.get("correction_count", 0)
    if corrections < max_correction_retries:
        return "corrective_action"

    return "fallback_response"


def route_after_rewrite(state: State) -> Sequence[Send] | str:
    """Route after query rewriting to either clarification pause or parallel agent subgraphs.

    - If questionIsClear is False: returns 'request_clarification'.
    - If questionIsClear is True: fans out with Send('agent', ...) for each rewritten question.
    """
    if not state.get("questionIsClear", False):
        return "request_clarification"

    questions = state.get("rewrittenQuestions", [])
    if not questions:
        fallback_q = state.get("originalQuery") or state.get("pendingQuery") or ""
        questions = [fallback_q] if fallback_q else []

    if not questions:
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage) and m.content:
                questions = [str(m.content).strip()]
                break

    return [
        Send("agent", {"question": q, "question_index": i, "messages": []})
        for i, q in enumerate(questions)
    ]
