"""Graph nodes for agent research subgraphs and conversation graphs."""

from __future__ import annotations

import contextlib
import json
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langgraph.types import Command

from self_rag.agent.prompts import (
    get_aggregation_prompt,
    get_answer_grader_prompt,
    get_context_compression_prompt,
    get_conversation_summary_prompt,
    get_document_grader_prompt,
    get_fallback_response_prompt,
    get_orchestrator_prompt,
    get_refine_query_prompt,
    get_rewrite_query_prompt,
)
from self_rag.agent.schemas import (
    GradeAnswer,
    GradeDocuments,
    QueryAnalysis,
    RefinedQuery,
)
from self_rag.agent.state import AgentState, State
from self_rag.agent.tools import _retrieval_contexts
from self_rag.text.tokens import estimate_messages_tokens, estimate_tokens


def orchestrator(
    state: AgentState,
    llm_with_tools: Runnable[Any, AIMessage],
) -> dict[str, Any]:
    """Execute one step of the orchestrator LLM in the sub-question agent loop.

    - First call: injects the user question and the mandatory retrieval instruction.
    - Subsequent calls: replays accumulated conversation history.
    - Injects compressed context summary if present in state.
    - Tags AI responses with name='agent_response'.
    - Updates tool_call_count and iteration_count.
    """
    messages = list(state.get("messages", []))
    question = state.get("question", "").strip()

    if not question:
        for msg in messages:
            if isinstance(msg, HumanMessage) and msg.content:
                question = str(msg.content).strip()
                break

    # Build system messages
    system_prompt = get_orchestrator_prompt()
    system_messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    # If a compressed context summary exists, inject it
    context_summary = state.get("context_summary", "").strip()
    if context_summary:
        system_messages.append(
            HumanMessage(content=f"Compressed Research Context:\n{context_summary}")
        )

    if not messages:
        # First step in agent loop: inject question and mandatory initial search prompt
        initial_content = (
            f"Question: {question}\n\n"
            "YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP to find relevant "
            "document evidence before attempting to answer."
        )
        initial_user_msg = HumanMessage(content=initial_content)
        prompt_messages = [*system_messages, initial_user_msg]
        response = llm_with_tools.invoke(prompt_messages)
        response.name = "agent_response"
        tool_calls = getattr(response, "tool_calls", [])
        return {
            "messages": [initial_user_msg, response],
            "tool_call_count": len(tool_calls),
            "iteration_count": 1,
        }

    # Subsequent step: replay history
    prompt_messages = [*system_messages, *messages]
    response = llm_with_tools.invoke(prompt_messages)
    response.name = "agent_response"
    tool_calls = getattr(response, "tool_calls", [])
    return {
        "messages": [response],
        "tool_call_count": len(tool_calls),
        "iteration_count": 1,
    }


def should_compress_context(
    state: AgentState,
    *,
    base_token_threshold: int = 2000,
    token_growth_factor: float = 0.9,
) -> Command[Literal["compress_context", "orchestrator"]]:
    """Inspect research context size and decide whether to compress.

    - Extracts search query and parent ID from the most recent tool calls into retrieval_keys.
    - Recomputes retrieved_contexts via _retrieval_contexts.
    - Computes current_tokens = tokens(messages) + tokens(context_summary).
    - Computes max_allowed = base_token_threshold + int(
        current_token_summary * token_growth_factor
      ).
    - If current_tokens > max_allowed: routes to 'compress_context'.
    - Otherwise: routes to 'orchestrator'.
    """
    messages = state.get("messages", [])
    new_keys: set[str] = set()

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                name = call.get("name", "")
                args = call.get("args", {})
                if name == "search_child_chunks":
                    q = str(args.get("query", "")).strip()
                    if q:
                        new_keys.add(f"search::{q}")
                elif name == "retrieve_parent_chunks":
                    pid = str(args.get("parent_id", "")).strip()
                    if pid:
                        new_keys.add(f"parent::{pid}")
            break

    contexts = _retrieval_contexts(messages)
    context_summary = state.get("context_summary", "")
    current_token_summary = estimate_tokens(context_summary) if context_summary else 0
    current_tokens = estimate_messages_tokens(messages) + current_token_summary
    max_allowed = base_token_threshold + int(current_token_summary * token_growth_factor)

    goto: Literal["compress_context", "orchestrator"] = (
        "compress_context" if current_tokens > max_allowed else "orchestrator"
    )

    update: dict[str, Any] = {
        "retrieval_keys": new_keys,
        "retrieved_contexts": contexts,
    }
    return Command(update=update, goto=goto)


def compress_context(
    state: AgentState,
    llm: BaseChatModel,
) -> dict[str, Any]:
    """Compress accumulated tool messages and conversation into a structured summary.

    - Serializes assistant text, tool calls, and tool outputs into a plain text representation.
    - Appends any prior compressed summary.
    - Asks LLM to produce a fresh Markdown summary using get_context_compression_prompt().
    - Appends an 'Already executed (do NOT repeat)' block with all recorded retrieval keys.
    - Emits RemoveMessage for all processed messages to reset working context.
    """
    messages = state.get("messages", [])
    lines: list[str] = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.content:
                lines.append(f"Assistant: {msg.content}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lines.append(f"Tool Call: {tc.get('name')}({tc.get('args')})")
        elif isinstance(msg, ToolMessage):
            name = msg.name or getattr(msg, "tool_call_id", "tool")
            lines.append(f"Tool Result ({name}): {msg.content}")

    serialized_execution = "\n".join(lines)
    prior_summary = state.get("context_summary", "")
    if prior_summary:
        full_text = (
            f"Prior Compressed Context:\n{prior_summary}\n\n"
            f"Recent Execution:\n{serialized_execution}"
        )
    else:
        full_text = serialized_execution

    prompt = get_context_compression_prompt()
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=full_text)])
    compressed_text = str(response.content).strip()

    retrieval_keys = sorted(state.get("retrieval_keys", set()))
    if retrieval_keys:
        block = "\n\n### Already executed (do NOT repeat):\n" + "\n".join(
            f"- {k}" for k in retrieval_keys
        )
        compressed_text += block

    removals = [RemoveMessage(id=m.id) for m in messages if m.id]
    return {
        "context_summary": compressed_text,
        "messages": removals,
    }


def fallback_response(
    state: AgentState,
    llm: BaseChatModel,
) -> dict[str, Any]:
    """Synthesize a direct answer when the agent's research budget has been reached.

    - Collects every unique raw ToolMessage content seen in the run.
    - Combines it with existing compressed context summary.
    - Calls LLM using get_fallback_response_prompt().
    - Returns AIMessage with name='agent_response'.
    """
    raw_tool_contents: list[str] = []
    seen: set[str] = set()

    for msg in state.get("messages", []):
        if isinstance(msg, ToolMessage) and msg.content:
            content_str = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            if content_str not in seen:
                seen.add(content_str)
                raw_tool_contents.append(content_str)

    context_parts: list[str] = []
    summary = state.get("context_summary", "").strip()
    if summary:
        context_parts.append(f"Compressed Research Context:\n{summary}")
    if raw_tool_contents:
        retrieved_blob = "\n\n".join(raw_tool_contents)
        context_parts.append(f"Retrieved Data:\n{retrieved_blob}")

    question = state.get("question", "").strip()
    if not question:
        for m in state.get("messages", []):
            if isinstance(m, HumanMessage) and m.content:
                question = str(m.content).strip()
                break

    context_parts.append(f"Question: {question}")
    full_prompt = "\n\n".join(context_parts)

    system_prompt = get_fallback_response_prompt()
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=full_prompt)])
    response.name = "agent_response"

    return {"messages": [response]}


def collect_answer(state: AgentState) -> dict[str, Any]:
    """Collect and package the final answer from accumulated conversation messages.

    - Scans messages backwards for the most recent AIMessage without tool calls.
    - If found and non-empty, accepts it as final_answer; otherwise 'Unable to generate an answer.'.
    - Extracts contexts using _retrieval_contexts.
    - Packages {index, question, answer, contexts} into agent_answers.
    """
    messages = state.get("messages", [])
    answer = "Unable to generate an answer."

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
            content = msg.content
            if isinstance(content, str):
                cleaned = content.strip()
            else:
                parts = [str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content]
                cleaned = " ".join(parts).strip()

            if cleaned:
                answer = cleaned
                break

    question = state.get("question", "")
    if not question:
        for msg in messages:
            if isinstance(msg, HumanMessage) and msg.content:
                question = str(msg.content).strip()
                break

    contexts = _retrieval_contexts(messages)
    index = state.get("question_index", 0)

    entry: dict[str, Any] = {
        "index": index,
        "question": question,
        "answer": answer,
        "contexts": contexts,
    }

    return {
        "final_answer": answer,
        "agent_answers": [entry],
        "retrieved_contexts": contexts,
    }


def summarize_history(
    state: State,
    llm: BaseChatModel,
    keep_count: int = 4,
) -> dict[str, Any]:
    """Prune and summarize older messages in the conversation history.

    - Resets agent_answers via {'__reset__': True}.
    - Keeps the most recent (keep_count - 1) plain messages verbatim.
    - Merges any older plain messages into conversation_summary using
      get_conversation_summary_prompt().
    - Emits RemoveMessage for all summarized older messages.
    """
    messages = state.get("messages", [])
    plain_msgs: list[AnyMessage] = [
        m for m in messages if not isinstance(m, ToolMessage) and not getattr(m, "name", None)
    ]

    keep_verbatim = max(1, keep_count - 1)
    if len(plain_msgs) <= keep_verbatim:
        return {"agent_answers": [{"__reset__": True}]}

    cutoff_msg = plain_msgs[-keep_verbatim]
    cutoff_idx = messages.index(cutoff_msg)
    messages_to_summarize = messages[:cutoff_idx]

    lines: list[str] = []
    for m in messages_to_summarize:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            lines.append(f"Assistant: {m.content}")

    if not lines:
        return {"agent_answers": [{"__reset__": True}]}

    formatted_messages = "\n".join(lines)
    existing_summary = state.get("conversation_summary", "").strip()
    if existing_summary:
        content = f"Existing Summary:\n{existing_summary}\n\nOlder Messages:\n{formatted_messages}"
    else:
        content = f"Older Messages:\n{formatted_messages}"

    prompt = get_conversation_summary_prompt()
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=content)])
    new_summary = str(response.content).strip()

    removals = [RemoveMessage(id=m.id) for m in messages_to_summarize if m.id]
    return {
        "conversation_summary": new_summary,
        "messages": removals,
        "agent_answers": [{"__reset__": True}],
    }


def rewrite_query(
    state: State,
    llm: BaseChatModel,
    max_subquestions: int = 2,
) -> dict[str, Any]:
    """Analyze and rewrite user query into self-contained search questions or request clarification.

    - Gathers conversation summary and recent conversation context.
    - If in a clarification round-trip (pendingQuery set), combines unresolved query
      and user clarifications.
    - Invokes LLM with QueryAnalysis structured output schema.
    - If query is clear: produces up to max_subquestions rewritten questions and clears
      pending state.
    - If query is unclear: stores pending query and clarifications, emits
      AIMessage(name='clarification').
    """
    messages = state.get("messages", [])
    current_input = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and m.content:
            current_input = str(m.content).strip()
            break

    pending_query = state.get("pendingQuery", "").strip()
    clarifications = list(state.get("pendingClarifications", []))
    if pending_query and current_input:
        clarifications.append(current_input)

    context_parts: list[str] = []
    summary = state.get("conversation_summary", "").strip()
    if summary:
        context_parts.append(f"Conversation Summary:\n{summary}")

    recent_lines: list[str] = []
    for m in messages:
        if getattr(m, "name", None) == "clarification":
            continue
        if pending_query and m.content == pending_query:
            continue
        if m.content == current_input and isinstance(m, HumanMessage):
            continue
        if isinstance(m, HumanMessage):
            recent_lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            recent_lines.append(f"Assistant: {m.content}")

    if recent_lines:
        context_parts.append("Recent Conversation:\n" + "\n".join(recent_lines))

    if pending_query:
        context_parts.append(f"Unresolved Query:\n{pending_query}")
        clarifications_text = "\n".join(f"- {c}" for c in clarifications)
        context_parts.append(f"User Clarifications:\n{clarifications_text}")
    else:
        context_parts.append(f"Current User Query:\n{current_input}")

    input_text = "\n\n".join(context_parts)
    system_prompt = get_rewrite_query_prompt()
    structured_llm = llm.with_structured_output(QueryAnalysis)
    raw_analysis: Any = structured_llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=input_text)]
    )

    if isinstance(raw_analysis, QueryAnalysis):
        analysis = raw_analysis
    elif isinstance(raw_analysis, dict):
        analysis = QueryAnalysis.model_validate(raw_analysis)
    else:
        analysis = QueryAnalysis(is_clear=True, questions=[current_input])

    if analysis.is_clear:
        questions = analysis.questions if analysis.questions else [pending_query or current_input]
        questions = questions[:max_subquestions]
        return {
            "questionIsClear": True,
            "rewrittenQuestions": questions,
            "originalQuery": state.get("originalQuery") or pending_query or current_input,
            "pendingQuery": "",
            "pendingClarifications": [],
        }

    clarification = (
        analysis.clarification_needed.strip() or "Could you please clarify your question?"
    )
    return {
        "questionIsClear": False,
        "originalQuery": state.get("originalQuery") or pending_query or current_input,
        "pendingQuery": pending_query or current_input,
        "pendingClarifications": clarifications,
        "messages": [AIMessage(content=clarification, name="clarification")],
    }


def request_clarification(state: State) -> dict[str, Any]:
    """No-op node serving as the interrupt_before pause point for user clarification."""
    return {}


def aggregate_answers(
    state: State,
    llm: BaseChatModel,
) -> dict[str, Any]:
    """Synthesize a unified final answer from accumulated sub-question agent_answers."""
    agent_answers = state.get("agent_answers", [])
    if not agent_answers:
        final_answer = (
            "I couldn't find any information to answer your question in the available sources."
        )
        return {"messages": [AIMessage(content=final_answer)]}

    sorted_answers = sorted(agent_answers, key=lambda a: int(a.get("index", 0)))
    blocks: list[str] = []
    for i, ans in enumerate(sorted_answers, start=1):
        content = str(ans.get("answer", "")).strip()
        blocks.append(f"Retrieved response {i}:\n{content}")
    answers_payload = "\n\n".join(blocks)

    question = state.get("originalQuery") or state.get("pendingQuery") or ""
    if not question:
        for m in reversed(state.get("messages", [])):
            if isinstance(m, HumanMessage) and m.content:
                question = str(m.content).strip()
                break

    user_content = f"Question: {question}\n\nRetrieved Answers:\n{answers_payload}"
    prompt = get_aggregation_prompt()
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=user_content)])
    final_answer = str(response.content).strip()
    return {"messages": [AIMessage(content=final_answer)]}


def grade_document_relevance(
    question: str,
    document: str,
    llm: BaseChatModel,
) -> GradeDocuments:
    """Evaluate whether a document excerpt is relevant to the question.

    Returns a GradeDocuments instance with binary_score ('yes' | 'no') and explanation.
    """
    system_prompt = get_document_grader_prompt()
    user_content = f"Question: {question}\n\nDocument:\n{document}"
    structured_llm = llm.with_structured_output(GradeDocuments)
    raw = structured_llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    )
    if isinstance(raw, GradeDocuments):
        return raw
    if isinstance(raw, dict):
        return GradeDocuments.model_validate(raw)
    return GradeDocuments(binary_score="yes", explanation="Default accept")


def grade_answer_groundedness(
    question: str,
    context: str,
    answer: str,
    llm: BaseChatModel,
) -> GradeAnswer:
    """Evaluate whether a generated answer is grounded in facts and addresses the question.

    Returns a GradeAnswer instance with binary_score ('yes' | 'no') and explanation.
    """
    system_prompt = get_answer_grader_prompt()
    user_content = (
        f"Question: {question}\n\n"
        f"Retrieved Document Context:\n{context}\n\n"
        f"Generated Answer:\n{answer}"
    )
    structured_llm = llm.with_structured_output(GradeAnswer)
    raw = structured_llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    )
    if isinstance(raw, GradeAnswer):
        return raw
    if isinstance(raw, dict):
        return GradeAnswer.model_validate(raw)
    return GradeAnswer(binary_score="yes", explanation="Default accept")


def refine_query_for_retrieval(
    question: str,
    feedback: str,
    llm: BaseChatModel,
) -> str:
    """Formulate an improved search query using feedback from a failed retrieval or grading."""
    system_prompt = get_refine_query_prompt()
    user_content = f"Question: {question}\n\nFeedback / Failure Reason:\n{feedback}"
    structured_llm = llm.with_structured_output(RefinedQuery)
    raw = structured_llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    )
    if isinstance(raw, RefinedQuery):
        return raw.query.strip()
    if isinstance(raw, dict):
        return str(raw.get("query", question)).strip()
    return question


def grade_evidence(
    state: AgentState,
    grader_llm: BaseChatModel,
) -> dict[str, Any]:
    """Grade whether the accumulated retrieved evidence is relevant to the question."""
    question = state.get("question", "").strip()
    messages = list(state.get("messages", []))
    if not question:
        for msg in messages:
            if isinstance(msg, HumanMessage) and msg.content:
                question = str(msg.content).strip()
                break

    contexts = list(state.get("retrieved_contexts", []))
    if not contexts:
        contexts = _retrieval_contexts(messages)

    if not contexts:
        return {
            "evidence_score": "no",
            "evidence_explanation": "No document evidence has been retrieved yet.",
        }

    combined_context = "\n\n".join(contexts[:5])
    result = grade_document_relevance(question, combined_context, grader_llm)
    score = result.binary_score.lower().strip()
    score = "yes" if "yes" in score else "no"
    return {
        "evidence_score": score,
        "evidence_explanation": result.explanation.strip(),
    }


def grade_answer(
    state: AgentState,
    grader_llm: BaseChatModel,
) -> dict[str, Any]:
    """Grade whether the candidate final answer is grounded in facts and addresses the question."""
    question = state.get("question", "").strip()
    messages = list(state.get("messages", []))
    if not question:
        for msg in messages:
            if isinstance(msg, HumanMessage) and msg.content:
                question = str(msg.content).strip()
                break

    draft_answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
            if isinstance(msg.content, str):
                draft_answer = msg.content.strip()
            else:
                parts = [
                    str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in msg.content
                ]
                draft_answer = " ".join(parts).strip()
            if draft_answer:
                break

    if not draft_answer:
        return {
            "answer_score": "no",
            "answer_explanation": "No candidate answer was produced by the agent.",
        }

    contexts = list(state.get("retrieved_contexts", []))
    if not contexts:
        contexts = _retrieval_contexts(messages)
    context_summary = state.get("context_summary", "").strip()

    context_parts: list[str] = []
    if context_summary:
        context_parts.append(f"Context Summary:\n{context_summary}")
    if contexts:
        context_parts.append("Retrieved Chunks:\n" + "\n\n".join(contexts[:5]))

    combined_context = "\n\n".join(context_parts)
    if not combined_context:
        return {
            "answer_score": "no",
            "answer_explanation": "Answer is ungrounded: no document evidence was retrieved.",
        }

    result = grade_answer_groundedness(question, combined_context, draft_answer, grader_llm)
    score = result.binary_score.lower().strip()
    score = "yes" if "yes" in score else "no"
    return {
        "answer_score": score,
        "answer_explanation": result.explanation.strip(),
    }


def corrective_action(
    state: AgentState,
    llm: BaseChatModel | None = None,
) -> dict[str, Any]:
    """Inject corrective re-retrieval guidance and increment correction retry count."""
    question = state.get("question", "").strip()
    explanation = (
        state.get("answer_explanation")
        or state.get("evidence_explanation")
        or "The retrieved evidence or answer was judged insufficient."
    )

    refined_query = question
    if llm is not None and question:
        with contextlib.suppress(Exception):
            refined_query = refine_query_for_retrieval(question, explanation, llm)

    corrective_guidance = (
        f"Corrective Re-retrieval Guidance:\n"
        f"The previous retrieval or answer was judged insufficient. Reason: {explanation}\n\n"
        f"Target query: '{refined_query}'\n"
        f"Please search again using 'search_child_chunks' with a targeted query to find supporting "
        f"evidence before answering."
    )
    user_msg = HumanMessage(content=corrective_guidance)

    return {
        "correction_count": 1,
        "messages": [user_msg],
    }
