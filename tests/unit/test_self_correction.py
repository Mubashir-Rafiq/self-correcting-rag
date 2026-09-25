"""Tests for Stage 11: Self-correction, grading, and corrective re-retrieval."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from self_rag.agent.edges import (
    route_after_answer_grading,
    route_after_evidence_grading,
    route_after_orchestrator_call,
)
from self_rag.agent.graph import build_agent_subgraph
from self_rag.agent.nodes import (
    corrective_action,
    grade_answer,
    grade_answer_groundedness,
    grade_document_relevance,
    grade_evidence,
    refine_query_for_retrieval,
)
from self_rag.agent.prompts import (
    get_answer_grader_prompt,
    get_document_grader_prompt,
    get_refine_query_prompt,
)
from self_rag.agent.schemas import GradeAnswer, GradeDocuments, RefinedQuery
from self_rag.agent.state import AgentState
from self_rag.config import Settings
from self_rag.system import RAGSystem


class TestGraderPromptsAndSchemas:
    """Tests verifying prompt contents and Pydantic schemas for self-correction."""

    def test_document_grader_prompt_structure(self) -> None:
        prompt = get_document_grader_prompt()
        assert "## Role" in prompt
        assert "grader assessing relevance" in prompt.lower()
        assert "'yes'" in prompt
        assert "'no'" in prompt

    def test_answer_grader_prompt_structure(self) -> None:
        prompt = get_answer_grader_prompt()
        assert "## Role" in prompt
        assert "grounded in facts" in prompt.lower()
        assert "'yes'" in prompt
        assert "'no'" in prompt

    def test_refine_query_prompt_structure(self) -> None:
        prompt = get_refine_query_prompt()
        assert "## Role" in prompt
        assert "query refinement" in prompt.lower()
        assert "targeted search query" in prompt.lower()

    def test_grade_documents_schema(self) -> None:
        valid = GradeDocuments(binary_score="yes", explanation="Contains relevant facts.")
        assert valid.binary_score == "yes"
        assert valid.explanation == "Contains relevant facts."

    def test_grade_answer_schema(self) -> None:
        valid = GradeAnswer(binary_score="no", explanation="Missing factual grounding.")
        assert valid.binary_score == "no"
        assert valid.explanation == "Missing factual grounding."

    def test_refined_query_schema(self) -> None:
        valid = RefinedQuery(query="hybrid search RRF", explanation="Targeting RRF details.")
        assert valid.query == "hybrid search RRF"
        assert valid.explanation == "Targeting RRF details."


class TestGraderHelperFunctions:
    """Tests for standalone grading and refinement helpers."""

    def test_grade_document_relevance_yes(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = GradeDocuments(
            binary_score="yes", explanation="Relevant document."
        )

        res = grade_document_relevance(
            question="What is Qdrant?",
            document="Qdrant is a vector database.",
            llm=mock_llm,
        )
        assert res.binary_score == "yes"
        assert res.explanation == "Relevant document."

    def test_grade_document_relevance_fallback_dict(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = {
            "binary_score": "no",
            "explanation": "Off-topic.",
        }

        res = grade_document_relevance(
            question="What is Qdrant?",
            document="Bananas are rich in potassium.",
            llm=mock_llm,
        )
        assert res.binary_score == "no"
        assert res.explanation == "Off-topic."

    def test_grade_answer_groundedness_yes(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = GradeAnswer(
            binary_score="yes", explanation="Fully supported."
        )

        res = grade_answer_groundedness(
            question="What is Qdrant?",
            context="Qdrant is a vector search engine.",
            answer="Qdrant is a vector search engine.",
            llm=mock_llm,
        )
        assert res.binary_score == "yes"
        assert res.explanation == "Fully supported."

    def test_refine_query_for_retrieval(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = RefinedQuery(
            query="Qdrant hybrid sparse dense vectors",
            explanation="Added sparse and dense vector keywords.",
        )

        refined = refine_query_for_retrieval(
            question="How does search work?",
            feedback="Too broad, missing vector engine context.",
            llm=mock_llm,
        )
        assert refined == "Qdrant hybrid sparse dense vectors"


class TestGraderNodes:
    """Tests for grade_evidence, grade_answer, and corrective_action nodes."""

    def test_grade_evidence_no_contexts_returns_no(self) -> None:
        mock_llm = MagicMock()
        state: AgentState = {"question": "What is FastEmbed?", "messages": []}

        res = grade_evidence(state, grader_llm=mock_llm)
        assert res["evidence_score"] == "no"
        assert "No document evidence" in res["evidence_explanation"]

    def test_grade_evidence_with_contexts_evaluates_yes(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = GradeDocuments(
            binary_score="yes", explanation="Contains FastEmbed info."
        )

        state: AgentState = {
            "question": "What is FastEmbed?",
            "retrieved_contexts": ["FastEmbed is a lightweight Python library for embeddings."],
            "messages": [],
        }

        res = grade_evidence(state, grader_llm=mock_llm)
        assert res["evidence_score"] == "yes"
        assert res["evidence_explanation"] == "Contains FastEmbed info."

    def test_grade_answer_no_candidate_answer_returns_no(self) -> None:
        mock_llm = MagicMock()
        state: AgentState = {
            "question": "What is FastEmbed?",
            "retrieved_contexts": ["Context"],
            "messages": [HumanMessage(content="What is FastEmbed?")],
        }

        res = grade_answer(state, grader_llm=mock_llm)
        assert res["answer_score"] == "no"
        assert "No candidate answer" in res["answer_explanation"]

    def test_grade_answer_no_context_returns_no(self) -> None:
        mock_llm = MagicMock()
        state: AgentState = {
            "question": "What is FastEmbed?",
            "retrieved_contexts": [],
            "messages": [
                HumanMessage(content="What is FastEmbed?"),
                AIMessage(content="FastEmbed is a library.", tool_calls=[]),
            ],
        }

        res = grade_answer(state, grader_llm=mock_llm)
        assert res["answer_score"] == "no"
        assert "Answer is ungrounded" in res["answer_explanation"]

    def test_grade_answer_with_valid_context_and_grounded_answer(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = GradeAnswer(
            binary_score="yes", explanation="Fully grounded in manual.pdf."
        )

        state: AgentState = {
            "question": "What is FastEmbed?",
            "retrieved_contexts": ["FastEmbed uses ONNX runtime."],
            "messages": [
                HumanMessage(content="What is FastEmbed?"),
                AIMessage(content="FastEmbed uses the ONNX runtime.", tool_calls=[]),
            ],
        }

        res = grade_answer(state, grader_llm=mock_llm)
        assert res["answer_score"] == "yes"
        assert res["answer_explanation"] == "Fully grounded in manual.pdf."

    def test_corrective_action_increments_count_and_injects_message(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = RefinedQuery(
            query="refined search query", explanation="Refined terms"
        )

        state: AgentState = {
            "question": "Original question",
            "answer_explanation": "Retrieved chunks lacked ONNX details.",
            "messages": [],
        }

        res = corrective_action(state, llm=mock_llm)
        assert res["correction_count"] == 1
        assert len(res["messages"]) == 1
        assert isinstance(res["messages"][0], HumanMessage)
        assert "Corrective Re-retrieval Guidance" in str(res["messages"][0].content)
        assert "refined search query" in str(res["messages"][0].content)


class TestRoutingEdges:
    """Tests for conditional routing edges in self-correction."""

    def test_route_after_evidence_grading_yes(self) -> None:
        state: AgentState = {"evidence_score": "yes", "messages": []}
        route = route_after_evidence_grading(state, max_correction_retries=2)
        assert route == "should_compress_context"

    def test_route_after_evidence_grading_no_within_budget(self) -> None:
        state: AgentState = {"evidence_score": "no", "correction_count": 0, "messages": []}
        route = route_after_evidence_grading(state, max_correction_retries=2)
        assert route == "corrective_action"

    def test_route_after_evidence_grading_no_budget_exhausted(self) -> None:
        state: AgentState = {"evidence_score": "no", "correction_count": 2, "messages": []}
        route = route_after_evidence_grading(state, max_correction_retries=2)
        assert route == "fallback_response"

    def test_route_after_answer_grading_yes(self) -> None:
        state: AgentState = {"answer_score": "yes", "messages": []}
        route = route_after_answer_grading(state, max_correction_retries=2)
        assert route == "collect_answer"

    def test_route_after_answer_grading_no_within_budget(self) -> None:
        state: AgentState = {"answer_score": "no", "correction_count": 1, "messages": []}
        route = route_after_answer_grading(state, max_correction_retries=2)
        assert route == "corrective_action"

    def test_route_after_answer_grading_no_budget_exhausted(self) -> None:
        state: AgentState = {"answer_score": "no", "correction_count": 2, "messages": []}
        route = route_after_answer_grading(state, max_correction_retries=2)
        assert route == "fallback_response"

    def test_route_after_orchestrator_call_with_self_correction_enabled(self) -> None:
        # Tool calls requested -> routes to tools
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": "search_child_chunks", "args": {"query": "q"}, "id": "1"}],
        )
        state_tools: AgentState = {"messages": [tool_call_msg]}
        assert route_after_orchestrator_call(state_tools, self_correction_enabled=True) == "tools"

        # No tool calls, self_correction enabled -> routes to grade_answer
        plain_msg = AIMessage(content="Candidate answer", tool_calls=[])
        state_answer: AgentState = {"messages": [plain_msg]}
        assert (
            route_after_orchestrator_call(state_answer, self_correction_enabled=True)
            == "grade_answer"
        )

        # No tool calls, self_correction disabled -> routes to collect_answer
        assert (
            route_after_orchestrator_call(state_answer, self_correction_enabled=False)
            == "collect_answer"
        )


class TestSelfCorrectionSubgraphIntegration:
    """Integration tests for build_agent_subgraph with self-correction enabled."""

    def test_subgraph_successful_on_first_pass(self) -> None:
        @tool
        def search_child_chunks(query: str) -> str:
            """Dummy search tool."""
            payload = [{"parent_id": "p1", "source": "guide.md", "content": "Evidence text."}]
            return json.dumps(payload)

        tool_msg = AIMessage(
            content="",
            tool_calls=[{"name": "search_child_chunks", "args": {"query": "test"}, "id": "t1"}],
        )
        final_msg = AIMessage(content="Verified answer from guide.md.\n\nSources:\n- guide.md")

        fake_orchestrator = GenericFakeChatModel(messages=iter([tool_msg, final_msg]))

        # Fake grader LLM: returns 'yes' for both evidence and answer
        mock_grader = MagicMock()
        mock_struct = MagicMock()
        mock_grader.with_structured_output.return_value = mock_struct
        mock_struct.invoke.side_effect = [
            GradeDocuments(binary_score="yes", explanation="Evidence is relevant."),
            GradeAnswer(binary_score="yes", explanation="Answer is supported."),
        ]

        graph = build_agent_subgraph(
            llm=fake_orchestrator,
            tools=[search_child_chunks],
            grader_llm=mock_grader,
            self_correction_enabled=True,
            grade_evidence_enabled=True,
            grade_answer_enabled=True,
            max_correction_retries=2,
        )

        out = graph.invoke({"question": "Test question?", "messages": []})
        assert "Verified answer" in out["final_answer"]
        assert out["evidence_score"] == "yes"
        assert out["answer_score"] == "yes"
        # No corrective retries were needed!
        assert out.get("correction_count", 0) == 0

    def test_subgraph_corrective_re_retrieval_loop(self) -> None:
        call_count = 0

        @tool
        def search_child_chunks(query: str) -> str:
            """Dummy search tool returning different payload on subsequent call."""
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps(
                    [{"parent_id": "p1", "source": "doc1.md", "content": "Vague info"}]
                )
            return json.dumps(
                [{"parent_id": "p2", "source": "doc2.md", "content": "Precise answer fact"}]
            )

        tool_call_1 = AIMessage(
            content="",
            tool_calls=[{"name": "search_child_chunks", "args": {"query": "q1"}, "id": "t1"}],
        )
        draft_answer_1 = AIMessage(content="Draft answer from vague info")
        tool_call_2 = AIMessage(
            content="",
            tool_calls=[{"name": "search_child_chunks", "args": {"query": "q2"}, "id": "t2"}],
        )
        final_answer_2 = AIMessage(
            content="Corrected answer based on doc2.md.\n\nSources:\n- doc2.md"
        )

        fake_orchestrator = GenericFakeChatModel(
            messages=iter([tool_call_1, draft_answer_1, tool_call_2, final_answer_2])
        )

        mock_grader = MagicMock()
        mock_struct = MagicMock()
        mock_grader.with_structured_output.return_value = mock_struct
        # Sequence of grading calls:
        # 1. Evidence 1 -> yes
        # 2. Answer 1 -> NO (triggers corrective re-retrieval!)
        # 3. Refine query -> refined query
        # 4. Evidence 2 -> yes
        # 5. Answer 2 -> YES
        mock_struct.invoke.side_effect = [
            GradeDocuments(binary_score="yes", explanation="Ok"),
            GradeAnswer(binary_score="no", explanation="Missing specific fact."),
            RefinedQuery(query="doc2 specific fact", explanation="Better keywords"),
            GradeDocuments(binary_score="yes", explanation="Great evidence"),
            GradeAnswer(binary_score="yes", explanation="Fully supported now."),
        ]

        graph = build_agent_subgraph(
            llm=fake_orchestrator,
            tools=[search_child_chunks],
            grader_llm=mock_grader,
            self_correction_enabled=True,
            grade_evidence_enabled=True,
            grade_answer_enabled=True,
            max_correction_retries=2,
        )

        out = graph.invoke({"question": "Where is the fact?", "messages": []})
        assert "Corrected answer based on doc2.md" in out["final_answer"]
        assert out["answer_score"] == "yes"
        # Successfully self-corrected with 1 retry!
        assert out["correction_count"] == 1

    def test_subgraph_budget_exhaustion_routes_to_fallback(self) -> None:
        @tool
        def search_child_chunks(query: str) -> str:
            """Dummy search tool."""
            return json.dumps([{"parent_id": "p1", "source": "doc.md", "content": "Facts"}])

        tool_call_1 = AIMessage(
            content="",
            tool_calls=[{"name": "search_child_chunks", "args": {"query": "q1"}, "id": "t1"}],
        )
        draft_answer_1 = AIMessage(content="Unsatisfactory draft answer")
        fallback_msg = AIMessage(content="Synthesized fallback answer.")

        fake_orchestrator = GenericFakeChatModel(
            messages=iter([tool_call_1, draft_answer_1, fallback_msg])
        )

        mock_grader = MagicMock()
        mock_struct = MagicMock()
        mock_grader.with_structured_output.return_value = mock_struct
        mock_struct.invoke.side_effect = [
            GradeDocuments(binary_score="yes", explanation="Ok"),
            GradeAnswer(binary_score="no", explanation="Unsupported claims."),
        ]

        # max_correction_retries=0 ensures immediate fallback on grading rejection
        graph = build_agent_subgraph(
            llm=fake_orchestrator,
            tools=[search_child_chunks],
            grader_llm=mock_grader,
            self_correction_enabled=True,
            max_correction_retries=0,
        )

        out = graph.invoke({"question": "Tough question?", "messages": []})
        assert out["final_answer"] == "Synthesized fallback answer."
        assert out["answer_score"] == "no"


class TestRAGSystemSelfCorrectionConfiguration:
    """Test self-correction wiring inside RAGSystem composition root."""

    def test_ragsystem_respects_self_correction_settings(self) -> None:
        settings = Settings(
            self_correction_enabled=True,
            max_correction_retries=3,
        )
        mock_llm = MagicMock()
        mock_grader = MagicMock()

        @tool
        def dummy_search(query: str) -> str:
            """Dummy search tool."""
            return "ok"

        system = RAGSystem(
            settings=settings,
            llm=mock_llm,
            grader_llm=mock_grader,
            tools=[dummy_search],
        )

        assert system.settings.self_correction_enabled is True
        assert system.settings.max_correction_retries == 3
        assert system.grader_llm is mock_grader
