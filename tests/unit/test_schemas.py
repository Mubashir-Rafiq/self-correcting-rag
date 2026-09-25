"""Unit tests for agent Pydantic schemas and structured output models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from self_rag.agent.schemas import (
    GradeAnswer,
    GradeDocuments,
    QueryAnalysis,
    SearchResultItem,
)


class TestSchemas:
    def test_query_analysis_clear(self) -> None:
        analysis = QueryAnalysis(
            is_clear=True,
            questions=["What is hybrid search in Qdrant?"],
        )
        assert analysis.is_clear is True
        assert analysis.questions == ["What is hybrid search in Qdrant?"]
        assert analysis.clarification_needed == ""

    def test_query_analysis_unclear(self) -> None:
        analysis = QueryAnalysis(
            is_clear=False,
            clarification_needed="Please specify which document you mean by 'it'.",
        )
        assert analysis.is_clear is False
        assert analysis.questions == []
        assert "Please specify" in analysis.clarification_needed

    def test_search_result_item(self) -> None:
        item = SearchResultItem(
            parent_id="manual_p0",
            source="manual.pdf",
            content="Sample extracted chunk text.",
        )
        assert item.parent_id == "manual_p0"
        assert item.source == "manual.pdf"
        assert "Sample" in item.content

    def test_grade_documents(self) -> None:
        grade = GradeDocuments(binary_score="yes", explanation="Directly answers query.")
        assert grade.binary_score == "yes"

        with pytest.raises(ValidationError):
            GradeDocuments()  # type: ignore[call-arg]

    def test_grade_answer(self) -> None:
        grade = GradeAnswer(binary_score="no", explanation="Answer is hallucinated.")
        assert grade.binary_score == "no"
