"""Pydantic schemas and structured output models for agent nodes and grading."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """Structured output for query analysis and rewriting."""

    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable.",
    )
    questions: list[str] = Field(
        default_factory=list,
        description="List of rewritten, self-contained questions.",
    )
    clarification_needed: str = Field(
        default="",
        description="Explanation or question to ask the user if the question is unclear.",
    )


class SearchResultItem(BaseModel):
    """Schema for a single retrieved chunk item returned to the agent."""

    parent_id: str = Field(
        description="Unique identifier of the parent document chunk.",
    )
    source: str = Field(
        description="Source document name or path.",
    )
    content: str = Field(
        description="Text content of the retrieved chunk.",
    )


class GradeDocuments(BaseModel):
    """Binary score to grade whether retrieved documents are relevant to a question."""

    binary_score: str = Field(
        description="Whether the document is relevant to the question: 'yes' or 'no'.",
    )
    explanation: str = Field(
        default="",
        description="Brief reasoning for the binary score.",
    )


class GradeAnswer(BaseModel):
    """Binary score to evaluate if an answer is grounded and addresses the question."""

    binary_score: str = Field(
        description="Whether the answer addresses the question: 'yes' or 'no'.",
    )
    explanation: str = Field(
        default="",
        description="Brief reasoning for whether the answer addresses the question.",
    )


class RefinedQuery(BaseModel):
    """Structured output for corrective query refinement."""

    query: str = Field(
        description="Refined search query targeting the missing document evidence.",
    )
    explanation: str = Field(
        default="",
        description="Brief reasoning for how the query was refined.",
    )
