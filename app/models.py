"""Data models for the LLM support assistant."""

from pydantic import BaseModel, Field, validator
from typing import Optional

class QuestionRequest(BaseModel):
    """Model for question request."""
    question: str = Field(..., description="The question to ask the LLM")

    @validator("question")
    def validate_question(cls, v):
        """Validate that question is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Question cannot be empty or whitespace only")
        if len(v) > 2000:
            raise ValueError("Question cannot exceed 2000 characters")
        return v.strip()

class AnswerResponse(BaseModel):
    """Model for answer response."""
    question: str
    answer: str