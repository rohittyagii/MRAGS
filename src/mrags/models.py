"""Core data models used across MRAGS.

These lightweight Pydantic models represent processed elements, retrieval
results, and the answer returned from the language model.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Modality(str, Enum):
    """A small enum describing the type of a document element."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class ProcessedElement(BaseModel):
    """A single chunk produced by the ingestion pipeline.

    Contains the original text, a short embedded summary, and optional
    metadata useful for display or filtering.
    """
    element_id: str
    modality: Modality
    raw_content: str
    embedded_summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedElement(BaseModel):
    """A retrieved item returned by the Retriever, with a similarity score."""
    element_id: str
    modality: Modality
    raw_content: str
    score: float
    embedded_summary: str | None = None


class LMMAnswer(BaseModel):
    """The LMM's generated answer paired with the source elements used."""
    answer: str
    sources: list[RetrievedElement]
