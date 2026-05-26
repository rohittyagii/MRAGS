from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Modality(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class ProcessedElement(BaseModel):
    element_id: str
    modality: Modality
    raw_content: str
    embedded_summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedElement(BaseModel):
    element_id: str
    modality: Modality
    raw_content: str
    score: float
    embedded_summary: str | None = None


class LMMAnswer(BaseModel):
    answer: str
    sources: list[RetrievedElement]
