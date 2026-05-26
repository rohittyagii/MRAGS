from __future__ import annotations

import hashlib
from typing import Any

import tiktoken

from mrags.models import Modality, ProcessedElement

from .base import ElementProcessor


class TextProcessor(ElementProcessor):
    def __init__(self, chunk_tokens: int, chunk_overlap: int) -> None:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self._splitter = TokenTextSplitter(chunk_tokens, chunk_overlap)

    async def process(self, element: Any) -> list[ProcessedElement]:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        text = getattr(element, "text", "") or ""
        chunks = self._splitter.split_text(text)
        element_id_base = _stable_element_id(text)
        metadata = _extract_metadata(element)
        processed: list[ProcessedElement] = []
        for index, chunk in enumerate(chunks):
            processed.append(
                ProcessedElement(
                    element_id=f"{element_id_base}:{index}",
                    modality=Modality.TEXT,
                    raw_content=chunk,
                    embedded_summary=chunk,
                    metadata=metadata,
                )
            )
        return processed


class TokenTextSplitter:
    def __init__(self, chunk_tokens: int, chunk_overlap: int) -> None:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self._chunk_tokens = chunk_tokens
        self._chunk_overlap = chunk_overlap
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def split_text(self, text: str) -> list[str]:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        if not text.strip():
            return []
        tokens = self._encoder.encode(text)
        if not tokens:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + self._chunk_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self._encoder.decode(chunk_tokens))
            if end == len(tokens):
                break
            start = max(end - self._chunk_overlap, 0)
        return chunks


def _stable_element_id(text: str) -> str:
    """Time Complexity: O(N)
    Space Complexity: O(1)
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"text:{digest}"


def _extract_metadata(element: Any) -> dict[str, str]:
    """Time Complexity: O(1)
    Space Complexity: O(1)
    """
    metadata = getattr(element, "metadata", None)
    if metadata is None:
        return {}
    page_number = getattr(metadata, "page_number", None)
    if page_number is None:
        return {}
    return {"page_number": str(page_number)}
