from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from mrags.models import Modality, ProcessedElement

from .base import ElementProcessor


class TableProcessor(ElementProcessor):
    async def process(self, element: Any) -> list[ProcessedElement]:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        table_markdown = _table_to_markdown(element)
        element_id = _stable_element_id(table_markdown)
        metadata = _extract_metadata(element)
        return [
            ProcessedElement(
                element_id=element_id,
                modality=Modality.TABLE,
                raw_content=table_markdown,
                embedded_summary=table_markdown,
                metadata=metadata,
            )
        ]


def _table_to_markdown(element: Any) -> str:
    """Time Complexity: O(N)
    Space Complexity: O(N)
    """
    metadata = getattr(element, "metadata", None)
    html = getattr(metadata, "text_as_html", None) if metadata else None
    if html:
        try:
            tables = pd.read_html(html)
            if tables:
                return tables[0].to_markdown(index=False)
        except ValueError:
            pass
    text = getattr(element, "text", "") or ""
    return text.strip()


def _stable_element_id(text: str) -> str:
    """Time Complexity: O(N)
    Space Complexity: O(1)
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"table:{digest}"


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
