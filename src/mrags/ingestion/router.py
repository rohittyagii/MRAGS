from __future__ import annotations

from typing import Any

from mrags.errors import PDFExtractionError

from .base import ElementProcessor


class ElementRouter:
    def __init__(
        self,
        processors_by_category: dict[str, ElementProcessor],
        default_processor: ElementProcessor | None = None,
    ) -> None:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self._processors_by_category = processors_by_category
        self._default_processor = default_processor

    def route(self, element: Any) -> ElementProcessor:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        category = _category_from_element(element)
        processor = self._processors_by_category.get(category)
        if processor is None:
            if self._default_processor is None:
                raise PDFExtractionError(f"No processor for category: {category}")
            return self._default_processor
        return processor


def _category_from_element(element: Any) -> str:
    """Time Complexity: O(1)
    Space Complexity: O(1)
    """
    category = getattr(element, "category", "")
    if not category:
        category = element.__class__.__name__
    return str(category).strip().lower()
