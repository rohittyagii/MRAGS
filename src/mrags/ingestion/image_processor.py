from __future__ import annotations

import hashlib
from typing import Any

from mrags.errors import PDFExtractionError
from mrags.models import Modality, ProcessedElement

from .base import ElementProcessor
from .vlm_client import VLMClient


class ImageProcessor(ElementProcessor):
    def __init__(self, vlm_client: VLMClient) -> None:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self._vlm_client = vlm_client

    async def process(self, element: Any) -> list[ProcessedElement]:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        base64_image = _extract_base64(element)
        summary = await self._vlm_client.describe_image(base64_image)
        element_id = _stable_element_id(base64_image)
        metadata = _extract_metadata(element)
        return [
            ProcessedElement(
                element_id=element_id,
                modality=Modality.IMAGE,
                raw_content=base64_image,
                embedded_summary=summary,
                metadata=metadata,
            )
        ]


def _extract_base64(element: Any) -> str:
    """Time Complexity: O(1)
    Space Complexity: O(1)
    """
    metadata = getattr(element, "metadata", None)
    base64_image = None
    if metadata is not None:
        base64_image = getattr(metadata, "image_base64", None)
    if not base64_image:
        raise PDFExtractionError("Missing base64 image content")
    return base64_image


def _stable_element_id(text: str) -> str:
    """Time Complexity: O(N)
    Space Complexity: O(1)
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"image:{digest}"


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
