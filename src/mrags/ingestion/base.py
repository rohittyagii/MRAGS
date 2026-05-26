from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Iterable, TYPE_CHECKING

from mrags.models import ProcessedElement

if TYPE_CHECKING:
    from .router import ElementRouter


class ElementProcessor(ABC):
    @abstractmethod
    async def process(self, element: Any) -> list[ProcessedElement]:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        raise NotImplementedError


class IngestionPipeline:
    def __init__(self, router: "ElementRouter", logger: logging.Logger | None = None) -> None:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self._router = router
        self._logger = logger or logging.getLogger(__name__)

    async def process(self, elements: Iterable[Any]) -> list[ProcessedElement]:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        tasks = [self._process_one(element) for element in elements]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed: list[ProcessedElement] = []
        for result in results:
            if isinstance(result, Exception):
                self._logger.warning("Element processing failed: %s", result)
                continue
            processed.extend(result)
        return processed

    async def _process_one(self, element: Any) -> list[ProcessedElement]:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        processor = self._router.route(element)
        return await processor.process(element)
