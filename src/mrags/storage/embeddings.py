from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from typing import Iterable

from mrags.errors import EmbeddingError

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class EmbeddingsClient:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        max_concurrency: int,
        timeout_s: int,
    ) -> None:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self._client = client
        self._model = model
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._timeout_s = timeout_s

    async def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        batches = _batch(texts, batch_size)
        tasks = [self._embed_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        vectors: list[list[float]] = []
        for result in results:
            if isinstance(result, Exception):
                raise EmbeddingError(str(result)) from result
            vectors.extend(result)
        return vectors

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        async with self._semaphore:
            response = await self._client.embeddings.create(
                model=self._model,
                input=batch,
                timeout=self._timeout_s,
            )
            return [item.embedding for item in response.data]


class LocalEmbeddingsClient:
    def __init__(self, model_name: str) -> None:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    async def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        if not texts:
            return []
        return await asyncio.to_thread(
            self._embed_sync,
            texts,
            batch_size,
        )

    def _embed_sync(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        return embeddings.tolist()


def _batch(items: Iterable[str], size: int) -> list[list[str]]:
    """Time Complexity: O(N)
    Space Complexity: O(N)
    """
    batch: list[str] = []
    batches: list[list[str]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches
