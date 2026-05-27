from __future__ import annotations

from mrags.models import RetrievedElement
from mrags.storage.embeddings import EmbeddingsClient
from mrags.storage.faiss_index import FaissIndex
from mrags.storage.sqlite_kv import SQLiteKVStore


class Retriever:
    """Finds the most relevant document pieces for a query using embeddings.

    The Retriever coordinates the embedder, FAISS index, and SQLite store to
    return `RetrievedElement` objects ranked by similarity.
    """

    def __init__(
        self,
        embedder: EmbeddingsClient,
        index: FaissIndex,
        store: SQLiteKVStore,
    ) -> None:
        """Create a retriever with the necessary storage and embedding clients.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self._embedder = embedder
        self._index = index
        self._store = store

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedElement]:
        """Return the top-k most relevant elements for `query`.

        Time Complexity: O(N)
        Space Complexity: O(N)
        """
        vectors = await self._embedder.embed_texts([query])
        ids, scores = self._index.search(vectors[0], top_k)
        metadata = self._store.get_vector_metadata(ids)
        meta_map = {vector_id: (element_id, modality) for vector_id, element_id, modality in metadata}
        element_ids = [element_id for element_id, _ in meta_map.values()]
        elements = {element.element_id: element for element in self._store.get_elements(element_ids)}
        score_map = {vector_id: score for vector_id, score in zip(ids, scores)}
        results: list[RetrievedElement] = []
        for vector_id in ids:
            meta = meta_map.get(vector_id)
            if meta is None:
                continue
            element_id, modality = meta
            element = elements.get(element_id)
            if element is None:
                continue
            score = score_map.get(vector_id, 0.0)
            results.append(
                RetrievedElement(
                    element_id=element_id,
                    modality=modality,
                    raw_content=element.raw_content,
                    embedded_summary=element.embedded_summary,
                    score=score,
                )
            )
        return results
