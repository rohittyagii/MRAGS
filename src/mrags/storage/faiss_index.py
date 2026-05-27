from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from mrags.config import ensure_parent_dir
from mrags.errors import StorageError


class FaissIndex:
    def __init__(self, index_path: str) -> None:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self._index_path = index_path
        self._index: faiss.Index | None = None

    def load_or_create(self, dimension: int) -> None:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        path = Path(self._index_path)
        if path.exists():
            self._index = faiss.read_index(str(path))
            return
        self._index = faiss.IndexFlatIP(dimension)

    def load_existing(self) -> None:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        path = Path(self._index_path)
        if not path.exists():
            raise StorageError("FAISS index file not found")
        self._index = faiss.read_index(str(path))

    def add(self, vectors: list[list[float]]) -> list[int]:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        if self._index is None:
            raise StorageError("FAISS index is not initialized")
        if not vectors:
            return []
        array = np.asarray(vectors, dtype="float32")
        array = _normalize(array)
        start_id = self._index.ntotal
        self._index.add(array)
        return list(range(start_id, start_id + len(vectors)))

    def search(self, query_vector: list[float], top_k: int) -> tuple[list[int], list[float]]:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        if self._index is None:
            raise StorageError("FAISS index is not initialized")
        query = np.asarray([query_vector], dtype="float32")
        query = _normalize(query)
        scores, ids = self._index.search(query, top_k)
        return ids[0].tolist(), scores[0].tolist()

    def save(self) -> None:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self._index is None:
            raise StorageError("FAISS index is not initialized")
        ensure_parent_dir(self._index_path)
        faiss.write_index(self._index, self._index_path)

    def exists(self) -> bool:
        """Return True if the FAISS index file exists on disk.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return Path(self._index_path).exists()

    def count(self) -> int:
        """Return the number of vectors stored in the loaded index.

        Raises StorageError if the index is not loaded.
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self._index is None:
            raise StorageError("FAISS index is not initialized")
        return int(self._index.ntotal)

    def dimension(self) -> int:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if self._index is None:
            raise StorageError("FAISS index is not initialized")
        return self._index.d


def _normalize(array: np.ndarray) -> np.ndarray:
    """Time Complexity: O(N)
    Space Complexity: O(N)
    """
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms
