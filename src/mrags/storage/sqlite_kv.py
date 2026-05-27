from __future__ import annotations

import json
import sqlite3

from mrags.config import ensure_parent_dir
from mrags.errors import StorageError
from mrags.models import Modality, ProcessedElement


class SQLiteKVStore:
    def __init__(self, sqlite_path: str) -> None:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self._sqlite_path = sqlite_path
        ensure_parent_dir(sqlite_path)
        self._conn = sqlite3.connect(sqlite_path)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS elements (
                element_id TEXT PRIMARY KEY,
                modality TEXT NOT NULL,
                raw_content TEXT NOT NULL,
                embedded_summary TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_meta (
                vector_id INTEGER PRIMARY KEY,
                element_id TEXT NOT NULL,
                modality TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def put_elements(self, elements: list[ProcessedElement]) -> None:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        cursor = self._conn.cursor()
        rows = [
            (
                element.element_id,
                element.modality.value,
                element.raw_content,
                element.embedded_summary,
                json.dumps(element.metadata),
            )
            for element in elements
        ]
        cursor.executemany(
            """
            INSERT OR REPLACE INTO elements (
                element_id, modality, raw_content, embedded_summary, metadata
            ) VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()

    def put_vector_metadata(
        self, vector_ids: list[int], element_ids: list[str], modalities: list[Modality]
    ) -> None:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        if len(vector_ids) != len(element_ids) or len(vector_ids) != len(modalities):
            raise StorageError("Vector metadata lengths must match")
        cursor = self._conn.cursor()
        rows = [
            (vector_id, element_id, modality.value)
            for vector_id, element_id, modality in zip(vector_ids, element_ids, modalities)
        ]
        cursor.executemany(
            """
            INSERT OR REPLACE INTO vector_meta (vector_id, element_id, modality)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()

    def get_elements(self, element_ids: list[str]) -> list[ProcessedElement]:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        if not element_ids:
            return []
        placeholders = ",".join(["?"] * len(element_ids))
        query = f"SELECT * FROM elements WHERE element_id IN ({placeholders})"
        cursor = self._conn.execute(query, element_ids)
        rows = cursor.fetchall()
        return [
            ProcessedElement(
                element_id=row["element_id"],
                modality=Modality(row["modality"]),
                raw_content=row["raw_content"],
                embedded_summary=row["embedded_summary"],
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]

    def count_elements(self) -> int:
        """Return the number of stored elements in the `elements` table.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        cursor = self._conn.execute("SELECT COUNT(1) as c FROM elements")
        row = cursor.fetchone()
        return int(row["c"]) if row is not None else 0

    def count_vector_meta(self) -> int:
        """Return the number of rows in the `vector_meta` table.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        cursor = self._conn.execute("SELECT COUNT(1) as c FROM vector_meta")
        row = cursor.fetchone()
        return int(row["c"]) if row is not None else 0

    def get_vector_metadata(self, vector_ids: list[int]) -> list[tuple[int, str, Modality]]:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        if not vector_ids:
            return []
        placeholders = ",".join(["?"] * len(vector_ids))
        query = f"SELECT * FROM vector_meta WHERE vector_id IN ({placeholders})"
        cursor = self._conn.execute(query, vector_ids)
        rows = cursor.fetchall()
        return [
            (row["vector_id"], row["element_id"], Modality(row["modality"]))
            for row in rows
        ]

    def close(self) -> None:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self._conn.close()

    def __enter__(self) -> "SQLiteKVStore":
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.close()
