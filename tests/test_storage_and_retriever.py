import asyncio

from mrags.models import ProcessedElement, Modality
from mrags.storage.faiss_index import FaissIndex
from mrags.storage.sqlite_kv import SQLiteKVStore
from mrags.retrieval.retriever import Retriever


def test_storage_and_retriever_integration(tmp_path):
    """Integration test: ensure FAISS + SQLite + Retriever work together.

    This test avoids heavy external dependencies by using tiny fixed vectors
    and a minimal async embedder that returns a query vector matching the
    first inserted element.
    """
    faiss_path = str(tmp_path / "faiss.index")
    sqlite_path = str(tmp_path / "mrags.sqlite")

    # Prepare three orthogonal unit vectors (dim=3)
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    elements = [
        ProcessedElement(
            element_id=f"e{i}",
            modality=Modality.TEXT,
            raw_content=f"Content {i}",
            embedded_summary=f"Summary {i}",
            metadata={},
        )
        for i in range(3)
    ]

    # Create and populate FAISS index
    index = FaissIndex(faiss_path)
    index.load_or_create(dimension=3)
    ids = index.add(vectors)
    index.save()

    # Persist elements and vector metadata
    with SQLiteKVStore(sqlite_path) as store:
        store.put_elements(elements)
        store.put_vector_metadata(ids, [e.element_id for e in elements], [e.modality for e in elements])

    # Dummy embedder that returns vector matching the first element
    class DummyEmbedder:
        async def embed_texts(self, texts):
            return [[1.0, 0.0, 0.0]]

    embedder = DummyEmbedder()

    # Load index and run retriever
    index2 = FaissIndex(faiss_path)
    index2.load_existing()

    with SQLiteKVStore(sqlite_path) as store2:
        # Basic sanity checks for counts
        assert store2.count_elements() == 3
        assert store2.count_vector_meta() == 3
        assert index2.count() == 3

        retriever = Retriever(embedder, index2, store2)
        results = asyncio.run(retriever.retrieve("query text", top_k=2))

        # Expect the first element (matching the query vector) to be highest-ranked
        assert len(results) >= 1
        assert results[0].element_id == "e0"
        assert all(hasattr(r, "score") for r in results)
