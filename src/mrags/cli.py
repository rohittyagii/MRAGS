from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import aiohttp
import typer
from openai import AsyncOpenAI
from rich.console import Console
from rich.table import Table
from unstructured.partition.pdf import partition_pdf

from mrags.config import AppSettings
from mrags.errors import PDFExtractionError
from mrags.generation.llm_client import LMMClient, LocalLMMClient
from mrags.ingestion.base import IngestionPipeline
from mrags.ingestion.image_processor import ImageProcessor
from mrags.ingestion.router import ElementRouter
from mrags.ingestion.table_processor import TableProcessor
from mrags.ingestion.text_processor import TextProcessor
from mrags.ingestion.vlm_client import NoopVLMClient, OpenAIVLMClient
from mrags.models import Modality
from mrags.retrieval.retriever import Retriever
from mrags.storage.embeddings import EmbeddingsClient, LocalEmbeddingsClient
from mrags.storage.faiss_index import FaissIndex
from mrags.storage.sqlite_kv import SQLiteKVStore

app = typer.Typer()
console = Console()


@app.command()
def ingest(pdf_path: str) -> None:
    """Time Complexity: O(N)
    Space Complexity: O(N)
    """
    asyncio.run(_ingest_async(pdf_path))


@app.command()
def query(question: str) -> None:
    """Time Complexity: O(N)
    Space Complexity: O(N)
    """
    asyncio.run(_query_async(question))


@app.command("validate-lmm")
def validate_lmm() -> None:
    """Time Complexity: O(1)
    Space Complexity: O(1)
    """
    settings = AppSettings.from_env()
    _configure_logging(settings.log_level)
    _validate_local_lmm(settings)
    console.print(
        "Local LMM is ready. "
        f"Model: {settings.lmm_model_path} | "
        f"n_ctx={settings.lmm_n_ctx} | "
        f"n_gpu_layers={settings.lmm_n_gpu_layers}"
    )


async def _ingest_async(pdf_path: str) -> None:
    """Time Complexity: O(N)
    Space Complexity: O(N)
    """
    settings = AppSettings.from_env()
    _configure_logging(settings.log_level)
    if not Path(pdf_path).exists():
        raise PDFExtractionError(f"PDF not found: {pdf_path}")

    # 1) Break the PDF into raw elements (text blocks, tables, images).
    raw_elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True,
    )
    async with aiohttp.ClientSession() as session:
        router = _build_router(settings, session)
        pipeline = IngestionPipeline(router, logging.getLogger("mrags.ingest"))
        # 2) Convert raw elements into processed chunks and summaries.
        processed_elements = await pipeline.process(raw_elements)
        if not processed_elements:
            console.print("No elements extracted.")
            return
        # 3) Embed the processed summaries.
        embedding_client = _build_embedder(settings)
        embeddings = await embedding_client.embed_texts(
            [element.embedded_summary for element in processed_elements]
        )
        index = FaissIndex(settings.faiss_index_path)
        index.load_or_create(dimension=len(embeddings[0]))
        vector_ids = index.add(embeddings)
        index.save()

        with SQLiteKVStore(settings.sqlite_path) as store:
            # 4) Persist both raw content and vector metadata.
            store.put_elements(processed_elements)
            store.put_vector_metadata(
                vector_ids,
                [element.element_id for element in processed_elements],
                [element.modality for element in processed_elements],
            )

        console.print(f"Ingested {len(processed_elements)} elements.")


async def _query_async(question: str) -> None:
    """Time Complexity: O(N)
    Space Complexity: O(N)
    """
    settings = AppSettings.from_env()
    _configure_logging(settings.log_level)

    # 1) Load embedding client and index.
    embedding_client = _build_embedder(settings)
    index = FaissIndex(settings.faiss_index_path)
    index.load_existing()

    with SQLiteKVStore(settings.sqlite_path) as store:
        # 2) Pull the top-k most relevant elements.
        retriever = Retriever(embedding_client, index, store)
        elements = await retriever.retrieve(question, settings.top_k)
        if not settings.enable_lmm:
            _render_answer("LMM disabled; showing retrieved context only.", elements)
            return
        # 3) Ask the LMM to answer using only retrieved context.
        lmm_client = _build_lmm_client(settings)
        answer = await lmm_client.answer(question, elements)
        _render_answer(answer.answer, elements)


def _build_router(settings: AppSettings, session: aiohttp.ClientSession) -> ElementRouter:
    """Time Complexity: O(1)
    Space Complexity: O(1)
    """
    text_processor = TextProcessor(settings.chunk_tokens, settings.chunk_overlap)
    table_processor = TableProcessor()

    if settings.enable_vlm:
        semaphore = asyncio.Semaphore(settings.max_concurrency)
        vlm_client = OpenAIVLMClient(
            api_key=_require_openai_key(settings),
            model=settings.vlm_model,
            prompt=settings.image_summary_prompt,
            timeout_s=settings.request_timeout_s,
            semaphore=semaphore,
            session=session,
        )
    else:
        vlm_client = NoopVLMClient()
    image_processor = ImageProcessor(vlm_client)

    # Use readable names for the element types emitted by unstructured.
    processors = {
        "text": text_processor,
        "title": text_processor,
        "narrativetext": text_processor,
        "listitem": text_processor,
        "table": table_processor,
        "image": image_processor,
        "figure": image_processor,
    }
    return ElementRouter(processors, default_processor=text_processor)


def _configure_logging(level: str) -> None:
    """Time Complexity: O(1)
    Space Complexity: O(1)
    """
    logging.basicConfig(level=level)


def _build_lmm_client(settings: AppSettings):
    """Time Complexity: O(1)
    Space Complexity: O(1)
    """
    system_prompt = (
        "You are a specialized analytical engine. Answer the user using only "
        "the provided text, tables, and images. Do not hallucinate." 
    )
    if settings.lmm_backend.lower() == "local":
        if not settings.lmm_model_path:
            raise PDFExtractionError("LMM_MODEL_PATH is required for local LMM")
        return LocalLMMClient(
            model_path=settings.lmm_model_path,
            system_prompt=system_prompt,
            max_tokens=settings.lmm_max_tokens,
            temperature=settings.lmm_temperature,
            n_ctx=settings.lmm_n_ctx,
            n_gpu_layers=settings.lmm_n_gpu_layers,
            n_threads=settings.lmm_n_threads,
        )
    async_client = _require_openai_client(settings)
    return LMMClient(async_client, settings.lmm_model, system_prompt)


def _validate_local_lmm(settings: AppSettings) -> None:
    """Time Complexity: O(1)
    Space Complexity: O(1)
    """
    if settings.lmm_backend.lower() != "local":
        raise PDFExtractionError("LMM_BACKEND must be 'local' for validation")
    model_path = Path(settings.lmm_model_path)
    if not model_path.exists():
        raise PDFExtractionError(f"LMM model not found: {model_path}")
    _build_lmm_client(settings)


def _build_embedder(settings: AppSettings):
    """Time Complexity: O(1)
    Space Complexity: O(1)
    """
    if settings.embedding_backend.lower() == "local":
        return LocalEmbeddingsClient(settings.local_embedding_model)
    async_client = _require_openai_client(settings)
    return EmbeddingsClient(
        client=async_client,
        model=settings.embedding_model,
        max_concurrency=settings.max_concurrency,
        timeout_s=settings.request_timeout_s,
    )


def _require_openai_key(settings: AppSettings) -> str:
    """Time Complexity: O(1)
    Space Complexity: O(1)
    """
    if not settings.openai_api_key:
        raise PDFExtractionError("OPENAI_API_KEY is required for OpenAI calls")
    return settings.openai_api_key


def _require_openai_client(settings: AppSettings) -> AsyncOpenAI:
    """Time Complexity: O(1)
    Space Complexity: O(1)
    """
    return AsyncOpenAI(api_key=_require_openai_key(settings))


def _render_answer(answer: str, elements) -> None:
    """Time Complexity: O(N)
    Space Complexity: O(N)
    """
    console.print("\nAnswer:\n")
    console.print(answer)
    table = Table(title="Retrieved Context")
    table.add_column("ID")
    table.add_column("Modality")
    table.add_column("Score")
    for element in elements:
        table.add_row(element.element_id, element.modality.value, f"{element.score:.3f}")
    console.print(table)


def main() -> None:
    """Time Complexity: O(1)
    Space Complexity: O(1)
    """
    app()


if __name__ == "__main__":
    main()
