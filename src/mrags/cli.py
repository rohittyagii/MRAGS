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
from mrags.models import Modality, RetrievedElement
from mrags.retrieval.retriever import Retriever
from mrags.storage.embeddings import EmbeddingsClient, LocalEmbeddingsClient
from mrags.storage.faiss_index import FaissIndex
from mrags.storage.sqlite_kv import SQLiteKVStore

app = typer.Typer()
console = Console()


@app.command()
def ingest(pdf_path: str) -> None:
    """Ingest a PDF into the local index and database.
    Time Complexity: O(N)
    Space Complexity: O(N)
    """
    asyncio.run(_ingest_async(pdf_path))


@app.command()
def query(question: str) -> None:
    """Run a retrieval (and optional local LMM) query against the index.
    Time Complexity: O(N)
    Space Complexity: O(N)
    """
    asyncio.run(_query_async(question))


@app.command("validate-lmm")
def validate_lmm() -> None:
    """Quick local LMM sanity check — confirms the configured GGUF loads.
    Time Complexity: O(1)
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
            console.print("No content was extracted from the PDF. Check the file and try again.")
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
    """Ask a question: retrieves relevant chunks and (optionally) runs the LMM.
    Time Complexity: O(N)
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
            _render_context_only(elements)
            return
        # 3) Ask the LMM to answer using only retrieved context.
        lmm_client = _build_lmm_client(settings)
        answer = await lmm_client.answer(question, elements)
        _render_answer(answer.answer, elements)


def _build_router(settings: AppSettings, session: aiohttp.ClientSession) -> ElementRouter:
    """Create processors and the router that maps PDF elements to handlers.
    Time Complexity: O(1)
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
    """Configure basic logging for the CLI.
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    logging.basicConfig(level=level)


def _build_lmm_client(settings: AppSettings):
    """Return a configured LMM client (local or OpenAI-backed).
    Time Complexity: O(1)
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
    """Ensure a local LMM backend is configured and the model exists.
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    if settings.lmm_backend.lower() != "local":
        raise PDFExtractionError("LMM_BACKEND must be 'local' for validation")
    model_path = Path(settings.lmm_model_path)
    if not model_path.exists():
        raise PDFExtractionError(f"LMM model not found: {model_path}")
    _build_lmm_client(settings)


def _build_embedder(settings: AppSettings):
    """Return the embeddings client based on settings (local or OpenAI).
    Time Complexity: O(1)
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
    """Read and validate the `OPENAI_API_KEY` setting from the environment.
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    if not settings.openai_api_key:
        raise PDFExtractionError("OPENAI_API_KEY is required for OpenAI calls")
    return settings.openai_api_key


def _require_openai_client(settings: AppSettings) -> AsyncOpenAI:
    """Create an `AsyncOpenAI` client using the configured API key.
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    return AsyncOpenAI(api_key=_require_openai_key(settings))


def _render_answer(answer: str, elements: list[RetrievedElement]) -> None:
    """Print the model's answer and the retrieved context used to form it.
    Time Complexity: O(N)
    Space Complexity: O(N)
    """
    console.print("\nAnswer:\n")
    console.print(answer)
    _render_retrieved_context(elements)


def _render_context_only(elements: list[RetrievedElement]) -> None:
    """Show the retrieved pieces of text when the LMM is turned off.
    Time Complexity: O(N)
    Space Complexity: O(N)
    """
    console.print("\nLMM is disabled. Showing retrieved context only:\n")
    _render_retrieved_context(elements)


def _render_retrieved_context(elements: list[RetrievedElement]) -> None:
    """Render a table with the top matching elements and a short preview.
    Time Complexity: O(N)
    Space Complexity: O(N)
    """
    table = Table(title="Retrieved Context (Top Matches)")
    table.add_column("ID", overflow="fold")
    table.add_column("Modality", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Preview", overflow="fold")
    for element in elements:
        preview = _preview_text(element.raw_content, max_len=180)
        table.add_row(
            element.element_id,
            element.modality.value,
            f"{element.score:.3f}",
            preview,
        )
    console.print(table)


def _preview_text(text: str, max_len: int = 160) -> str:
    """Return a short, cleaned preview of the input text for CLI display.
    Time Complexity: O(N)
    Space Complexity: O(N)
    """
    normalized = " ".join(text.split())
    if len(normalized) <= max_len:
        return normalized
    return f"{normalized[: max_len - 3]}..."


def main() -> None:
    """Time Complexity: O(1)
    Space Complexity: O(1)
    """
    app()


if __name__ == "__main__":
    main()
