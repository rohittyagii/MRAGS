# MRAGS Context Tree

## Root
- README.md: Usage and setup notes, including local embeddings and local LMM.
- pyproject.toml: Python package config and dependencies.
- .env.example: Default env vars for offline/local operation.
- src/mrags: Main package source.
- tests: Smoke test coverage.
- docs: Project documentation (this file, handoff).

## src/mrags
- cli.py: Typer CLI entrypoint (ingest, query, validate-lmm).
- config.py: AppSettings and environment parsing.
- errors.py: Custom exceptions.
- models.py: Core data models (ProcessedElement, Modality).
- generation:
  - llm_client.py: OpenAI LMM client and LocalLMMClient (llama-cpp).
  - prompt_builder.py: Local prompt builder and OpenAI messages.
- ingestion:
  - base.py: ElementProcessor ABC and IngestionPipeline.
  - router.py: ElementRouter for routing unstructured elements.
  - text_processor.py: Text chunking processor.
  - table_processor.py: Table handling processor.
  - image_processor.py: Image handling processor.
  - vlm_client.py: OpenAI VLM client + NoopVLMClient.
- retrieval:
  - retriever.py: Retrieval pipeline from embeddings + FAISS + SQLite.
- storage:
  - embeddings.py: Embeddings clients (OpenAI + local sentence-transformers).
  - faiss_index.py: FAISS index wrapper.
  - sqlite_kv.py: SQLite content store for elements and metadata.

## tests
- test_smoke.py: Basic import and CLI smoke validation.
