# MRAGS

Enterprise Multimodal RAG System.

## Overview
This project reads complex PDFs, splits them into easy-to-handle parts (text, tables, images),
summarizes images with a vision model when enabled, embeds everything into vectors, and then
retrieves the most relevant pieces to answer questions.

## Quick start
1. Create a virtual environment and install dependencies.
2. Set environment variables from `.env.example`.
3. Ingest a PDF:
   - `mrags ingest path/to/file.pdf`
4. Ask a question:
   - `mrags query "What does the chart show?"`

## Beginner-friendly local run (no API key)
If you want everything to run on your machine without any online API calls:

1. Copy `.env.example` to `.env`.
2. Set these values:
   - `EMBEDDING_BACKEND=local`
   - `LOCAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`
   - `ENABLE_VLM=false` (skips image summaries)
   - `ENABLE_LMM=true` and set `LMM_BACKEND=local`
   - `LMM_MODEL_PATH` to your GGUF file path
3. Ingest your PDF:
   - `mrags ingest path/to/file.pdf`
4. Ask a question:
   - `mrags query "Summarize the ingested PDF."`

Tip: If you want retrieval only (no answer generation), set `ENABLE_LMM=false`.

## Local files and GitHub
Your local PDFs, indexes, and model files stay on your machine. This repo already ignores:
`.env`, `data/`, `*.pdf`, and `*.gguf`. GitHub only receives files you `git add` and `git commit`.

## FAQ (beginner friendly)
**Do I need an OpenAI key?**
No. You can run fully offline by using local embeddings and a local GGUF model.

**Why is my answer slow?**
Local models run on your CPU/GPU, which can take time. Smaller GGUF files are faster.

**Can I see the retrieved text?**
Yes. The CLI now shows a short preview of each top match so you can see what it used.

**Where does my PDF go?**
It stays on your machine. The extracted text and vectors are stored in `data/` locally.

### Local embeddings (no API key)
Set `EMBEDDING_BACKEND=local` and `LOCAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`.
If you do not have an OpenAI key, set `ENABLE_VLM=false` and `ENABLE_LMM=false` to skip image
summarization and answer generation.

### Local LMM (RTX 4060 8GB)
Set `LMM_BACKEND=local`, point `LMM_MODEL_PATH` to a GGUF file, and keep `ENABLE_LMM=true`.
For 8GB VRAM, a 4-bit 3B GGUF like `qwen2.5-3b-instruct-q4_k_m.gguf` fits well with
`LMM_N_GPU_LAYERS=20`.

## Daily update flow (GitHub)
Use this after you make changes and want to push them.

1. Check what changed:
   - `git status`
2. Add files you want to publish:
   - `git add .`
3. Commit with a clear message:
   - `git commit -m "Daily update"`
4. Push to GitHub:
   - `git push`

If this is your first push from this machine, run once:
- `git remote -v` to confirm your GitHub remote is set.
- If needed: `git push -u origin main`

## Notes
- The ingestion pipeline uses `unstructured` with the `hi_res` strategy.
- Images are summarized with `gpt-4o-mini` and stored as base64 in SQLite.
- Vectors are stored in FAISS; raw content is stored in SQLite.
