# MRAGS Handoff

## Status
- Local embeddings and local LMM support are implemented.
- CLI commands: ingest, query, validate-lmm.
 - CLI commands: ingest, query, validate-lmm, status.
- validate-lmm runs successfully with local GGUF model path.
- Tests: smoke test is green.
- Repo created and pushed: https://github.com/rohittyagii/MRAGS (branch: main).
- Application.pdf ingested successfully (20 elements).
- Summary still needs a local GGUF model path or OPENAI_API_KEY.

## Current Environment
- Python env detected by tooling: system Python 3.14.
- Local LMM uses llama-cpp-python and a GGUF model.
- Default env sample: .env.example

## Key Commands
- Validate local LMM:
  - C:/Users/rohit/AppData/Local/Python/pythoncore-3.14-64/python.exe -m mrags.cli validate-lmm
- Ingest a PDF:
  - C:/Users/rohit/AppData/Local/Python/pythoncore-3.14-64/python.exe -m mrags.cli ingest path/to/file.pdf
- Query:
  - C:/Users/rohit/AppData/Local/Python/pythoncore-3.14-64/python.exe -m mrags.cli query "Your question here"
 - Query (context-only):
   - C:/Users/rohit/AppData/Local/Python/pythoncore-3.14-64/python.exe -m mrags.cli query "Your question here" --context-only
 - Status (index + DB):
   - C:/Users/rohit/AppData/Local/Python/pythoncore-3.14-64/python.exe -m mrags.cli status

## Required Files
- GGUF model file at the path set in LMM_MODEL_PATH (see .env.example).
- FAISS index and SQLite DB are created by ingest.

## Next Steps
- Run ingest on a target PDF and then run query to validate end-to-end answers.
 - Run ingest on a target PDF and then run query to validate end-to-end answers.
 - Use `--context-only` to see retrieved passages when you want to verify sources.
 - Use `status` to inspect FAISS vector counts and SQLite element counts before/after ingest.
- Confirm .venv usage if desired and align tooling to that interpreter.

## Notes
- VLM is disabled by default for offline mode (ENABLE_VLM=false).
- LMM is enabled by default for local mode (ENABLE_LMM=true, LMM_BACKEND=local).
- Tesseract is installed but PATH was set per session when running ingest.
