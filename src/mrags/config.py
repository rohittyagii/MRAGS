from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel


class AppSettings(BaseModel):
    """Application configuration read from environment variables.

    This holds sensible defaults so the CLI works out of the box.
    """
    openai_api_key: str | None = None
    embedding_backend: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vlm_model: str = "gpt-4o-mini"
    lmm_model: str = "gpt-4o"
    lmm_backend: str = "openai"
    lmm_model_path: str = ""
    lmm_n_gpu_layers: int = 20
    lmm_n_ctx: int = 4096
    lmm_n_threads: int = 8
    lmm_max_tokens: int = 512
    lmm_temperature: float = 0.2
    enable_vlm: bool = True
    enable_lmm: bool = True
    request_timeout_s: int = 60
    max_concurrency: int = 5
    chunk_tokens: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    faiss_index_path: str = "data/faiss.index"
    sqlite_path: str = "data/mrags.sqlite"
    log_level: str = "INFO"
    image_summary_prompt: str = (
        "You are an analytical engine. Describe the chart or diagram with axes, "
        "units, trends, and key data points. If a table is visible, summarize it."
    )

    @classmethod
    def from_env(cls) -> "AppSettings":
        """Build `AppSettings` by reading environment variables.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        defaults = cls()
        openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip() or None
        return cls(
            openai_api_key=openai_api_key,
            embedding_backend=os.environ.get("EMBEDDING_BACKEND", defaults.embedding_backend),
            embedding_model=os.environ.get("EMBEDDING_MODEL", defaults.embedding_model),
            local_embedding_model=os.environ.get(
                "LOCAL_EMBEDDING_MODEL", defaults.local_embedding_model
            ),
            vlm_model=os.environ.get("VLM_MODEL", defaults.vlm_model),
            lmm_model=os.environ.get("LMM_MODEL", defaults.lmm_model),
            lmm_backend=os.environ.get("LMM_BACKEND", defaults.lmm_backend),
            lmm_model_path=os.environ.get("LMM_MODEL_PATH", defaults.lmm_model_path),
            lmm_n_gpu_layers=int(
                os.environ.get("LMM_N_GPU_LAYERS", defaults.lmm_n_gpu_layers)
            ),
            lmm_n_ctx=int(os.environ.get("LMM_N_CTX", defaults.lmm_n_ctx)),
            lmm_n_threads=int(os.environ.get("LMM_N_THREADS", defaults.lmm_n_threads)),
            lmm_max_tokens=int(os.environ.get("LMM_MAX_TOKENS", defaults.lmm_max_tokens)),
            lmm_temperature=float(
                os.environ.get("LMM_TEMPERATURE", defaults.lmm_temperature)
            ),
            enable_vlm=_parse_bool(os.environ.get("ENABLE_VLM"), defaults.enable_vlm),
            enable_lmm=_parse_bool(os.environ.get("ENABLE_LMM"), defaults.enable_lmm),
            request_timeout_s=int(
                os.environ.get("REQUEST_TIMEOUT_S", defaults.request_timeout_s)
            ),
            max_concurrency=int(os.environ.get("MAX_CONCURRENCY", defaults.max_concurrency)),
            chunk_tokens=int(os.environ.get("CHUNK_TOKENS", defaults.chunk_tokens)),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", defaults.chunk_overlap)),
            top_k=int(os.environ.get("TOP_K", defaults.top_k)),
            faiss_index_path=os.environ.get("FAISS_INDEX_PATH", defaults.faiss_index_path),
            sqlite_path=os.environ.get("SQLITE_PATH", defaults.sqlite_path),
            log_level=os.environ.get("LOG_LEVEL", defaults.log_level),
        )


def ensure_parent_dir(path_str: str) -> None:
    """Create the parent directory for a given path if it doesn't exist.

    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_bool(value: str | None, default: bool) -> bool:
    """Parse common truthy strings into a boolean.

    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
