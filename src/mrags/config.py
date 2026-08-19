from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application configuration read from environment variables or a .env file.

    pydantic-settings automatically reads every field from its matching
    environment variable (case-insensitive) and, when a `.env` file is
    present, loads it first.  No manual ``os.environ.get`` calls are needed.

    Example — override any field from the shell::

        EMBEDDING_BACKEND=local mrags ingest file.pdf
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Don't raise if .env is missing — it's optional.
        env_ignore_empty=False,
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")

    embedding_backend: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    vlm_model: str = "gpt-4o-mini"
    enable_vlm: bool = True

    lmm_model: str = "gpt-4o"
    lmm_backend: str = "openai"
    lmm_model_path: str = ""
    lmm_n_gpu_layers: int = 20
    lmm_n_ctx: int = 4096
    lmm_n_threads: int = 8
    lmm_max_tokens: int = 512
    lmm_temperature: float = 0.2
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
        """Build ``AppSettings`` from the environment (and optional .env file).

        This is a thin compatibility shim — callers that used the old
        ``AppSettings.from_env()`` pattern continue to work unchanged.
        pydantic-settings handles all the env / .env loading automatically
        when the class is instantiated.

        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        return cls()


def ensure_parent_dir(path_str: str) -> None:
    """Create the parent directory for a given path if it doesn't exist.

    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
