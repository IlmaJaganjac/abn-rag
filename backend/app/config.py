from __future__ import annotations

from pathlib import Path

from openai import OpenAI
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed runtime configuration loaded from `.env` and default values."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    openai_api_key: SecretStr = Field(default=SecretStr(""))
    llama_cloud_api_key: SecretStr = Field(default=SecretStr(""))
    openai_embedding_model: str = "text-embedding-3-small"
    openai_answer_model: str = "gpt-5.4-mini"
    openai_extract_model: str = "gpt-5.4-mini"
    openai_validate_model: str = "gpt-5.4-mini"

    chroma_persist_dir: Path = PROJECT_ROOT / "backend/data/chroma"
    chroma_collection: str = "annual_reports"
    processed_dir: Path = PROJECT_ROOT / "backend/data/processed"

    reports_dir: Path = PROJECT_ROOT / "backend/data/reports"

    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120
    top_k: int = 12
    rerank_top_n: int = 20
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_batch_size: int = 64

    def get_chroma_path(self) -> Path:
        """Ensure the Chroma directory exists and return its path."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        return self.chroma_persist_dir

    def get_processed_path(self) -> Path:
        """Ensure the processed-data directory exists and return its path."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        return self.processed_dir


settings = Settings()


def openai_client(timeout: float | None = None) -> OpenAI:
    """Return an authenticated OpenAI client, optionally with a request timeout."""
    key = settings.openai_api_key.get_secret_value()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set (check .env)")
    return OpenAI(api_key=key, **({"timeout": timeout} if timeout is not None else {}))
