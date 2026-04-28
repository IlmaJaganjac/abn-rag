from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    openai_api_key: SecretStr = Field(default=SecretStr(""))
    openai_embedding_model: str = "text-embedding-3-small"
    openai_answer_model: str = "gpt-4o-mini"

    chroma_persist_dir: Path = Path("backend/data/chroma")
    chroma_collection: str = "annual_reports"

    reports_dir: Path = Path("reports")

    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120
    top_k: int = 8
    embedding_batch_size: int = 64

    def get_chroma_path(self) -> Path:
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        return self.chroma_persist_dir


settings = Settings()
