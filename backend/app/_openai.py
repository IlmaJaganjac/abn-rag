from __future__ import annotations

from openai import OpenAI

from backend.app.config import settings


def openai_client(timeout: float | None = None) -> OpenAI:
    key = settings.openai_api_key.get_secret_value()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set (check .env)")
    return OpenAI(api_key=key, **({"timeout": timeout} if timeout is not None else {}))
