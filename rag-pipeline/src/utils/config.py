"""
Application settings for OPT-RAG.

This module centralizes configuration loaded from environment variables
(or a .env file) using pydantic-settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_dotenv() -> Optional[str]:
    """Locate the project .env file regardless of the current working directory.

    Searches upward from this file's location and from the current working
    directory so the app works whether it is started from the repo root, from
    rag-pipeline/, or from inside a container.
    """
    candidates = []
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for parent in [start, *start.parents]:
            candidates.append(parent / ".env")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


_DOTENV_PATH = _find_dotenv()
if _DOTENV_PATH:
    load_dotenv(dotenv_path=_DOTENV_PATH, override=False)


class Settings(BaseSettings):
    """Runtime settings for the OPT-RAG backend."""

    model_config = SettingsConfigDict(
        env_file=_DOTENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # RunPod (primary LLM provider)
    runpod_api_key: Optional[str] = None
    runpod_endpoint_id: Optional[str] = None
    runpod_model_name: str = "qwen/qwen2.5-1.5b-instruct"

    # OpenAI (fallback LLM provider)
    openai_api_key: Optional[str] = None
    openai_model_name: str = "gpt-4o-mini"

    # Vector store location
    vector_store_path: str = "./vector_store"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
