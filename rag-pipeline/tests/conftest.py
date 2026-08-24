"""
Root conftest for OPT-RAG tests.

Ensures a clean environment for every test so settings do not leak
between cases.
"""

import pytest

from src.utils.config import get_settings


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Remove LLM provider credentials and reset the settings cache per test."""
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.delenv("RUNPOD_ENDPOINT_ID", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VECTOR_STORE_PATH", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
