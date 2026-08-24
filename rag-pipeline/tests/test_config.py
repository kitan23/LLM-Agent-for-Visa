"""
Tests for application settings loading.
"""

from pathlib import Path

import pytest

from src.utils.config import Settings, get_settings


class TestSettings:
    """Test suite for the Settings class."""

    def test_defaults(self, monkeypatch):
        """Settings fall back to sane defaults when no env vars are set."""
        get_settings.cache_clear()
        settings = get_settings()

        assert settings.runpod_api_key is None
        assert settings.runpod_endpoint_id is None
        assert settings.openai_api_key is None
        assert "instruct" in settings.runpod_model_name
        assert settings.vector_store_path == "./vector_store"

    def test_env_loading(self, monkeypatch):
        """Settings pick up values from environment variables."""
        monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
        monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "test-endpoint")
        monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
        monkeypatch.setenv("VECTOR_STORE_PATH", "/tmp/store")

        get_settings.cache_clear()
        settings = get_settings()

        assert settings.runpod_api_key == "test-key"
        assert settings.runpod_endpoint_id == "test-endpoint"
        assert settings.openai_api_key == "oai-key"
        assert settings.vector_store_path == "/tmp/store"

    def test_model_name_overrides(self, monkeypatch):
        """Custom model names override the defaults."""
        monkeypatch.setenv("RUNPOD_MODEL_NAME", "my-model")
        monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-4-turbo")

        get_settings.cache_clear()
        settings = get_settings()

        assert settings.runpod_model_name == "my-model"
        assert settings.openai_model_name == "gpt-4-turbo"

    def test_get_settings_cached(self, monkeypatch):
        """get_settings returns a cached instance until the cache is cleared."""
        get_settings.cache_clear()
        first = get_settings()
        second = get_settings()
        assert first is second

        get_settings.cache_clear()
        third = get_settings()
        assert third is not first

    def test_extra_env_vars_ignored(self, monkeypatch):
        """Unknown environment variables do not break settings creation."""
        monkeypatch.setenv("SOME_UNRELATED_VAR", "value")
        settings = Settings()
        assert settings is not None
