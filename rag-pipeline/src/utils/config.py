"""
Configuration management for OPT-RAG.

This module handles loading and validating configuration settings from
environment variables and config files.
"""

import os
import logging
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import field_validator

logger = logging.getLogger("opt_rag.config")

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Model settings
    model_path: str = "./models/qwen2.5-1.5b-instruct"
    vector_store_path: str = "./rag-pipeline/vector_store"
    device: str = None  # Auto-detect if None
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    
    # Application settings
    app_name: str = "OPT-RAG International Student Visa Assistant"
    enable_streaming: bool = True
    
    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v):
        """Validate model path exists."""
        path = Path(v)
        if not path.exists():
            # Return the value anyway, but log a warning
            logger.warning(f"Model path {v} does not exist. It will need to be downloaded.")
            return v
        return v
    
    @field_validator("vector_store_path")
    @classmethod
    def validate_vector_store_path(cls, v):
        """Validate vector store path exists."""
        path = Path(v)
        if not path.exists():
            # Return the value anyway, but log a warning
            logger.warning(f"Vector store path {v} does not exist. It will be created if needed.")
            return v
        return v
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_prefix = "OPT_RAG_"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()