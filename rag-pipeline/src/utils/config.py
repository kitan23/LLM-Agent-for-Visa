"""
Configuration management for OPT-RAG.

This module handles loading and validating configuration settings from
environment variables and config files.
"""

import os
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator

logger = logging.getLogger("opt_rag.config")

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Model settings
    model_path: str = "./models/qwen2.5-1.5b"
    vector_store_path: str = "./vector_store"
    device: Optional[str] = None  # Auto-detect if None
    
    # API-based LLM settings
    use_api_llm: bool = True
    llm_api_provider: str = "openai"
    llm_api_key: Optional[str] = None
    llm_api_model: str = "gpt-4o-mini"
    llm_api_base_url: Optional[str] = None
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    
    # Tracing settings
    otlp_endpoint: str = "http://jaeger:4317"
    
    # Application settings
    app_name: str = "OPT-RAG International Student Visa Assistant"
    enable_streaming: bool = True
    
    @field_validator("llm_api_key")
    @classmethod
    def validate_api_key(cls, v):
        """Validate API key exists, with fallback to OPENAI_API_KEY."""
        if v:
            return v
        
        # Fallback to standard OpenAI environment variable
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            logger.info("Using OPENAI_API_KEY as fallback for LLM API key")
            return openai_key
        
        # If no key found, return None (will be validated later in assistant)
        return None
    
    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v, values):
        """Validate model path exists."""
        # Skip validation if using API mode
        if values.data.get('use_api_llm', True):
            return v
            
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