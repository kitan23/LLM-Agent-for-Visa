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
    
    # Vector store settings
    vector_store_path: str = "./vector_store"
    
    # RunPod settings
    runpod_api_key: Optional[str] = None
    runpod_endpoint_id: Optional[str] = None
    runpod_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Default model
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    
    # Application settings
    app_name: str = "OPT-RAG International Student Visa Assistant"
    enable_streaming: bool = True
    
    # Optional: Tracing settings
    otlp_endpoint: Optional[str] = None
    
    @field_validator("runpod_api_key")
    @classmethod
    def validate_runpod_key(cls, v):
        """Validate RunPod API key exists."""
        if v:
            return v
        
        # Try to get from environment
        runpod_key = os.environ.get("RUNPOD_API_KEY")
        if not runpod_key:
            raise ValueError("RunPod API key not found. Set RUNPOD_API_KEY environment variable.")
        
        logger.info("Using RUNPOD_API_KEY from environment")
        return runpod_key
    
    @field_validator("runpod_endpoint_id")
    @classmethod
    def validate_runpod_endpoint(cls, v):
        """Validate RunPod endpoint ID exists."""
        if v:
            return v
        
        # Try to get from environment
        endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
        if not endpoint_id:
            raise ValueError("RunPod endpoint ID not found. Set RUNPOD_ENDPOINT_ID environment variable.")
        
        logger.info("Using RUNPOD_ENDPOINT_ID from environment")
        return endpoint_id
    
    @field_validator("vector_store_path")
    @classmethod
    def validate_vector_store_path(cls, v):
        """Validate vector store path exists."""
        path = Path(v)
        if not path.exists():
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