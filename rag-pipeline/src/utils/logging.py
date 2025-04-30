"""
Logging configuration for OPT-RAG.

This module provides a centralized way to configure logging for the application.
"""

import logging
import sys
from pathlib import Path


def setup_logging(log_level=logging.INFO, log_file=None):
    """Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional file path for file-based logging
    """
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(console_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Set specific logger levels 
    logging.getLogger("opt_rag").setLevel(log_level)
    
    # Reduce verbosity for third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    
    logging.info("Logging configured successfully") 