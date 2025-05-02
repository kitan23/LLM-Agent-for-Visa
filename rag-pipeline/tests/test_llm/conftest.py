"""
Test fixtures for OPT-RAG Assistant tests.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import torch

@pytest.fixture
def test_model_path(tmp_path):
    """Create a mock model directory."""
    model_dir = tmp_path / "test_model"
    model_dir.mkdir(exist_ok=True)
    return model_dir

@pytest.fixture
def test_vector_store_path(tmp_path):
    """Create a temporary directory for vector store data."""
    vector_store_dir = tmp_path / "vector_store"
    vector_store_dir.mkdir(exist_ok=True)
    return vector_store_dir

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.model_max_length = 512
    return tokenizer

@pytest.fixture
def mock_model():
    """Mock LLM model for testing."""
    model = MagicMock()
    return model

@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    vector_store = MagicMock()
    # Mock the as_retriever method
    retriever = MagicMock()
    vector_store.as_retriever.return_value = retriever
    # Mock the index attribute for vector count
    vector_store.index = MagicMock()
    vector_store.index.ntotal = 100
    return vector_store

@pytest.fixture
def mock_hardware_detection():
    """Mock hardware detection to always return CPU for tests."""
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False):
        yield "cpu" 