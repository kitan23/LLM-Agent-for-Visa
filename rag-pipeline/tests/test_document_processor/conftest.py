"""
Test fixtures and configuration for OPT-RAG tests.
"""

import os
import pytest
from pathlib import Path

@pytest.fixture 
def test_data_dir():
    """Return the path to the test data directory."""
    # Use a relative path from the test directory to the test data
    base_dir = Path(__file__).parent
    # Log the base directory path for debugging
    return base_dir / "examples"


@pytest.fixture 
def sample_pdf_path(test_data_dir):
    """Return path to a sample PDF file."""
    pdf_path = test_data_dir / "sample.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Sample PDF not found at {pdf_path}")
    return pdf_path


@pytest.fixture
def temp_vector_store(tmp_path):
    """Create a temporary directory for vector store data."""
    vector_store_dir = tmp_path / "vector_store"
    vector_store_dir.mkdir(exist_ok = True)
    return vector_store_dir