"""
Fixtures for retriever (vector store) tests.
"""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding


@pytest.fixture
def fake_embeddings():
    """Deterministic fake embeddings so no model download is required."""
    return DeterministicFakeEmbedding(size=32)
