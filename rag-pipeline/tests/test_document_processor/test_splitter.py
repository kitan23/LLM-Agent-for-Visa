"""
Tests for document splitting functionality.
"""

import pytest
from langchain_core.documents import Document

from document_processor.splitter import split_documents


def test_split_documents():
    """Test splitting documents into chunks."""
    # Create sample documents
    doc1 = Document(page_content="A" * 1500)
    doc2 = Document(page_content="B" * 1500)
    documents = [doc1, doc2]
    
    # Test with default parameters
    chunks = split_documents(documents)
    
    assert chunks is not None
    assert len(chunks) > 2  # Should split into more chunks than documents
    assert all(len(chunk) <= 1000 for chunk in chunks)  # Default chunk size


def test_split_documents_custom_params():
    """Test splitting with custom chunk size and overlap."""
    doc = Document(page_content="C" * 3000)
    
    # Test with custom parameters
    chunks = split_documents([doc], chunk_size=500, chunk_overlap=100)
    
    assert chunks is not None
    assert len(chunks) > 5  # Should split into multiple chunks
    assert all(len(chunk) <= 500 for chunk in chunks)  # Custom chunk size


def test_split_documents_empty():
    """Test splitting with empty documents."""
    chunks = split_documents([])
    assert chunks == []  # Should return empty list for empty input