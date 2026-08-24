"""
Tests for document splitting functionality.
"""

import pytest
from langchain_core.documents import Document

from src.document_processor.splitter import split_documents


def test_split_documents():
    """Test splitting documents into chunks."""
    doc1 = Document(page_content="A" * 1500, metadata={"source": "a.pdf"})
    doc2 = Document(page_content="B" * 1500, metadata={"source": "b.pdf"})
    documents = [doc1, doc2]

    # Test with default parameters
    chunks = split_documents(documents)

    assert chunks is not None
    assert len(chunks) > 2  # Should split into more chunks than documents
    assert all(len(chunk.page_content) <= 1000 for chunk in chunks)  # Default chunk size


def test_split_documents_custom_params():
    """Test splitting with custom chunk size and overlap."""
    doc = Document(page_content="C" * 3000)

    chunks = split_documents([doc], chunk_size=500, chunk_overlap=100)

    assert chunks is not None
    assert len(chunks) > 5
    assert all(len(chunk.page_content) <= 500 for chunk in chunks)


def test_split_documents_empty():
    """Test splitting with empty documents."""
    chunks = split_documents([])
    assert chunks == []


def test_split_documents_preserves_metadata():
    """Chunks must carry the source metadata for citation purposes."""
    doc = Document(
        page_content="Word " * 2000,
        metadata={"source": "opt_guide.pdf", "document_type": "immigration"},
    )

    chunks = split_documents([doc], chunk_size=800, chunk_overlap=100)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "opt_guide.pdf"
        assert chunk.metadata["document_type"] == "immigration"
