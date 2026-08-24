"""
Tests for document loading functionality.
"""

import pytest
import asyncio
from pathlib import Path
from src.document_processor.loader import load_pdf_documents


@pytest.mark.asyncio
async def test_load_pdf_documents_single_file(sample_pdf_path):
    """Test loading documents from a single PDF file."""

    documents = await load_pdf_documents(sample_pdf_path)
    
    assert documents is not None
    assert len(documents) > 0
    assert hasattr(documents[0], 'page_content')
    assert len(documents[0].page_content) > 0



@pytest.mark.asyncio
async def test_load_pdf_documents_directory(test_data_dir):
    """Test loading documents from a directory of PDFs."""
    documents = await load_pdf_documents(test_data_dir)
    
    assert documents is not None
    assert len(documents) > 0


@pytest.mark.asyncio
async def test_load_pdf_documents_nonexistent_file():
    """Test loading from a nonexistent file raises an error."""
    with pytest.raises(FileNotFoundError):
        await load_pdf_documents("nonexistent_file.pdf")


@pytest.mark.asyncio
async def test_load_pdf_documents_txt_file(tmp_path):
    """Test loading a plain text file."""
    text_file = tmp_path / "notes.txt"
    text_file.write_text("OPT filing guidance text")

    documents = await load_pdf_documents(text_file)
    assert len(documents) == 1
    assert documents[0].page_content == "OPT filing guidance text"
    assert documents[0].metadata["source"] == str(text_file)


@pytest.mark.asyncio
async def test_load_pdf_documents_unsupported_format(tmp_path):
    """Test loading an unsupported file format returns no documents."""
    bad_file = tmp_path / "image.png"
    bad_file.write_bytes(b"\x89PNG\r\n")

    documents = await load_pdf_documents(bad_file)
    assert len(documents) == 0