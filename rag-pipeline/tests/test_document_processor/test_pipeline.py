"""
Tests for document processing pipeline.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.document_processor.pipeline import process_documents, run_processing_pipeline


@pytest.mark.asyncio
async def test_process_documents(sample_pdf_path, temp_vector_store):
    """Test the complete document processing pipeline with a real sample PDF."""
    with patch('src.document_processor.pipeline.build_vector_store') as mock_build_store:
        mock_vector_store = MagicMock()
        mock_build_store.return_value = mock_vector_store

        result = await process_documents(
            source_path=sample_pdf_path,
            vector_store_path=temp_vector_store,
            chunk_size=500,
            chunk_overlap=50
        )

        # Verify pipeline execution
        assert result is not None
        assert result["status"] == "success"
        assert result["document_count"] > 0
        assert len(result["chunks"]) > 0

        # Verify the vector store was built with the processed chunks
        mock_build_store.assert_called_once()
        call_kwargs = mock_build_store.call_args[1]

        assert len(call_kwargs["chunks"]) == len(result["chunks"])
        assert call_kwargs["vector_store_path"] == temp_vector_store


@pytest.mark.asyncio
async def test_process_documents_txt_file(tmp_path, temp_vector_store):
    """Test the pipeline with a plain text file."""
    txt = tmp_path / "notes.txt"
    txt.write_text("A" * 1500)  # forces at least one split

    with patch('src.document_processor.pipeline.build_vector_store') as mock_build_store:
        mock_build_store.return_value = MagicMock()

        result = await process_documents(
            source_path=txt,
            vector_store_path=temp_vector_store,
        )

    assert result["status"] == "success"
    assert result["chunk_count"] >= 2


@pytest.mark.asyncio
async def test_process_documents_no_documents_loaded(tmp_path, temp_vector_store):
    """Test that an empty source aborts the pipeline with an error status."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = await process_documents(
        source_path=empty_dir,
        vector_store_path=temp_vector_store,
    )

    assert result["status"] == "error"
    assert "No documents" in result["error"]


def test_run_processing_pipeline(sample_pdf_path, temp_vector_store):
    """Test the synchronous wrapper for the pipeline."""
    with patch('src.document_processor.pipeline.process_documents', new_callable=AsyncMock) as mock_process:
        expected = {"status": "success", "chunks": ["c1"]}
        mock_process.return_value = expected

        result = run_processing_pipeline(
            source_path=sample_pdf_path,
            vector_store_path=temp_vector_store,
        )

        assert result == expected
        mock_process.assert_called_once()


@pytest.mark.asyncio
async def test_process_documents_list_of_paths(tmp_path, temp_vector_store):
    """Test processing a list of files loads all of them."""
    f1 = tmp_path / "a.txt"
    f1.write_text("alpha content")
    f2 = tmp_path / "b.txt"
    f2.write_text("beta content")

    with patch('src.document_processor.pipeline.build_vector_store') as mock_build_store:
        mock_build_store.return_value = MagicMock()

        result = await process_documents(
            source_path=[str(f1), str(f2)],
            vector_store_path=temp_vector_store,
        )

    assert result["status"] == "success"
    assert result["document_count"] == 2
