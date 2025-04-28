"""
Tests for document processing pipeline.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from document_processor.pipeline import process_documents, run_processing_pipeline


@pytest.mark.asyncio
async def test_process_documents(sample_pdf_path, temp_vector_store):
    """Test the complete document processing pipeline."""
    with patch('document_processor.pipeline.build_vector_store') as mock_build_store:
        # Mock the vector store creation
        mock_vector_store = MagicMock()
        mock_build_store.return_value = mock_vector_store
        
        # Run the pipeline
        result = await process_documents(
            source_path=sample_pdf_path,
            vector_store_path=temp_vector_store,
            device="cpu",  # Use CPU for tests
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Verify pipeline execution
        assert result is not None
        assert result == mock_vector_store
        
        # Verify the vector store was built with appropriate parameters
        mock_build_store.assert_called_once()
        call_args = mock_build_store.call_args[1]
        
        assert "chunks" in call_args
        assert len(call_args["chunks"]) > 0
        assert call_args["vector_store_path"] == temp_vector_store
        assert call_args["device"] == "cpu"


def test_run_processing_pipeline(sample_pdf_path, temp_vector_store):
    """Test the synchronous wrapper for the pipeline."""
    with patch('document_processor.pipeline.process_documents', new_callable=AsyncMock) as mock_process:
        # Mock the async process_documents function
        mock_vector_store = MagicMock()
        
        # Create a future that's already done
        mock_process.return_value = mock_vector_store 
        
        # Run the synchronous wrapper
        result = run_processing_pipeline(
            source_path=sample_pdf_path,
            vector_store_path=temp_vector_store,
            device="cpu"
        )
        
        # Verify results
        assert result == mock_vector_store
        mock_process.assert_called_once()