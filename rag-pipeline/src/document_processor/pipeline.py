"""
Document processing pipeline for OPT-RAG.

This module provides a unified pipeline for loading, preprocessing,
and splitting documents before vectorization.
"""

import logging
import asyncio
from pathlib import Path
from typing import List, Union, Optional

from document_processor.loader import load_pdf_documents
from document_processor.splitter import split_documents
from retriever.vector_store import build_vector_store


logger = logging.getLogger("opt_rag.document_processor.pipeline")

async def process_documents(
        source_path: Union[str, Path], 
        vector_store_path: Union[str, Path], 
        device: str, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200, 
        force_rebuild: bool = False
): 
    """Process documents from source path and build a vector store.
    
    This function orchestrates the complete document processing pipeline:
    1. Load documents from source
    2. Split into chunks
    3. Build/update vector store
    
    Args:
        source_path: Path to document(s) source
        vector_store_path: Path to save/load vector store
        device: Device to use for embeddings (cuda, mps, cpu)
        chunk_size: Maximum chunk size for splitting
        chunk_overlap: Overlap between chunks
        force_rebuild: Force rebuilding vector store
        
    Returns:
        FAISS vector store
    """

    logger.info(f"Starting document processing pipeline from {source_path}")

    # Load documents 
    documents = await load_pdf_documents(source_path)
    if not documents: 
        logger.warning("No documents were loaded, aborting pipeline")
        return None 


    # Split the documents into chunk 
    chunks = split_documents(
        documents, 
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap
    )

    if not chunks:
        chunks = ["This is a dummy chunk."]

    # Build vector store
    vector_store = build_vector_store(
        chunks = chunks, 
        vector_store_path = vector_store_path, 
        device = device, 
        force_rebuild = force_rebuild
    )

    logger.info("Document processing pipeline completed successfully")
    return vector_store



def run_processing_pipeline(
    source_path: Union[str, Path],
    vector_store_path: Union[str, Path],
    device: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    force_rebuild: bool = False
):
    """Synchronous wrapper for the document processing pipeline.
    
    Args:
        source_path: Path to document(s) source
        vector_store_path: Path to save/load vector store
        device: Device to use for embeddings (cuda, mps, cpu)
        chunk_size: Maximum chunk size for splitting
        chunk_overlap: Overlap between chunks
        force_rebuild: Force rebuilding vector store
        
    Returns:
        FAISS vector store
    """
    return asyncio.run(process_documents(
        source_path=source_path,
        vector_store_path=vector_store_path,
        device=device,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        force_rebuild=force_rebuild
    ))