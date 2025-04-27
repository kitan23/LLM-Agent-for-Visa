"""
Text splitting functionality for OPT-RAG.

This module handles splitting documents into chunks for embedding and retrieval.
"""

import logging
from typing import List, Union

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("opt_rag.document_processor.splitter")

def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """Split documents into chunks for embedding.
    
    Args:
        documents: List of Document objects to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    logger.info(f"Splitting {len(documents)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
    
    # Extract text from documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = []
    for document in documents:
        document_chunks = text_splitter.split_text(document.page_content)
        chunks.extend(document_chunks)
    
    logger.info(f"Generated {len(chunks)} text chunks from documents")
    return chunks