"""
Text splitting functionality for OPT-RAG.

This module handles splitting documents into chunks for embedding and retrieval.
Metadata (e.g. the source file) is preserved on every chunk so that retrieval
results can cite where information came from.
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("opt_rag.document_processor.splitter")


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """Split documents into chunks while preserving their metadata.

    Args:
        documents: List of Document objects to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunked Document objects carrying the original metadata
    """
    logger.info(f"Splitting {len(documents)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})")

    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    logger.info(f"Generated {len(chunks)} document chunks")
    return chunks
