"""
Vector store management for OPT-RAG.

This module handles loading and interacting with the FAISS vector store
that contains embedded visa document chunks.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger("opt_rag.vector_store")


def compute_document_hash(text_chunks: List[str], embedding_model: str) -> str:
    """Compute a hash of the document content using the model name.
    
    Args:
        text_chunks: List of text chunks
        embedding_model: Name of the embedding model
        
    Returns:
        Hash of the document content
    """
    content = "".join(text_chunks) + embedding_model
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def create_or_load_vector_store(
    chunks: List[str], 
    embeddings: HuggingFaceEmbeddings, 
    cache_dir: str
) -> FAISS:
    """Create or load a vector store from text chunks.
    
    This function will:
    1. Check if a cached vector store exists
    2. Verify if the cache is valid using content hash
    3. Load from cache if valid, otherwise create a new vector store
    
    Args:
        chunks: List of text chunks to embed
        embeddings: Embeddings model to use
        cache_dir: Directory to save/load the vector store
        
    Returns:
        FAISS vector store
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = Path(cache_dir)

    # Generate hash for current content
    embedding_model_name = embeddings.model_name
    current_hash = compute_document_hash(chunks, embedding_model_name)

    # Define paths for cache validation
    hash_file = cache_path / "content_hash.txt"
    index_file = cache_path / "index.faiss"

    # Check if cached index exists and is valid
    if index_file.exists() and hash_file.exists():
        cached_hash = hash_file.read_text().strip()

        if current_hash == cached_hash:
            logger.info("Loading cached FAISS index for document retrieval")
            return FAISS.load_local(
                folder_path=str(cache_path),
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            logger.info("Cache invalidated: Documents or embeddings have changed")
            logger.info("Rebuilding vector store for updated documents...")

            # Clean up existing cache files
            for file in cache_path.iterdir():
                file.unlink()
    else:
        logger.info("No valid cache found. Creating new vector store...")

    # Create and save new vector store
    logger.info(f"Creating new vector store with {len(chunks)} document chunks...")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    # Save the new index and hash
    vector_store.save_local(str(cache_path))
    hash_file.write_text(current_hash)
    
    logger.info("Vector store created and cached successfully!")
    return vector_store


def load_vector_store(vector_store_path: str, device: str, force_reload: bool = False):
    """Load the FAISS vector store.
    
    Args:
        vector_store_path: Path to the vector store
        device: Device to use for embeddings
        force_reload: If True, ignore cache and force reload
        
    Returns:
        FAISS vector store
    """
    vector_store_path = Path(vector_store_path)
    logger.info(f"Loading vector store from {vector_store_path}")
    
    if not vector_store_path.exists():
        raise FileNotFoundError(f"Vector store not found: {vector_store_path}")
    
    try:
        # Initialize embedding model for consistency with vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )
        
        # Load vector store
        vector_store = FAISS.load_local(
            folder_path=str(vector_store_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        
        logger.info(f"Vector store loaded successfully with {vector_store._index.ntotal} vectors")
        return vector_store
        
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        raise


def build_vector_store(
    chunks: List[str],
    vector_store_path: str,
    device: str,
) -> FAISS:
    """Build or load a vector store from document chunks.
    
    Args:
        chunks: List of text chunks to embed
        vector_store_path: Path to save/load the vector store
        device: Device to use for embeddings
        force_rebuild: If True, rebuild the vector store regardless of cache
        
    Returns:
        FAISS vector store
    """
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )
    
    # Create or load vector store with caching
    return create_or_load_vector_store(
        chunks=chunks,
        embeddings=embeddings,
        cache_dir=vector_store_path
    )


