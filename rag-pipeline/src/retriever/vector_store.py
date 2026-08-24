"""
Vector store management for OPT-RAG.

This module handles loading and interacting with the FAISS vector store
that contains embedded visa document chunks.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger("opt_rag.vector_store")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Placeholder text used to bootstrap empty stores.
DUMMY_CHUNK = "This is a dummy chunk."

# Placeholder variants that must never be treated as real corpus content.
_DUMMY_TEXTS = {DUMMY_CHUNK, "dummy text", "This is a dummy chunk."}

ChunkLike = Union[str, Document]


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the embedding model used across the project."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


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


def coerce_documents(chunks: Optional[Iterable[ChunkLike]]) -> Tuple[List[Document], List[str]]:
    """Normalize chunks into (documents, texts), preserving metadata when present."""
    documents: List[Document] = []
    texts: List[str] = []
    for chunk in chunks or []:
        if isinstance(chunk, Document):
            documents.append(chunk)
            texts.append(chunk.page_content)
        else:
            documents.append(Document(page_content=chunk))
            texts.append(chunk)
    return documents, texts


def _is_real_chunk(chunk: ChunkLike) -> bool:
    text = chunk.page_content if isinstance(chunk, Document) else chunk
    return bool(text) and text not in _DUMMY_TEXTS


def build_faiss_store(chunks: List[ChunkLike], embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Create a FAISS store from strings or Documents, keeping metadata."""
    documents, _ = coerce_documents(chunks)
    return FAISS.from_documents(documents, embedding=embeddings)


def store_corpus_texts(vector_store: FAISS) -> List[str]:
    """Return the real corpus texts held by a FAISS store, excluding placeholders."""
    texts = []
    if hasattr(vector_store, "docstore"):
        for doc in vector_store.docstore._dict.values():
            content = getattr(doc, "page_content", None)
            if content and content not in _DUMMY_TEXTS:
                texts.append(content)
    return texts


def append_to_vector_store(
    existing_store: Optional[FAISS],
    new_chunks: List[ChunkLike],
    vector_store_path,
) -> FAISS:
    """Merge new chunks into an existing store (or build a fresh one).

    Persists the merged store and updates the content hash so subsequent
    startups load the combined corpus instead of rebuilding. Metadata on
    incoming Document chunks (e.g. source file) is preserved.

    Args:
        existing_store: Currently loaded store (may contain only placeholder data)
        new_chunks: Text chunks or Documents to add
        vector_store_path: Directory used for persistence

    Returns:
        The merged FAISS vector store
    """
    embeddings = get_embeddings()

    # Filter placeholders out of the incoming chunks
    real_new_chunks = [c for c in (new_chunks or []) if _is_real_chunk(c)]

    has_existing_content = bool(existing_store is not None and store_corpus_texts(existing_store))

    if has_existing_content and real_new_chunks:
        new_store = build_faiss_store(real_new_chunks, embeddings)
        existing_store.merge_from(new_store)
        merged_store = existing_store
    elif real_new_chunks:
        merged_store = build_faiss_store(real_new_chunks, embeddings)
    else:
        # Nothing real to add; keep whatever we already have
        merged_store = existing_store

    if merged_store is None:
        merged_store = build_faiss_store([DUMMY_CHUNK], embeddings)

    cache_path = Path(vector_store_path)
    os.makedirs(cache_path, exist_ok=True)

    merged_store.save_local(str(cache_path))

    # Persist a hash over the full merged corpus for cache validation
    full_corpus = store_corpus_texts(merged_store)
    hash_file = cache_path / "content_hash.txt"
    if full_corpus:
        hash_file.write_text(compute_document_hash(full_corpus, EMBEDDING_MODEL_NAME))

    logger.info(f"Vector store now contains {merged_store.index.ntotal} vectors")
    return merged_store


def create_or_load_vector_store(
    chunks: List[ChunkLike],
    embeddings: HuggingFaceEmbeddings,
    cache_dir,
) -> FAISS:
    """Create or load a vector store from text chunks.

    This function will:
    1. Check if a cached vector store exists
    2. Verify if the cache is valid using content hash
    3. Load from cache if valid, otherwise create a new vector store

    Args:
        chunks: List of text chunks (or Documents) to embed
        embeddings: Embeddings model to use
        cache_dir: Directory to save/load the vector store

    Returns:
        FAISS vector store
    """
    _, texts = coerce_documents(chunks)

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = Path(cache_dir)

    # Generate hash for current content
    current_hash = compute_document_hash(texts, EMBEDDING_MODEL_NAME)

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
                if file.is_file():
                    file.unlink()
    else:
        logger.info("No valid cache found. Creating new vector store...")

    # Create and save new vector store
    logger.info(f"Creating new vector store with {len(chunks)} document chunks...")
    vector_store = build_faiss_store(chunks, embeddings)

    # Save the new index and hash
    vector_store.save_local(str(cache_path))
    hash_file.write_text(current_hash)

    logger.info("Vector store created and cached successfully!")
    return vector_store


def load_vector_store(vector_store_path, force_reload: bool = False):
    """Load the FAISS vector store.

    Args:
        vector_store_path: Path to the vector store
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
        embeddings = get_embeddings()

        # Load vector store
        vector_store = FAISS.load_local(
            folder_path=str(vector_store_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        logger.info(f"Vector store loaded successfully with {vector_store.index.ntotal} vectors")
        return vector_store

    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        raise


def build_vector_store(
    chunks: List[ChunkLike],
    vector_store_path,
    force_rebuild: bool = False
) -> FAISS:
    """Build or load a vector store from document chunks.

    Args:
        chunks: List of text chunks (or Documents) to embed
        vector_store_path: Path to save/load the vector store
        force_rebuild: If True, rebuild the vector store regardless of cache

    Returns:
        FAISS vector store
    """
    # Initialize embeddings model
    embeddings = get_embeddings()

    if not chunks:
        logger.warning("No chunks provided. Returning dummy FAISS store.")
        chunks = [DUMMY_CHUNK]

    # Create or load vector store with caching
    return create_or_load_vector_store(
        chunks=chunks,
        embeddings=embeddings,
        cache_dir=vector_store_path
    )
