"""
Document loading functionality for OPT-RAG.

This module handles loading documents from various sources 
(PDFs, text files, etc.) for processing.
"""


import logging 
from pathlib import Path
from typing import List, Union, AsyncIterator

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_core.documents import Document



logger = logging.getLogger("opt_rag.document_processor.loader")

async def load_pdf_documents(source_path: Union[str, Path]) -> List[Document]:
    """Load documents from a PDF file or directory of PDFs.
    
    Args:
        source_path: Path to PDF file or directory containing PDFs
        
    Returns:
        List of Document objects with document content
    """
    source_path = Path(source_path)
    logger.info(f"Loading documents from {source_path}")
    
    documents = []
    try:
        if source_path.is_file():
            ext = source_path.suffix.lower()
            if ext == '.pdf':
                loader = PyPDFLoader(str(source_path))
                documents = loader.load()
                logger.info(f"Loaded {len(documents)} pages from {source_path.name}")
            elif ext == '.txt':
                with open(source_path, "r", encoding="utf-8") as f:
                    text = f.read()
                documents = [Document(page_content=text, metadata={"source": str(source_path)})]
                logger.info(f"Loaded 1 document from {source_path.name}")
            else:
                logger.warning(f"Unsupported file format: {source_path}")
        elif source_path.is_dir():
            loader = PyPDFDirectoryLoader(str(source_path))
            # Use async loading for better performance with large document sets
            async for document in loader.alazy_load():
                documents.append(document)
            logger.info(f"Loaded {len(documents)} pages from directory {source_path}")
        else:
            logger.error(f"Source path does not exist: {source_path}")
            raise FileNotFoundError(f"Source path does not exist: {source_path}")
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise
        
    return documents