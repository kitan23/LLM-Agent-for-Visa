#!/usr/bin/env python3
"""
Simplified test script for document processing.

This script only tests the document processing functionality without loading the LLM.
"""

import asyncio
import argparse
from pathlib import Path

from src.document_processor.pipeline import process_documents
from src.utils.logging import setup_logging


async def test_document_processing(document_path, vector_store_path="vector_store"):
    """Test document processing functionality."""
    print(f"Processing document: {document_path}")
    
    result = await process_documents(
        source_path=document_path,
        vector_store_path=vector_store_path,
        device="cpu",  # Use CPU to reduce resource usage
        chunk_size=1000,
        chunk_overlap=200
    )
    
    print("\nDocument processing result:")
    print(f"Status: {result.get('status', 'unknown')}")
    print(f"Documents processed: {result.get('document_count', 0)}")
    print(f"Chunks created: {result.get('chunk_count', 0)}")
    
    # Print first chunk as sample if available
    chunks = result.get("chunks", [])
    if chunks:
        print("\nSample chunk content:")
        print("=" * 50)
        print(chunks[0][:300] + "..." if len(chunks[0]) > 300 else chunks[0])
        print("=" * 50)
    
    return result


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test document processing functionality")
    parser.add_argument("--document", type=str, required=True, 
                        help="Path to a document to process")
    parser.add_argument("--vector-store", type=str, default="vector_store",
                        help="Path to the vector store directory")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Run the document processing test
    asyncio.run(test_document_processing(
        document_path=args.document,
        vector_store_path=args.vector_store
    ))


if __name__ == "__main__":
    main() 