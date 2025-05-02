#!/usr/bin/env python3
"""
Test script for document retrieval from vector store.

This script tests the retrieval part of the RAG pipeline without using the LLM.
"""

import os
import argparse
import asyncio
from pathlib import Path

from src.retriever.vector_store import load_vector_store
from src.utils.logging import setup_logging


def test_retrieval(vector_store_path, query, device="cpu", num_results=5):
    """Test document retrieval from vector store.
    
    Args:
        vector_store_path: Path to the vector store
        query: Query to search for
        device: Device to use (cpu, cuda, mps)
        num_results: Number of results to retrieve
    """
    print(f"Loading vector store from {vector_store_path}")
    vector_store = load_vector_store(
        vector_store_path=vector_store_path,
        device=device,
        force_reload=False
    )
    
    print(f"Vector store loaded with {vector_store.index.ntotal} vectors")
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": num_results}
    )
    
    print(f"\nSearching for: {query}")
    
    # Retrieve documents
    try:
        retrieval_result = retriever.invoke(query)
        docs = retrieval_result if isinstance(retrieval_result, list) else retrieval_result['documents']
        
        print(f"Retrieved {len(docs)} documents")
        
        # Print the documents
        for i, doc in enumerate(docs):
            print(f"\n--- Document {i+1} ---")
            # Truncate long documents for display
            content = doc.page_content
            if len(content) > 500:
                content = content[:500] + "... [truncated]"
            print(content)
    
    except Exception as e:
        print(f"Error during retrieval: {e}")
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test document retrieval")
    parser.add_argument(
        "--vector-store",
        type=str,
        default="vector_store",
        help="Path to the vector store"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What is OPT and who is eligible for it?",
        help="Query to search for"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use"
    )
    parser.add_argument(
        "--num-results",
        type=int,
        default=5,
        help="Number of results to retrieve"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Run retrieval test
    test_retrieval(
        vector_store_path=args.vector_store,
        query=args.query,
        device=args.device,
        num_results=args.num_results
    )


if __name__ == "__main__":
    main() 