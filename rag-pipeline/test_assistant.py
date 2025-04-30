# #!/usr/bin/env python3
# """
# Manual testing script for OPT-RAG Assistant.

# This script demonstrates how to initialize and use the OPTRagAssistant
# with real models and documents. Unlike the automated tests in the tests/ 
# directory, this script is meant for interactive testing with actual models.

# Usage:
#     python manual_test_assistant.py [options]

# Requirements:
#     - HuggingFace model downloaded to models/
#     - Sample documents in examples/
# """

# import os
# import asyncio
# import time
# import argparse
# import logging
# from pathlib import Path

# from src.llm.assistant import OPTRagAssistant
# from src.utils.logging import setup_logging


# async def test_assistant(model_path, vector_store_path, device=None, 
#                         document_path=None, query=None, stream=False):
#     """Test the OPTRagAssistant with real models and documents."""
    
#     print(f"Initializing OPT-RAG Assistant with model at {model_path}")
#     print(f"Using vector store at {vector_store_path}")
#     print(f"Device: {device or 'auto-detect'}")
    
#     # Create assistant
#     assistant = OPTRagAssistant(
#         model_path=model_path,
#         vector_store_path=vector_store_path,
#         device=device
#     )
    
#     # Add documents if specified
#     if document_path:
#         print(f"\nAdding document: {document_path}")
#         result = await assistant.add_documents(document_path)
#         print(f"Document processing result: {result}")
    
#     # Process query if specified
#     if query:
#         print(f"\nProcessing query: {query}")
        
#         start_time = time.time()
        
#         if stream:
#             print("\nStreaming response:")
#             print("-" * 50)
#             async for token in assistant.astream_response(query):
#                 print(token, end="", flush=True)
#             print("\n" + "-" * 50)
#         else:
#             result = assistant.answer_question(query)
#             print("\nResponse:")
#             print("-" * 50)
#             print(result.get("answer", "No answer generated"))
#             print("-" * 50)
        
#         print(f"Total time: {time.time() - start_time:.2f} seconds")
    
#     # List documents in vector store
#     print("\nVector store information:")
#     print(assistant.list_documents())


# def main():
#     """Main entry point for the script."""
#     parser = argparse.ArgumentParser(description="Manual testing for the OPT-RAG Assistant")
#     parser.add_argument("--model", type=str, default="models/Qwen2.5-0.5B-Instruct",
#                         help="Path to the HuggingFace model")
#     parser.add_argument("--vector-store", type=str, default="vector_store",
#                         help="Path to the vector store directory")
#     parser.add_argument("--document", type=str, help="Path to a document to process")
#     parser.add_argument("--query", type=str, help="Query to process")
#     parser.add_argument("--device", type=str, choices=["cuda", "mps", "cpu"], 
#                         help="Device to use (default: auto-detect)")
#     parser.add_argument("--stream", action="store_true", help="Stream the response")
    
#     args = parser.parse_args()
    
#     # Setup logging
#     setup_logging()
    
#     # Run the assistant test
#     asyncio.run(test_assistant(
#         model_path=args.model,
#         vector_store_path=args.vector_store,
#         device=args.device,
#         document_path=args.document,
#         query=args.query,
#         stream=args.stream
#     ))


# if __name__ == "__main__":
#     main() 