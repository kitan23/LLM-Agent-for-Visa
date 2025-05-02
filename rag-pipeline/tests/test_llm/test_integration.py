"""
Integration tests for the OPT-RAG pipeline.

These tests verify the interaction between actual components of the system.
Unlike unit tests, these tests use minimal mocking and test the integration
between different parts of the system.

For proper integration testing, run with:
    RUN_INTEGRATION_TESTS=1 pytest rag-pipeline/tests/test_llm/test_integration.py -v
"""

import os
import pytest
import tempfile
import shutil
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from document_processor.pipeline import process_documents
from document_processor.loader import load_pdf_documents
from retriever.vector_store import load_vector_store, build_vector_store
from llm.assistant import OPTRagAssistant


# Check if integration tests should run
should_run = os.environ.get("RUN_INTEGRATION_TESTS", "0") == "1"
skip_integration = pytest.mark.skipif(
    not should_run,
    reason="Integration tests require RUN_INTEGRATION_TESTS=1"
)


@pytest.fixture
def test_document_path():
    """Return path to test document."""
    return Path(__file__).parent / "examples" / "sample_visa_info.txt"


@pytest.fixture
def temp_vector_store():
    """Create a temporary vector store directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Clean up
    shutil.rmtree(temp_dir)


@skip_integration
class TestRagPipelineIntegration:
    """Integration tests for the RAG Pipeline."""
    
    # @pytest.mark.asyncio
    # async def test_document_processing_pipeline(self, test_document_path, temp_vector_store):
    #     """Test the document processing pipeline with a real document."""
    #     # Process a real document
    #     result = await process_documents(
    #         source_path=test_document_path,
    #         vector_store_path=temp_vector_store,
    #         device="cpu",
    #         chunk_size=200,  # Smaller chunks for faster testing
    #         chunk_overlap=50
    #     )
        
    #     # Verify vector store was created
    #     assert temp_vector_store.exists()
    #     assert (temp_vector_store / "index.faiss").exists()
    #     assert (temp_vector_store / "content_hash.txt").exists()
        
    #     # Verify we can load the vector store
    #     vector_store = load_vector_store(
    #         vector_store_path=temp_vector_store,
    #         device="cpu"
    #     )
        
    #     # Verify vector store has content
    #     assert hasattr(vector_store, "_index")
    #     assert vector_store._index.ntotal > 0
    
    # @pytest.mark.asyncio
    # async def test_document_loading(self, test_document_path):
    #     """Test loading real documents."""
    #     # Load the text document
    #     documents = await load_pdf_documents(test_document_path)
        
    #     # Verify documents were loaded
    #     assert len(documents) > 0
    #     assert "OPT" in documents[0].page_content
    #     assert "Optional Practical Training" in documents[0].page_content
    
    # def test_vector_store_building(self, test_document_path, temp_vector_store):
    #     """Test building a vector store with actual text chunks."""
    #     # Create some text chunks
    #     chunks = ["OPT stands for Optional Practical Training.",
    #              "OPT allows international students to work in the United States.",
    #              "Students must be in F-1 status to apply for OPT.",
    #              "Post-completion OPT lasts for 12 months."]
        
    #     # Build vector store
    #     vector_store = build_vector_store(
    #         chunks=chunks,
    #         vector_store_path=temp_vector_store,
    #         device="cpu"
    #     )
        
    #     # Verify vector store was created
    #     assert (temp_vector_store / "index.faiss").exists()
        
    #     # Test retrieval
    #     retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    #     results = retriever.get_relevant_documents("How long does OPT last?")
        
    #     # Verify retrieval works
    #     assert len(results) == 2
    #     assert any("12 months" in doc.page_content for doc in results)
    
    # def test_end_to_end_with_mocked_model(self, test_document_path, test_model_path, temp_vector_store):
    #     """Test the end-to-end pipeline with a mocked model."""
    #     # Step 1: Process documents to create vector store
    #     # Using synchronous wrapper for simplicity in test
    #     from document_processor.pipeline import run_processing_pipeline
    #     run_processing_pipeline(
    #         source_path=test_document_path,
    #         vector_store_path=temp_vector_store,
    #         device="cpu",
    #         chunk_size=200,
    #         chunk_overlap=50
    #     )
        
    #     # Step 2: Initialize assistant with mocked model
    #     with patch("llm.assistant.AutoTokenizer.from_pretrained") as mock_tokenizer, \
    #          patch("llm.assistant.AutoModelForCausalLM.from_pretrained") as mock_model, \
    #          patch("llm.assistant.HuggingFacePipeline") as mock_llm, \
    #          patch("llm.assistant.pipeline") as mock_pipeline, \
    #          patch("llm.assistant.create_retrieval_chain") as mock_retrieval_chain:
            
    #         # Configure mock tokenizer
    #         tokenizer_instance = MagicMock()
    #         tokenizer_instance.pad_token_id = 0
    #         tokenizer_instance.eos_token_id = 1 
    #         mock_tokenizer.return_value = tokenizer_instance
            
    #         # Configure mock chain response
    #         mock_chain = MagicMock()
    #         mock_chain.invoke.return_value = {
    #             "answer": "OPT allows international students to work for 12 months after graduation.",
    #             "context": "Sample context from the vector store"
    #         }
    #         mock_retrieval_chain.return_value = mock_chain
            
    #         # Initialize assistant
    #         assistant = OPTRagAssistant(
    #             model_path=str(test_model_path),
    #             vector_store_path=str(temp_vector_store),
    #             device="cpu"
    #         )
            
    #         # Step 3: Test query
    #         result = assistant.answer_question("How long does OPT last?")
            
    #         # Verify result
    #         assert result is not None
    #         assert "answer" in result
    #         assert "OPT" in result["answer"]
    #         assert "12 months" in result["answer"]
            
    #         # Verify the vector store was used for retrieval
    #         mock_chain.invoke.assert_called_once()
    #         invoke_args = mock_chain.invoke.call_args[0][0]
    #         assert "question" in invoke_args
    #         assert invoke_args["question"] == "How long does OPT last?"
    
    # @pytest.mark.asyncio
    # async def test_document_addition_integration(self, test_document_path, test_model_path, temp_vector_store):
    #     """Test adding documents to an existing assistant."""
    #     # Setup mock model components
    #     with patch("llm.assistant.AutoTokenizer.from_pretrained"), \
    #          patch("llm.assistant.AutoModelForCausalLM.from_pretrained"), \
    #          patch("llm.assistant.HuggingFacePipeline"), \
    #          patch("llm.assistant.pipeline"), \
    #          patch("llm.assistant.create_retrieval_chain"):
            
    #         # Initialize assistant with empty vector store
    #         assistant = OPTRagAssistant(
    #             model_path=str(test_model_path),
    #             vector_store_path=str(temp_vector_store),
    #             device="cpu"
    #         )
            
    #         # Check initial vector store status
    #         initial_status = assistant.list_documents()
            
    #         # Add documents
    #         result = await assistant.add_documents(test_document_path)
            
    #         # Verify result
    #         assert result["status"] == "success"
    #         assert result["document_processed"] > 0
    #         assert result["chunks_created"] > 0
            
    #         # Verify vector store was updated
    #         final_status = assistant.list_documents()
    #         assert final_status["vector_count"] > initial_status["vector_count"]
    
    @pytest.mark.asyncio
    async def test_streaming_integration(self, test_document_path, test_model_path, temp_vector_store):
        """Test the streaming functionality with the integration of components."""
        # Step 1: Process documents to create vector store
        from document_processor.pipeline import process_documents
        # run_processing_pipeline(
        #     source_path=test_document_path,
        #     vector_store_path=temp_vector_store,
        #     device="cpu",
        #     chunk_size=200,
        #     chunk_overlap=50
        # )

        await process_documents(
            source_path=test_document_path,
            vector_store_path=temp_vector_store,
            device="cpu",
            chunk_size=200,
            chunk_overlap=50
        )
        
        # Step 2: Setup mocks for streaming
        with patch("llm.assistant.AutoTokenizer.from_pretrained"), \
             patch("llm.assistant.AutoModelForCausalLM.from_pretrained"), \
             patch("llm.assistant.AsyncStreamingCallbackHandler") as mock_handler_class, \
             patch("llm.assistant.HuggingFacePipeline"), \
             patch("llm.assistant.pipeline"), \
             patch("asyncio.create_task") as mock_create_task:
            
            # Configure mock streaming handler
            mock_handler = AsyncMock()
            mock_handler_class.return_value = mock_handler
            
            # Setup mock queue
            mock_queue = AsyncMock()
            mock_queue.get = AsyncMock(side_effect=[
                "OPT ", "stands ", "for ", "Optional ", "Practical ", "Training", 
                asyncio.TimeoutError
            ])
            mock_queue.task_done = MagicMock()
            
            # Create a completed future for the task
            mock_future = asyncio.Future()
            mock_future.set_result(None)
            mock_create_task.return_value = mock_future
            
            # Initialize assistant
            assistant = OPTRagAssistant(
                model_path=str(test_model_path),
                vector_store_path=str(temp_vector_store),
                device="cpu"
            )
            
            # Replace asyncio.Queue with our mock
            with patch("asyncio.Queue", return_value=mock_queue):
                # Collect streaming tokens
                tokens = []
                async for token in assistant.astream_response("What is OPT?"):
                    tokens.append(token)
                
                # Verify streaming behavior
                assert len(tokens) > 1
                assert tokens[0] == "Searching visa regulations...\n\n"
                
                # Combine remaining tokens to verify content
                response_text = "".join(tokens[1:])
                assert "OPT" in response_text
                assert "Optional Practical Training" in response_text


# @skip_integration
# class TestRagFailureScenarios:
#     """Test various failure scenarios in the RAG pipeline."""
    
#     def test_missing_document_handling(self, temp_vector_store, test_model_path):
#         """Test handling of missing documents."""
#         non_existent_file = Path("this_file_does_not_exist.pdf")
        
#         with patch("llm.assistant.AutoTokenizer.from_pretrained"), \
#              patch("llm.assistant.AutoModelForCausalLM.from_pretrained"), \
#              patch("llm.assistant.MODEL_LOAD_TIME"), \
#              patch("llm.assistant.VECTOR_COUNT"):
            
#             # Initialize assistant
#             assistant = OPTRagAssistant(
#                 model_path=str(test_model_path),
#                 vector_store_path=str(temp_vector_store),
#                 device="cpu"
#             )
            
#             # Test adding non-existent document
#             with pytest.raises(FileNotFoundError):
#                 asyncio.run(assistant.add_documents(non_existent_file))
    
#     def test_empty_vector_store_query(self, test_model_path, temp_vector_store):
#         """Test querying with an empty vector store."""
#         with patch("llm.assistant.AutoTokenizer.from_pretrained"), \
#              patch("llm.assistant.AutoModelForCausalLM.from_pretrained"), \
#              patch("llm.assistant.HuggingFacePipeline") as mock_llm, \
#              patch("llm.assistant.pipeline") as mock_pipeline, \
#              patch("llm.assistant.create_retrieval_chain") as mock_retrieval_chain, \
#              patch("llm.assistant.MODEL_LOAD_TIME"), \
#              patch("llm.assistant.VECTOR_COUNT"):
            
#             # Setup mock for empty retrieval
#             mock_chain = MagicMock()
#             mock_chain.invoke.return_value = {
#                 "answer": "I don't have specific information about OPT in my knowledge base.",
#                 "context": ""
#             }
#             mock_retrieval_chain.return_value = mock_chain
            
#             # Initialize assistant with empty vector store
#             assistant = OPTRagAssistant(
#                 model_path=str(test_model_path),
#                 vector_store_path=str(temp_vector_store),
#                 device="cpu"
#             )
            
#             # Test query
#             result = assistant.answer_question("What is OPT?")
            
#             # Verify result acknowledges lack of information
#             assert "I don't have specific information" in result["answer"]


# @skip_integration
# class TestRagPerformanceMetrics:
#     """Test performance metrics tracking in the RAG pipeline."""
    
#     def test_metrics_tracking(self, test_document_path, test_model_path, temp_vector_store):
#         """Test that metrics are properly tracked during operation."""
#         # Process documents
#         from document_processor.pipeline import run_processing_pipeline
#         run_processing_pipeline(
#             source_path=test_document_path,
#             vector_store_path=temp_vector_store,
#             device="cpu",
#             chunk_size=200,
#             chunk_overlap=50
#         )
        
#         # Setup mock metrics collectors
#         with patch("llm.assistant.AutoTokenizer.from_pretrained"), \
#              patch("llm.assistant.AutoModelForCausalLM.from_pretrained"), \
#              patch("llm.assistant.HuggingFacePipeline"), \
#              patch("llm.assistant.pipeline"), \
#              patch("llm.assistant.create_retrieval_chain") as mock_retrieval_chain, \
#              patch("llm.assistant.QUERY_COUNT.labels") as mock_query_count, \
#              patch("llm.assistant.QUERY_LATENCY.observe") as mock_latency_observe, \
#              patch("llm.assistant.VECTOR_RETRIEVAL_LATENCY.observe") as mock_retrieval_latency:
            
#             # Configure mock chain response
#             mock_chain = MagicMock()
#             mock_chain.invoke.return_value = {
#                 "answer": "OPT stands for Optional Practical Training.",
#                 "context": "Sample context"
#             }
#             mock_retrieval_chain.return_value = mock_chain
            
#             # Setup query count mock to return a counter mock
#             mock_counter = MagicMock()
#             mock_counter.inc = MagicMock()
#             mock_query_count.return_value = mock_counter
            
#             # Initialize assistant
#             assistant = OPTRagAssistant(
#                 model_path=str(test_model_path),
#                 vector_store_path=str(temp_vector_store),
#                 device="cpu"
#             )
            
#             # Execute query
#             result = assistant.answer_question("What is OPT?")
            
#             # Verify metrics were tracked
#             # Query count metrics
#             mock_query_count.assert_any_call(status="started", query_type="standard")
#             mock_query_count.assert_any_call(status="completed", query_type="standard")
#             mock_counter.inc.assert_called()
            
#             # Latency metrics
#             mock_latency_observe.assert_called_once()
#             mock_retrieval_latency.assert_called_once()
            
#             # Verify processing time was recorded in response
#             assert "processing_time" in result
#             assert isinstance(result["processing_time"], float)
