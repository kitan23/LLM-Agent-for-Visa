"""
Unit tests for the OPT-RAG Assistant class.

These tests verify the functionality of the OPTRagAssistant class using mock 
objects and fixtures. For interactive testing with real models, use the 
manual_test_assistant.py script in the project root.
"""

import pytest
import time
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

# Import the module under test
from llm.assistant import OPTRagAssistant


class TestOPTRagAssistant:
    """Test suite for the OPTRagAssistant class."""
    
    @pytest.fixture
    def mock_assistant_dependencies(self, mock_tokenizer, mock_model, mock_vector_store):
        """Set up mocks for assistant dependencies."""
        with patch("llm.assistant.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("llm.assistant.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
             patch("llm.assistant.load_vector_store", return_value=mock_vector_store), \
             patch("llm.assistant.VECTOR_COUNT"), \
             patch("llm.assistant.MODEL_LOAD_TIME"):
            yield
    
    def test_init(self, test_model_path, test_vector_store_path, mock_assistant_dependencies, mock_hardware_detection):
        """Test successful initialization of the assistant."""
        # Arrange & Act
        assistant = OPTRagAssistant(
            model_path=str(test_model_path),
            vector_store_path=str(test_vector_store_path),
            device="cpu"
        )
        
        # Assert
        assert assistant is not None
        assert assistant.model_path == Path(test_model_path)
        assert assistant.vector_store_path == Path(test_vector_store_path)
        assert assistant.device == "cpu"
        
        # Verify components were initialized
        assert assistant.tokenizer is not None
        assert assistant.model is not None
        assert assistant.vector_store is not None
    
    def test_detect_hardware(self, mock_assistant_dependencies):
        """Test hardware detection logic."""
        with patch("torch.backends.mps.is_available", return_value=False), \
            patch("torch.cuda.is_available", return_value=False), \
            patch("pathlib.Path.exists", return_value=True):

            assistant = OPTRagAssistant(
                model_path="mock_path",
                vector_store_path="mock_path",
                device=None  # triggers hardware detection
            )

            assert assistant.device == "cpu"
    
    # @pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
    # def test_load_model(self, device, test_model_path, test_vector_store_path):
    #     """Test model loading for different devices."""
    #     # Arrange
    #     with patch("llm.assistant.AutoTokenizer.from_pretrained"), \
    #          patch("llm.assistant.AutoModelForCausalLM.from_pretrained") as mock_load_model, \
    #          patch("llm.assistant.load_vector_store"), \
    #          patch("llm.assistant.VECTOR_COUNT"), \
    #          patch("llm.assistant.MODEL_LOAD_TIME"), \
    #          patch("llm.assistant.OPTRagAssistant._detect_hardware", return_value=device):
            
    #         # Act
    #         assistant = OPTRagAssistant(
    #             model_path=str(test_model_path),
    #             vector_store_path=str(test_vector_store_path)
    #         )
            
    #         # Assert
    #         mock_load_model.assert_called_once()
            
    #         # Check if quantization was configured correctly based on device
    #         if device == "cuda":
    #             # Should use 4-bit quantization for CUDA
    #             assert "quantization_config" in mock_load_model.call_args[1]
    #             assert mock_load_model.call_args[1]["quantization_config"].load_in_4bit is True
    #         elif device == "mps":
    #             # Should use float16 for MPS
    #             assert "torch_dtype" in mock_load_model.call_args[1]
    #         elif device == "cpu":
    #             # Should use 8-bit quantization for CPU
    #             assert "quantization_config" in mock_load_model.call_args[1]
    #             assert mock_load_model.call_args[1]["quantization_config"].load_in_8bit is True
    
    def test_answer_question(self, test_model_path, test_vector_store_path, mock_assistant_dependencies):
        """Test question answering functionality."""
        # Arrange
        with patch("llm.assistant.HuggingFacePipeline") as mock_llm, \
             patch("llm.assistant.pipeline") as mock_pipeline, \
             patch("llm.assistant.create_stuff_documents_chain") as mock_create_chain, \
             patch("llm.assistant.create_retrieval_chain") as mock_retrieval_chain, \
             patch("llm.assistant.QUERY_COUNT"), \
             patch("llm.assistant.QUERY_LATENCY"), \
             patch("llm.assistant.VECTOR_RETRIEVAL_LATENCY"):
                
            # Setup mock chain response
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = {
                "answer": "This is a test answer about visa regulations.",
                "context": "Sample context"
            }
            mock_retrieval_chain.return_value = mock_chain
            
            assistant = OPTRagAssistant(
                model_path=str(test_model_path),
                vector_store_path=str(test_vector_store_path),
                device="cpu"
            )
            
            # Act
            result = assistant.answer_question("What are OPT visa requirements?")
            
            # Assert
            assert result is not None
            assert "answer" in result
            assert result["answer"] == "This is a test answer about visa regulations."
            assert "processing_time" in result
            mock_chain.invoke.assert_called_once_with({"question": "What are OPT visa requirements?"})
    
    # @pytest.mark.asyncio
    # async def test_add_documents(self, test_model_path, test_vector_store_path, mock_assistant_dependencies):
    #     """Test document addition functionality."""
    #     # Arrange
    #     with patch("llm.assistant.process_documents", new_callable=AsyncMock) as mock_process, \
    #          patch("llm.assistant.load_vector_store") as mock_load_store, \
    #          patch("llm.assistant.VECTOR_COUNT"):
            
    #         # Setup mock process_documents response
    #         mock_process.return_value = {
    #             "documents": ["doc1", "doc2"],
    #             "chunks": ["chunk1", "chunk2", "chunk3"]
    #         }
            
    #         # Create a refreshed mock vector store for the reload
    #         refreshed_store = MagicMock()
    #         refreshed_store._index = MagicMock()
    #         refreshed_store._index.ntotal = 103  # Original 100 + 3 new chunks
    #         mock_load_store.return_value = refreshed_store
            
    #         assistant = OPTRagAssistant(
    #             model_path=str(test_model_path),
    #             vector_store_path=str(test_vector_store_path),
    #             device="cpu"
    #         )
            
    #         # Act
    #         result = await assistant.add_documents("test_document.pdf")
            
    #         # Assert
    #         assert result["status"] == "success"
    #         assert result["document_processed"] == 2
    #         assert result["chunks_created"] == 3
    #         assert "processing_time" in result
    #         mock_process.assert_called_once_with(
    #             source_path="test_document.pdf", 
    #             vector_store_path=Path(test_vector_store_path),
    #             device="cpu",
    #             chunk_size=1000,
    #             chunk_overlap=200
    #         )
    #         # Check if vector store was reloaded after adding documents
    #         mock_load_store.assert_called_with(
    #             vector_store_path=Path(test_vector_store_path),
    #             device="cpu",
    #             force_reload=True
    #         )
    
    # @pytest.mark.asyncio
    # async def test_astream_response(self, test_model_path, test_vector_store_path, mock_assistant_dependencies):
    #     """Test streaming response functionality."""
    #     # Arrange
    #     with patch("llm.assistant.AsyncStreamingCallbackHandler") as mock_handler_class, \
    #          patch("llm.assistant.AsyncCallbackManager"), \
    #          patch("llm.assistant.HuggingFacePipeline") as mock_llm, \
    #          patch("llm.assistant.QUERY_COUNT"), \
    #          patch("llm.assistant.QUERY_LATENCY"):
            
    #         # Setup mock streaming
    #         mock_handler = AsyncMock()
    #         mock_handler_class.return_value = mock_handler
            
    #         # Mock queue for tokens
    #         mock_queue = AsyncMock()
    #         mock_queue.get = AsyncMock(side_effect=[
    #             "First token", 
    #             "Second token", 
    #             asyncio.TimeoutError  # To exit the streaming loop
    #         ])
            
    #         # Setup mock LLM
    #         mock_llm_instance = AsyncMock()
    #         mock_llm.return_value = mock_llm_instance
    #         mock_llm_instance.ainvoke = AsyncMock()
            
    #         # Create a mock task that completes after first timeout
    #         class MockTask:
    #             def __init__(self):
    #                 self._call_count = 0
                
    #             def done(self):
    #                 self._call_count += 1
    #                 return self._call_count > 1
            
    #         mock_task = MockTask()
            
    #         with patch("asyncio.Queue", return_value=mock_queue), \
    #              patch("asyncio.create_task", return_value=mock_task):
                
    #             assistant = OPTRagAssistant(
    #                 model_path=str(test_model_path),
    #                 vector_store_path=str(test_vector_store_path),
    #                 device="cpu"
    #             )
                
    #             # Act
    #             tokens = []
    #             async for token in assistant.astream_response("What are OPT visa requirements?"):
    #                 tokens.append(token)
                
    #             # Assert
    #             assert len(tokens) == 3  # Initial message + 2 tokens
    #             assert tokens[0] == "Searching visa regulations...\n\n"
    #             assert tokens[1] == "First token"
    #             assert tokens[2] == "Second token" 