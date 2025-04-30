"""
Integration tests for OPT-RAG Assistant.

These tests use the actual document processing and vector store components
but mock the LLM to avoid heavy computation during testing.
"""

import os
import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

from llm.assistant import OPTRagAssistant


@pytest.fixture
def test_documents_path():
    """Path to test documents."""
    return Path(__file__).parent / "examples"


@pytest.fixture
def sample_visa_doc(test_documents_path):
    """Path to a sample visa document."""
    doc_path = test_documents_path / "sample_visa_info.txt"
    if not doc_path.exists():
        pytest.skip(f"Sample document not found at {doc_path}")
    return doc_path


class TestOPTRagIntegration:
    """Integration tests for OPT-RAG Assistant."""
    
    @pytest.mark.skipif(
        not os.environ.get("RUN_INTEGRATION_TESTS"), 
        reason="Integration tests are not enabled. Set RUN_INTEGRATION_TESTS=1 to run."
    )
    @pytest.mark.asyncio
    async def test_end_to_end_document_to_query(self, test_model_path, test_vector_store_path, sample_visa_doc):
        """Test the complete flow from document ingestion to query."""
        # Skip if integration tests aren't explicitly enabled
        
        # Mock LLM components but use real document processing and vector storage
        with patch("llm.assistant.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
             patch("llm.assistant.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("llm.assistant.pipeline") as mock_pipeline, \
             patch("llm.assistant.HuggingFacePipeline") as mock_llm_cls, \
             patch("llm.assistant.torch.cuda.is_available", return_value=False), \
             patch("llm.assistant.torch.backends.mps.is_available", return_value=False):
            
            # Setup necessary mocks for the LLM part
            mock_tokenizer = MagicMock()
            mock_tokenizer_cls.return_value = mock_tokenizer
            
            mock_model = MagicMock()
            mock_model_cls.return_value = mock_model
            
            mock_pipe = MagicMock()
            mock_pipeline.return_value = mock_pipe
            
            mock_llm = MagicMock()
            mock_llm_cls.return_value = mock_llm
            
            # Setup mock chain response with expected format
            with patch("llm.assistant.create_stuff_documents_chain") as mock_create_chain, \
                 patch("llm.assistant.create_retrieval_chain") as mock_retrieval_chain:
                
                mock_chain = MagicMock()
                mock_chain.invoke.return_value = {
                    "answer": "OPT stands for Optional Practical Training. It allows F-1 students to work for up to 12 months in their field of study.",
                    "context": "Sample context from the document"
                }
                mock_retrieval_chain.return_value = mock_chain
                
                # Initialize assistant
                assistant = OPTRagAssistant(
                    model_path=str(test_model_path), 
                    vector_store_path=str(test_vector_store_path),
                    device="cpu"
                )
                
                # 1. Add documents
                add_result = await assistant.add_documents(str(sample_visa_doc))
                assert add_result["status"] == "success"
                assert add_result["document_processed"] > 0
                assert add_result["chunks_created"] > 0
                
                # 2. List documents to verify they were added
                list_result = assistant.list_documents()
                assert list_result["status"] == "success"
                assert list_result["vector_count"] > 0
                
                # 3. Query the assistant
                query_result = assistant.answer_question("What is OPT and how long does it last?")
                
                # 4. Verify the response
                assert "answer" in query_result
                assert "processing_time" in query_result
                
                # Verify proper chain context was built from our document
                prompt_context = mock_chain.invoke.call_args[0][0]["question"]
                assert "OPT" in prompt_context
    
    @pytest.mark.skipif(
        not os.environ.get("RUN_INTEGRATION_TESTS"), 
        reason="Integration tests are not enabled. Set RUN_INTEGRATION_TESTS=1 to run."
    )
    @pytest.mark.asyncio
    async def test_streaming_with_real_documents(self, test_model_path, test_vector_store_path, sample_visa_doc):
        """Test streaming responses with real documents but mocked LLM."""
        # Mock just the LLM components
        with patch("llm.assistant.AutoTokenizer.from_pretrained") as mock_tokenizer_cls, \
             patch("llm.assistant.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
             patch("llm.assistant.pipeline") as mock_pipeline, \
             patch("llm.assistant.HuggingFacePipeline") as mock_llm_cls, \
             patch("llm.assistant.torch.cuda.is_available", return_value=False), \
             patch("llm.assistant.torch.backends.mps.is_available", return_value=False):
            
            # Setup mocks
            mock_tokenizer = MagicMock()
            mock_tokenizer_cls.return_value = mock_tokenizer
            
            mock_model = MagicMock()
            mock_model_cls.return_value = mock_model
            
            mock_pipe = MagicMock()
            mock_pipeline.return_value = mock_pipe
            
            mock_llm = MagicMock()
            mock_llm.ainvoke = MagicMock(return_value="This is a streaming response about OPT.")
            mock_llm_cls.return_value = mock_llm
            
            # Mock queue for streaming
            queue = asyncio.Queue()
            await queue.put("First token")
            await queue.put("Second token")
            await queue.put("Final token about OPT duration.")
            
            # Initialize assistant
            assistant = OPTRagAssistant(
                model_path=str(test_model_path), 
                vector_store_path=str(test_vector_store_path),
                device="cpu"
            )
            
            # 1. Add documents
            add_result = await assistant.add_documents(str(sample_visa_doc))
            assert add_result["status"] == "success"
            
            # 2. Test streaming response
            with patch("asyncio.Queue", return_value=queue), \
                 patch("asyncio.create_task") as mock_create_task:
                
                # Mock task completion
                mock_task = MagicMock()
                mock_task.done.side_effect = [False, False, False, True]
                mock_create_task.return_value = mock_task
                
                # Collect streaming tokens
                tokens = []
                async for token in assistant.astream_response("What is the duration of OPT?"):
                    tokens.append(token)
                
                # Verify streaming response
                assert len(tokens) > 0
                assert any("OPT" in token for token in tokens) 