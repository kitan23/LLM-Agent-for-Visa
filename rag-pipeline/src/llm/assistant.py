"""
OPT-RAG Assistant Implementation

This module contains the core implementation of the OPT-RAG International Student 
Visa Assistant, handling model loading, vector store management, and query processing.
"""


import logging
import time
import asyncio
import threading
from typing import Dict, Any, Optional, AsyncIterator, Iterator, Union, List
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from langchain_core.prompts import ChatPromptTemplate

from .callbacks import AsyncStreamingCallbackHandler
from ..utils.metrics import QUERY_LATENCY, QUERY_COUNT, MODEL_LOAD_TIME, VECTOR_COUNT, VECTOR_RETRIEVAL_LATENCY, QUERY_ERRORS
from ..retriever.vector_store import load_vector_store, build_vector_store
from ..document_processor.pipeline import process_documents

logger = logging.getLogger("opt_rag.assistant")


class OPTRagAssistant: 
    """OPT-RAG International Student Visa Assistant using RAG architecture."""

    def __init__(
        self, 
        model_path: str, 
        vector_store_path: str, 
        device: Optional[str] = None, 
        embedding_model_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    ): 
        """Initialize the OPT-RAG Assistant.
        
        Args:
            model_path: Path to the downloaded LLM model
            vector_store_path: Path to FAISS vector store
            device: Device to use (cuda, mps, cpu, or None for auto-detection)
        """
        start_time = time.time()
        self.model_path = Path(model_path)
        self.vector_store_path = Path(vector_store_path)
        self.embedding_model_name = embedding_model_name

        # Detect hardware if not specified
        self.device = device or self._detect_hardware()

        logger.info(f"Using device: {self.device}")

        # initialize components 
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.vector_store = self._load_vector_store()

        # Store prompt template
        self.visa_prompt = self._create_prompt_template()

        # Record load time 
        load_time = time.time() - start_time
        MODEL_LOAD_TIME.observe(load_time)


        # Update vector count metric 
        if hasattr(self.vector_store, "index"):
            VECTOR_COUNT.set(self.vector_store.index.ntotal)

        logger.info("OPT-RAG Assistant initialized successfully")

    def _detect_hardware(self) -> str:
        """Detect the hardware device to use for inference."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _load_tokenizer(self):
        """Load the tokenizer."""
        logger.info(f"Loading tokenizer from {self.model_path}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        return AutoTokenizer.from_pretrained(self.model_path)
    
    def _load_model(self):
        """Load and configure the model."""
        logger.info(f"Loading model from {self.model_path}")

        try: 
            if self.device == "cuda":
                # NVIDIA GPU configuration with 4-bit quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    quantization_config=quantization_config
                )
                
                # Compile for additional performance if supported
                if torch.__version__ >= "2.0.0":
                    try:
                        logger.info("Compiling model for optimized execution")
                        model = torch.compile(model)
                    except Exception as e:
                        logger.warning(f"Model compilation failed, using uncompiled model: {e}")
                
                logger.info("Model loaded successfully")
                
            elif self.device == "mps":
                # Apple Silicon configuration
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                
            else:
                # CPU configuration - 8-bit quantization for better memory usage
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    quantization_config=quantization_config
                )
            
            return model
        
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise 

    
    def _load_vector_store(self):
        """Load the FAISS vector store if it exists, or create an empty one."""
        try: 
            return load_vector_store(
                vector_store_path = self.vector_store_path,
                device = self.device, 
                force_reload = False
            )
        
        except (FileNotFoundError, RuntimeError):
            logger.warning(f"Vector store not found at {self.vector_store_path}, creating new one")
            
            # Create directory if it doesn't exist
            self.vector_store_path.mkdir(parents=True, exist_ok=True)

            # Return empty vector store 
            return build_vector_store(
                chunks = ["This is a dummy chunk."],
                vector_store_path=self.vector_store_path,
                device = self.device, 
            )
        
    def _create_prompt_template(self):
        """Create prompt template for visa assistance."""
        return ChatPromptTemplate.from_template("""
        You are OPT-RAG, an expert assistant specializing in international student visa regulations and processes in the United States.

        ## ROLE AND GUIDELINES
        - ONLY provide information that is explicitly supported by the context below
        - DO NOT make any claims or assertions that aren't directly supported by the provided context
        - Focus specifically on visa-related issues: OPT applications, CPT authorization, study/work permits, and visa status questions
        - If information is not available in the context, clearly state "Based on the provided context, I don't have specific information about that"
        - Never fabricate information or provide speculative advice on visa matters
        - When answering, always check if your response contradicts any information in the context - if it does, defer to the context
        - Always indicate the source of information in your responses
        - Avoid legal advice; clarify when questions require consultation with immigration attorneys

        ## CONTEXT
        {context}

        ## USER INFORMATION
        Student status: International student in the United States
        Primary concern: Visa and immigration matters

        ## RESPONSE FORMAT
        - Begin with a direct and factually accurate answer to the question based ONLY on the context provided
        - Provide specific, relevant details from official sources in the context
        - Include citation to specific documents/policies when available
        - Highlight important deadlines or requirements mentioned in the context
        - If applicable, mention next steps the student should take according to the context
        - End with a disclaimer that this information is not legal advice

        ## QUESTION
        {question}
        """)

    async def add_documents(self, 
                      file_path: Union[str, List[str]], 
                      document_type: Optional[str] = None, 
                      chunk_size: int = 1000, 
                      chunk_overlap: int = 200 
                      ) -> Dict[str, Any]:
        """Add documents to the vector store.


        Args: 
            file_path: Path to the document file or list of file paths
            document_type: Type of document (e.g., "policy", "faq", "news")
            chunk_size: Number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks

        Returns: 
            Dictionary containing the number of chunks added and the total number of chunks in the vector store
        """

        logger.info(f"Adding documents from {file_path}")
        start_time = time.time()

        try: 
            # Measure vector count updates 
            before_count = 0 
            if hasattr(self.vector_store, "index"):
                before_count = self.vector_store.index.ntotal
            
            # call async process_documents 
            result = await process_documents(
                source_path = file_path, 
                vector_store_path = self.vector_store_path, 
                device = self.device, 
                chunk_size = chunk_size, 
                chunk_overlap = chunk_overlap, 
            )
            
            # Check status of document processing
            if result.get("status") == "error":
                return {
                    "status": "error",
                    "error": result.get("error", "Unknown error during document processing")
                }
            
            # Extract processing information from result
            document_count = result.get("document_count", 0)
            chunk_count = result.get("chunk_count", 0)
            
            # Get the vector store from the result
            vector_store = result.get("vector_store")
            if vector_store is None:
                return {
                    "status": "error",
                    "error": "No vector store returned from processing"
                }
                
            # Reload vector store after adding documents
            self.vector_store = load_vector_store(
                vector_store_path=self.vector_store_path, 
                device=self.device, 
                force_reload=True
            )

            after_count = 0 
            # Update vector count metric 
            if hasattr(self.vector_store, "index"):
                after_count = self.vector_store.index.ntotal
                VECTOR_COUNT.set(after_count)
            
            processing_time = time.time() - start_time 
            logger.info(f"Documents added successfully in {processing_time:.2f} seconds")

            vectors_added = after_count - before_count
            
            return {
                "status": "success", 
                "documents_processed": document_count,
                "chunks_created": chunk_count,
                "processing_time": processing_time,
                "vectors_added": vectors_added
            }
        
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {
                "status": "error", 
                "error": str(e)
            }
            
    def remove_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """Remove documents from the vector store.
        
        Args: 
            document_ids: List of document IDs to remove

        Returns:
            Dictionary with processing information
        """

        # This would require implementation in your vector_store.py
        # For now, we'll just log that this isn't implemented
        logger.warning("Document removal not implemented yet")
        return {
            "status": "error",
            "error": "Document removal not implemented yet"
        }

            
    def list_documents(self) -> Dict[str, Any]:
        """List all documents in the vector store.
        
        Returns:
            Dictionary with document information
        """
        # This would require implementation in your vector_store.py
        # For now, we'll just return basic info
        if hasattr(self.vector_store, 'index'):
            return {
                "status": "success",
                "vector_count": self.vector_store.index.ntotal,
                "vector_store_path": str(self.vector_store_path)
            }
        return {
            "status": "success",
            "vector_count": 0,
            "vector_store_path": str(self.vector_store_path)
        }   
    

    def answer_question(self, query: str, stream: bool = False) -> Dict[str, Any]:
        """Process a user query and return a response with relevant context.
        
        Args:
            query: User's question about visa matters
            stream: Whether to stream the response
            
        Returns:
            Dict containing the answer and retrieved documents
        """
        logger.info(f"Processing query: {query}")
        QUERY_COUNT.labels(status="started", query_type="standard").inc()
        
        try:
            start_time = time.time()

            # Create retriever
            retrieval_start_time = time.time()
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Retrieve top 5 most relevant documents
            )

            # Retrieve documents
            retrieval_result = retriever.invoke(query)
            docs = retrieval_result if isinstance(retrieval_result, list) else retrieval_result['documents']
            
            # Measure retrieval time 
            VECTOR_RETRIEVAL_LATENCY.observe(time.time() - retrieval_start_time)

            # Format documents for context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create prompt from template
            prompt_text = self.visa_prompt.format(
                question=query,
                context=context
            )
            
            # Tokenize the input
            inputs = self.tokenizer(prompt_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)
            
            # Generate the response directly using the model
            if stream:
                # Use streaming mode (handled by astream_response method)
                # This branch should not normally be taken, but is here for completeness
                streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)
                
                # Create generation kwargs
                generation_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "streamer": streamer,
                    "max_new_tokens": 512,
                    "do_sample": True,
                    "temperature": 0.01,
                    "repetition_penalty": 1.1,
                }
                
                # Run generation in a separate thread
                thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()
                
                # Collect all tokens
                response_text = ""
                for new_text in streamer:
                    response_text += new_text
                
                # Ensure thread completes
                thread.join()
                
                answer = response_text
            else:
                # Generate without streaming
                output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.01,
                    repetition_penalty=1.1,
                )
                
                # Decode the generated output, skipping the prompt
                answer = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            # Create response dictionary
            response = {
                "answer": answer,
                "source_documents": docs
            }

            # Log performance metrics
            elapsed_time = time.time() - start_time
            logger.info(f"Query processed in {elapsed_time:.2f} seconds")
            QUERY_LATENCY.observe(elapsed_time)
            
            # Add timing information to response
            response["processing_time"] = elapsed_time
            
            QUERY_COUNT.labels(status="completed", query_type="standard").inc()

            return response

        except Exception as e:
            # Track error properly 
            QUERY_ERRORS.labels(error_type = type(e).__name__).inc()
            QUERY_COUNT.labels(status="error", query_type="standard").inc()
            logger.error(f"Error answering question: {e}")
            return {
                "answer": "I encountered an error while processing your question. Please try again.", 
                "error": str(e)
            }
        

    async def astream_response(self, query: str) -> AsyncIterator[str]:
        """Stream response asynchronously for use with FastAPI.
        
        Args:
            query: User's question about visa matters
            
        Yields:
            Tokens as they're generated
        """
        QUERY_COUNT.labels(status="started", query_type="streaming").inc()
        
        try:
            start_time = time.time()
            
            # First yield to show we're processing
            yield "Searching visa regulations...\n\n"
            
            # Process documents retrieval first (this isn't streamed)
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Use .invoke instead of get_relevant_documents
            retrieval_result = retriever.invoke(query)
            docs = retrieval_result if isinstance(retrieval_result, list) else retrieval_result['documents']
            
            # Format documents for context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create prompt from template
            prompt_text = self.visa_prompt.format(
                question=query,
                context=context
            )
            
            # Create a custom StreamingOutputCallback that yields tokens
            # Set skip_prompt=True to skip the input prompt in the output
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)
            
            # Tokenize the input
            inputs = self.tokenizer(prompt_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)
            
            # Create the generation kwargs
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "streamer": streamer,
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 0.01,
                "repetition_penalty": 1.1,
            }
            
            # Run generation in a separate thread to avoid blocking
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream the output as it's generated
            for new_text in streamer:
                yield new_text
            
            # Ensure thread completes
            thread.join()
            
            # Log performance metrics
            elapsed_time = time.time() - start_time
            logger.info(f"Streaming query processed in {elapsed_time:.2f} seconds")
            QUERY_LATENCY.observe(elapsed_time)
            QUERY_COUNT.labels(status="completed", query_type="streaming").inc()

        except Exception as e:
            # Track error properly 
            QUERY_ERRORS.labels(error_type=type(e).__name__).inc()
            QUERY_COUNT.labels(status="error", query_type="streaming").inc()
            logger.error(f"Error streaming response: {e}")
            yield f"\n I'm sorry, an error occurred: {str(e)}"
        


