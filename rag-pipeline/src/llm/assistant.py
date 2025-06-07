"""
OPT-RAG Assistant Implementation

This module contains the core implementation of the OPT-RAG International Student 
Visa Assistant, handling model loading, vector store management, and query processing.

UPDATED: Now supports both local model inference and API-based inference (OpenAI, etc.)
Set USE_API_LLM=true environment variable to use API-based mode.
"""


import logging
import time
import asyncio
import threading
import os
from typing import Dict, Any, Optional, AsyncIterator, Iterator, Union, List
from pathlib import Path
from threading import Event

# ===== LOCAL MODEL IMPORTS (Original Implementation) =====
# These imports are used when USE_API_LLM=false (default behavior)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from langchain_core.prompts import ChatPromptTemplate
from opentelemetry import trace

# ===== API-BASED MODEL IMPORTS (New Implementation) =====
# These imports are used when USE_API_LLM=true
import openai
from openai import OpenAI
import httpx
import json

from src.llm.callbacks import AsyncStreamingCallbackHandler
from src.llm.direct_streamer import DirectStreamer
from src.utils.metrics import QUERY_LATENCY, QUERY_COUNT, MODEL_LOAD_TIME, VECTOR_COUNT, VECTOR_RETRIEVAL_LATENCY, QUERY_ERRORS
from src.retriever.vector_store import load_vector_store, build_vector_store
from src.document_processor.pipeline import process_documents
from src.utils.tracing import get_tracer
from src.utils.config import get_settings

logger = logging.getLogger("opt_rag.assistant")

# Get a tracer for this module
tracer = get_tracer("opt_rag.assistant")


class OPTRagAssistant: 
    """OPT-RAG International Student Visa Assistant using RAG architecture.
    
    UPDATED: Now supports both local model inference and API-based inference.
    - Set USE_API_LLM=true to use API-based mode (OpenAI, Anthropic, etc.)
    - Set USE_API_LLM=false (default) to use local model mode
    """

    def __init__(
        self, 
        model_path: str, 
        vector_store_path: str, 
        device: Optional[str] = None, 
        embedding_model_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    ): 
        """Initialize the OPT-RAG Assistant.
        
        Args:
            model_path: Path to the downloaded LLM model (used only in local mode)
            vector_store_path: Path to FAISS vector store
            device: Device to use (cuda, mps, cpu, or None for auto-detection) - used only in local mode
            embedding_model_name: Sentence transformer model for embeddings
        """
        with tracer.start_as_current_span("initialize_opt_rag_assistant"):
            start_time = time.time()
            
            # Load settings
            self.settings = get_settings()
            
            # Common initialization for both modes
            self.vector_store_path = Path(vector_store_path)
            self.embedding_model_name = embedding_model_name

            # Check if we should use API-based LLM from settings
            self.use_api_llm = self.settings.use_api_llm
            
            if self.use_api_llm:
                logger.info("=== USING API-BASED LLM MODE ===")
                self._init_api_mode()
                # Set model_path to None since we don't need it in API mode
                self.model_path = None
            else:
                logger.info("=== USING LOCAL MODEL MODE ===")
                self.model_path = Path(model_path)
                self._init_local_mode(device)

            # Common initialization: Vector store (used by both modes)
            self.vector_store = self._load_vector_store()

            # Store prompt template (used by both modes)
            self.visa_prompt = self._create_prompt_template()

            # Record load time 
            load_time = time.time() - start_time
            MODEL_LOAD_TIME.observe(load_time)

            # Update vector count metric 
            if hasattr(self.vector_store, "index"):
                VECTOR_COUNT.set(self.vector_store.index.ntotal)

            logger.info("OPT-RAG Assistant initialized successfully")

    def _init_api_mode(self):
        """Initialize API-based LLM mode."""
        logger.info("Initializing API-based LLM mode")
        
        # Get API configuration from settings
        self.api_provider = self.settings.llm_api_provider
        self.api_model = self.settings.llm_api_model
        self.api_base_url = self.settings.llm_api_base_url
        
        # FIXED: Directly read API key from environment variables with explicit fallback
        self.api_key = (
            os.environ.get("OPENAI_API_KEY") or 
            os.environ.get("OPT_RAG_LLM_API_KEY") or
            self.settings.llm_api_key
        )
        
        if not self.api_key:
            raise ValueError("API key not found! Please set OPENAI_API_KEY or OPT_RAG_LLM_API_KEY environment variable")
        
        logger.info(f"Using API key from environment (length: {len(self.api_key)} chars)")
        
        # Initialize API client based on provider
        if self.api_provider == "openai":
            self.api_client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base_url  # None is fine for OpenAI's default endpoint
            )
            logger.info(f"Initialized OpenAI client with model: {self.api_model}")
        else:
            # Future: Add support for other providers (Anthropic, etc.)
            raise ValueError(f"Unsupported API provider: {self.api_provider}")
        
        # API mode doesn't need tokenizer, model, or device
        self.tokenizer = None
        self.model = None
        self.device = None
        logger.info("API mode initialization complete")

    def _init_local_mode(self, device: Optional[str] = None):
        """Initialize local model mode (original implementation)."""
        logger.info("Initializing local model mode")
        
        if not self.model_path or not self.model_path.exists():
            raise ValueError(f"Model path {self.model_path} does not exist")
            
        # Detect hardware if not specified
        self.device = device or self._detect_hardware()
        logger.info(f"Using device: {self.device}")

        # Initialize local model components 
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        
        # API mode variables (set to None for local mode)
        self.api_client = None
        self.api_provider = None
        self.api_model = None
        logger.info("Local mode initialization complete")

    def _detect_hardware(self) -> str:
        """Detect the hardware device to use for inference."""
        with tracer.start_as_current_span("detect_hardware"):
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
    
    def _load_tokenizer(self):
        """Load the tokenizer."""
        with tracer.start_as_current_span("load_tokenizer") as span:
            span.set_attribute("model_path", str(self.model_path))
            logger.info(f"Loading tokenizer from {self.model_path}")

            if not self.model_path.exists():
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            return AutoTokenizer.from_pretrained(self.model_path)
    
    def _load_model(self):
        """Load and configure the model."""
        with tracer.start_as_current_span("load_model") as span:
            span.set_attribute("model_path", str(self.model_path))
            span.set_attribute("device", self.device)
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
                    # CPU configuration - full precision
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        device_map="auto",
                    )
                
                return model
            
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                span.record_exception(e)
                logger.error(f"Failed to initialize model: {e}")
                raise 

    
    def _load_vector_store(self):
        """Load the FAISS vector store if it exists, or create an empty one."""
        with tracer.start_as_current_span("load_vector_store") as span:
            span.set_attribute("vector_store_path", str(self.vector_store_path))
            try: 
                return load_vector_store(
                    vector_store_path = self.vector_store_path,
                    device = self.device, 
                    force_reload = False
                )
            
            except (FileNotFoundError, RuntimeError) as e:
                span.set_attribute("creating_new_vector_store", True)
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
        - NEVER fabricate information or provide speculative advice on visa matters
        - When answering, always check if your response contradicts any information in the context - if it does, defer to the context
        - If the context indicates no documentation is available, be honest about this limitation
        - NEVER pretend to have information that isn't in the context
        - Always indicate the source of information in your responses
        - Avoid legal advice; clarify when questions require consultation with immigration attorneys
        - DO NOT UNDER ANY CIRCUMSTANCES prefix your response with "A:", "Assistant:", or any similar prefix - just provide the answer directly
        - If the context says there is no information on the topic, clearly state this and don't try to answer the question
        - DO NOT repeat the question in your answer or include "and how does it work" or similar phrases
        - DO NOT include any prefixes like "A:" or "Assistant:" anywhere in your response, not just at the beginning
        - Your response should NOT contain multiple answers or repetitions - just provide a single coherent answer

        ## CONTEXT
        {context}

        ## USER INFORMATION
        Student status: International student in the United States
        Primary concern: Visa and immigration matters

        ## RESPONSE FORMAT
        - Begin with a direct and factually accurate answer to the question based ONLY on the context provided
        - DO NOT start with "A:" or "Assistant:" or any similar prefix
        - DO NOT repeat the question in your answer
        - If the context doesn't contain relevant information, clearly state this limitation
        - Provide specific, relevant details from official sources in the context ONLY IF AVAILABLE
        - Include citation to specific documents/policies when available
        - Highlight important deadlines or requirements mentioned in the context
        - If applicable, mention next steps the student should take according to the context
        - End with a disclaimer that this information is not legal advice
        - IMPORTANT: Your response should be a single, coherent answer - DO NOT provide multiple versions of the same answer

        ## QUESTION
        {question}
        """)

    async def add_documents(
        self, 
        file_path: Union[str, List[str]], 
        document_type: Optional[str] = None, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200 
    ) -> Dict[str, Any]:
        """Add documents to the vector store.
        
        Args:
            file_path: Path to the file or list of file paths
            document_type: Optional type of document for metadata
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Dictionary with information about the added documents
        """
        with tracer.start_as_current_span("add_documents") as span:
            try:
                # Convert to list if a single path else use the provided list
                source_path = file_path if isinstance(file_path, list) else [file_path]
                if isinstance(file_path, str):
                    file_path = [file_path]
                
                span.set_attribute("num_documents", len(file_path))
                span.set_attribute("document_type", document_type or "unknown")
                
                start_time = time.time()
                
                # Process documents
                processed_info = await process_documents(
                    source_path=source_path, 
                    vector_store_path=self.vector_store_path,
                    device=self.device,
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap,
                )
                
                # If no chunks were generated, return error
                if not processed_info["chunks"]:
                    span.set_status(trace.Status(trace.StatusCode.ERROR))
                    span.set_attribute("error", "No chunks generated")
                    return {"status": "error", "message": "No chunks were generated from the documents"}
                
                # Get chunks and metadata
                chunks = processed_info["chunks"]
                metadata_list = None

                import uuid

                document_ids = [str(uuid.uuid4()) for _ in file_path]
                
                # Add to vector store
                self.vector_store = build_vector_store(
                    chunks=chunks,
                    vector_store_path=self.vector_store_path,
                    device=self.device,
                    # existing_store=self.vector_store,
                    # metadata_list=metadata_list
                )
                
                # Update vector count metric
                if hasattr(self.vector_store, "index"):
                    VECTOR_COUNT.set(self.vector_store.index.ntotal)
                
                processing_time = time.time() - start_time
                
                # Create response
                return {
                    "status": "success",
                    "document_ids": document_ids,
                    "document_count": len(file_path),
                    "chunk_count": len(chunks),
                    "processing_time": processing_time
                }
                
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                span.record_exception(e)
                logger.error(f"Failed to add documents: {e}")
                return {"status": "error", "message": str(e)}
    
    def remove_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """Remove documents from the vector store.
        
        Args:
            document_ids: List of document IDs to remove
            
        Returns:
            Dictionary with status and information
        """
        with tracer.start_as_current_span("remove_documents") as span:
            span.set_attribute("document_ids", str(document_ids))
            # Not implemented yet
            return {"status": "error", "message": "Document removal not implemented yet"}
    
    def list_documents(self) -> Dict[str, Any]:
        """List documents in the vector store.
        
        Returns:
            Dictionary with document information
        """
        with tracer.start_as_current_span("list_documents"):
            # Get documents from vector store
            documents = []
            document_count = 0
            
            if hasattr(self.vector_store, "docstore"):
                document_count = len(self.vector_store.docstore._dict)
                
                # Extract unique documents (removing chunks)
                unique_docs = {}
                for doc_id, doc in self.vector_store.docstore._dict.items():
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        doc_source = doc.metadata["source"]
                        if doc_source not in unique_docs:
                            unique_docs[doc_source] = {
                                "source": doc_source,
                                "document_type": doc.metadata.get("document_type", "unknown"),
                                "chunk_count": 0
                            }
                        unique_docs[doc_source]["chunk_count"] += 1
                
                documents = list(unique_docs.values())

            return {
                "status": "success",
                "document_count": len(documents),
                "total_chunks": document_count,
                "documents": documents
            }
    
    def answer_question(self, query: str, stream: bool = False) -> Dict[str, Any]:
        """Answer a question using the RAG pipeline.
        
        UPDATED: Now supports both local model and API-based inference.
        
        Args:
            query: User's question
            stream: Whether to stream the response (not used in API mode for this method)
            
        Returns:
            Dictionary with the answer and processing time
        """
        with tracer.start_as_current_span("answer_question") as span:
            span.set_attribute("query", query)
            span.set_attribute("stream", stream)
            span.set_attribute("use_api_llm", self.use_api_llm)
            
            start_time = time.time()
            status = "success"
            error_type = None
            
            try:
                # Record query count (starting)
                QUERY_COUNT.labels(status="started", query_type="standard").inc()
                
                # Retrieve relevant context (SAME FOR BOTH MODES)
                with tracer.start_as_current_span("retrieve_context"):
                    retrieval_start = time.time()
                    
                    if hasattr(self.vector_store, "similarity_search"):
                        results = self.vector_store.similarity_search(query, k=4)
                        context = "\n\n".join([doc.page_content for doc in results])
                    else:
                        # No documents in the vector store
                        context = "No relevant documentation is available."
                        
                    # Record vector retrieval time
                    retrieval_time = time.time() - retrieval_start
                    VECTOR_RETRIEVAL_LATENCY.observe(retrieval_time)
                    span.set_attribute("retrieval_time", retrieval_time)
                
                # Prepare prompt with context and question (SAME FOR BOTH MODES)
                with tracer.start_as_current_span("prepare_prompt"):
                    prompt = self.visa_prompt.format(
                        context=context,
                        question=query
                    )
                
                # Generate LLM response (DIFFERENT FOR EACH MODE)
                if self.use_api_llm:
                    # ===== NEW: API-BASED RESPONSE GENERATION =====
                    with tracer.start_as_current_span("generate_api_answer") as gen_span:
                        gen_span.set_attribute("api_provider", self.api_provider)
                        gen_span.set_attribute("api_model", self.api_model)
                        
                        # Prepare messages for API
                        messages = [{"role": "user", "content": prompt}]
                        
                        # Call API
                        if self.api_provider == "openai":
                            response = self.api_client.chat.completions.create(
                                model=self.api_model,
                                messages=messages,
                                max_tokens=1024,
                                temperature=0.7,
                                top_p=0.9
                            )
                            response_text = response.choices[0].message.content
                        else:
                            raise ValueError(f"Unsupported API provider: {self.api_provider}")
                
                else:
                    # ===== ORIGINAL: LOCAL MODEL RESPONSE GENERATION (PRESERVED) =====
                    with tracer.start_as_current_span("generate_answer") as gen_span:
                        gen_span.set_attribute("model_name", str(self.model_path).split('/')[-1])
                        
                        messages = [{"role": "user", "content": prompt}]
                        
                        if stream:
                            # Streaming response (LOCAL MODEL ONLY)
                            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                            gen_thread = threading.Thread(
                                target=self._generate_response,
                                args=(messages, streamer)
                            )
                            gen_thread.start()
                            
                            response_text = ""
                            for response_chunk in streamer:
                                response_text += response_chunk
                                
                        else:
                            # Standard response (LOCAL MODEL ONLY)
                            inputs = self.tokenizer.apply_chat_template(
                                messages,
                                add_generation_prompt=True,
                                return_tensors="pt"
                            ).to(self.device)
                            
                            outputs = self.model.generate(
                                inputs,
                                max_new_tokens=1024,
                                temperature=0.7,
                                top_p=0.9,
                                repetition_penalty=1.1,
                                do_sample=True
                            )
                            
                            response_text = self.tokenizer.decode(
                                outputs[0][inputs.shape[1]:],
                                skip_special_tokens=True
                            )
                
                # Calculate total processing time (SAME FOR BOTH MODES)
                processing_time = time.time() - start_time
                
                # Record metrics (SAME FOR BOTH MODES)
                QUERY_LATENCY.observe(processing_time)
                QUERY_COUNT.labels(status="success", query_type="standard").inc()
                
                # Add processing time to the span
                span.set_attribute("processing_time", processing_time)
                
                # Return result (SAME FOR BOTH MODES)
                return {
                    "answer": response_text,
                    "processing_time": processing_time
                }
                
            except Exception as e:
                # Record failure (SAME FOR BOTH MODES)
                status = "error"
                error_type = type(e).__name__
                
                # Update span with error info
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                span.record_exception(e)
                
                # Log error
                logger.error(f"Error processing query: {e}")
                
                # Record metrics
                processing_time = time.time() - start_time
                QUERY_LATENCY.observe(processing_time)
                QUERY_COUNT.labels(status="error", query_type="standard").inc()
                QUERY_ERRORS.labels(error_type=error_type).inc()
                
                # Return error response
                return {
                    "answer": f"I encountered an error while processing your query: {str(e)}",
                    "error": str(e),
                    "processing_time": processing_time
                }

    def _generate_response(self, messages, streamer):
        """Helper method to generate response with a streamer."""
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        
        self.model.generate(
            inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            streamer=streamer
        )

    async def astream_response(self, query: str, cancel_event: Optional[threading.Event] = None) -> AsyncIterator[str]:
        """Stream response to a question asynchronously.
        
        UPDATED: Now supports both local model and API-based streaming.
        
        Args:
            query: User's question
            cancel_event: Optional event that can be set to cancel generation
            
        Yields:
            Response tokens as they are generated
        """
        with tracer.start_as_current_span("astream_response") as span:
            span.set_attribute("query", query)
            span.set_attribute("use_api_llm", self.use_api_llm)
            
            start_time = time.time()
            status = "success"
            error_type = None
            
            try:
                # Record query count (starting) - SAME FOR BOTH MODES
                QUERY_COUNT.labels(status="started", query_type="streaming").inc()
                
                # Retrieve relevant context - SAME FOR BOTH MODES
                with tracer.start_as_current_span("retrieve_context"):
                    # Check early for cancellation
                    if cancel_event and cancel_event.is_set():
                        logger.info("Generation cancelled before context retrieval")
                        status = "cancelled"
                        QUERY_COUNT.labels(status="cancelled", query_type="streaming").inc()
                        return
                        
                    retrieval_start = time.time()
                    
                    if hasattr(self.vector_store, "similarity_search"):
                        results = self.vector_store.similarity_search(query, k=4)
                        if results and len(results) > 0:
                            context = "\n\n".join([doc.page_content for doc in results])
                            logger.info(f"Retrieved {len(results)} relevant documents")
                        else:
                            # No relevant documents found in vector store
                            context = "No information about this topic was found in the available documents. Please provide a response stating clearly that you don't have information on this specific topic and avoid making assumptions or providing information not supported by documentation."
                            logger.warning("No relevant documents found for query")
                    else:
                        # No documents in the vector store
                        context = "The knowledge base is currently empty. No documents have been loaded. Please provide a response stating clearly that you don't have any documentation available and cannot provide specific information."
                        logger.warning("Vector store is empty or unavailable")
                        
                    # Record vector retrieval time
                    retrieval_time = time.time() - retrieval_start
                    VECTOR_RETRIEVAL_LATENCY.observe(retrieval_time)
                    span.set_attribute("retrieval_time", retrieval_time)
                
                # Check again for cancellation - SAME FOR BOTH MODES
                if cancel_event and cancel_event.is_set():
                    logger.info("Generation cancelled after context retrieval")
                    status = "cancelled"
                    QUERY_COUNT.labels(status="cancelled", query_type="streaming").inc()
                    return
                
                # Prepare prompt with context and question - SAME FOR BOTH MODES
                with tracer.start_as_current_span("prepare_prompt"):
                    prompt = self.visa_prompt.format(
                        context=context,
                        question=query
                    )
                    logger.info("Prepared prompt for streaming generation")
                
                # Generate streaming response - DIFFERENT FOR EACH MODE
                if self.use_api_llm:
                    # ===== NEW: API-BASED STREAMING =====
                    with tracer.start_as_current_span("generate_api_streaming_answer"):
                        logger.info("Starting API-based streaming generation")
                        
                        # Prepare messages for API
                        messages = [{"role": "user", "content": prompt}]
                        token_count = 0
                        
                        if self.api_provider == "openai":
                            # Use OpenAI streaming
                            stream = self.api_client.chat.completions.create(
                                model=self.api_model,
                                messages=messages,
                                max_tokens=1024,
                                temperature=0.7,
                                top_p=0.9,
                                stream=True
                            )
                            
                            for chunk in stream:
                                # Check for cancellation
                                if cancel_event and cancel_event.is_set():
                                    logger.info(f"API streaming cancelled after {token_count} tokens")
                                    status = "cancelled"
                                    break
                                
                                # Extract content from chunk
                                if chunk.choices[0].delta.content:
                                    content = chunk.choices[0].delta.content
                                    token_count += 1
                                    if token_count % 10 == 0:
                                        logger.info(f"Streamed {token_count} API tokens so far")
                                    yield content
                        else:
                            raise ValueError(f"Streaming not supported for API provider: {self.api_provider}")
                        
                        logger.info(f"API streaming complete. Total tokens: {token_count}")
                
                else:
                    # ===== ORIGINAL: LOCAL MODEL STREAMING (PRESERVED) =====
                    with tracer.start_as_current_span("generate_streaming_answer"):
                        # Create direct streamer for token streaming
                        direct_streamer = DirectStreamer(self.model, self.tokenizer, self.device)
                        
                    # Start streaming tokens
                    logger.info("Starting to stream tokens with direct streamer")
                    token_count = 0
                    
                    # Generate and stream tokens with direct streamer
                    async for token in direct_streamer.generate_and_stream(prompt, max_tokens=1024, cancel_event=cancel_event):
                        # Check for cancellation (additional safety check)
                        if cancel_event and cancel_event.is_set():
                            logger.info(f"Streaming cancelled after {token_count} tokens")
                            status = "cancelled"
                            break
                            
                        token_count += 1
                        if token_count % 10 == 0:
                            logger.info(f"Streamed {token_count} tokens so far")
                        yield token
                    
                        logger.info(f"Local streaming complete. Total tokens: {token_count}")
                
                # Record metrics - SAME FOR BOTH MODES
                processing_time = time.time() - start_time
                QUERY_LATENCY.observe(processing_time)
                QUERY_COUNT.labels(status=status, query_type="streaming").inc()
                
                # Add processing time to the span
                span.set_attribute("processing_time", processing_time)
                
            except Exception as e:
                # Record failure - SAME FOR BOTH MODES
                status = "error"
                error_type = type(e).__name__
                
                # Update span with error info
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                span.record_exception(e)
                
                # Log error
                logger.error(f"Error processing streaming query: {e}", exc_info=True)
                
                # Record metrics
                processing_time = time.time() - start_time
                QUERY_LATENCY.observe(processing_time)
                QUERY_COUNT.labels(status="error", query_type="streaming").inc()
                QUERY_ERRORS.labels(error_type=error_type).inc()
                
                # Yield error message
                yield f"Error: {str(e)}"


