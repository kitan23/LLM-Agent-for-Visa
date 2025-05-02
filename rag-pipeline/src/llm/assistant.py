"""
OPT-RAG Assistant Implementation

This module contains the core implementation of the OPT-RAG International Student 
Visa Assistant, handling model loading, vector store management, and query processing.
"""


import logging
import time
import asyncio
from typing import Dict, Any, Optional, AsyncIterator, Iterator, Union, List
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from llm.callbacks import StreamingCallbackHandler, AsyncStreamingCallbackHandler
from utils.metrics import QUERY_LATENCY, QUERY_COUNT, MODEL_LOAD_TIME, VECTOR_COUNT, VECTOR_RETRIEVAL_LATENCY, QUERY_ERRORS
from retriever.vector_store import load_vector_store, build_vector_store
from document_processor.pipeline import process_documents

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
        if hasattr(self.vector_store, "_index"):
            VECTOR_COUNT.set(self.vector_store._index.ntotal)

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
        - Provide accurate information based ONLY on official USCIS documents, university policies, and government regulations in the context provided
        - Focus specifically on visa-related issues: OPT applications, CPT authorization, study/work permits, and visa status questions
        - NEVER fabricate information or provide speculative advice on visa matters
        - If information is not available in the context, clearly state this limitation
        - Always indicate the source of information in your responses
        - Avoid legal advice; clarify when questions require consultation with immigration attorneys

        ## CONTEXT
        {context}

        ## USER INFORMATION
        Student status: International student in the United States
        Primary concern: Visa and immigration matters

        ## RESPONSE FORMAT
        - Begin with a direct answer to the question
        - Provide specific, relevant details from official sources
        - Include citation to specific documents/policies when available
        - Highlight important deadlines or requirements
        - If applicable, mention next steps the student should take
        - End with a disclaimer that this information is not legal advice

        ## QUESTION
        {question}
        """)

    def _create_llm_with_callbacks(self, callbacks=None):
        """Create LLM with specified callbacks for streaming"""      

        pipe = pipeline(
            "text-generation", 
            model = self.model, 
            tokenizer = self.tokenizer, 
            max_new_token = 512, 
            repetition_penalty = 1.1, 
            temperature = 0.1, # lower temperature for more deterministic responses
            do_sample = True
        )
    
        return HuggingFacePipeline(
            pipeline = pipe, 
            callbacks = callbacks, 
        )

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
            if hasattr(self.vector_store, "_index"):
                before_count = self.vector_store._index.ntotal
            

            # call async process_documents 
            result = await process_documents(
                source_path = file_path, 
                vector_store_path = self.vector_store_path, 
                device = self.device, 
                chunk_size = chunk_size, 
                chunk_overlap = chunk_overlap, 
            )

            
            # Reload vector store after adding documents
            self.vector_store = load_vector_store(
                vector_store_path=self.vector_store_path, 
                device=self.device, 
                force_reload=True
            )

            after_count = 0 
            # Update vector count metric 
            if hasattr(self.vector_store, "_index"):
                after_count = self.vector_store._index.ntotal
                VECTOR_COUNT.set(after_count)
            
            processing_time = time.time() - start_time 
            logger.info(f"Documents added successfully in {processing_time:.2f} seconds")

            return {
                "status": "success", 
                "document_processed": len(result.get("documents", [])),
                "chunks_created": len(result.get("chunks", [])),
                "processing_time": processing_time,
                "vectors_added": after_count - before_count
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
        if hasattr(self.vector_store, '_index'):
            return {
                "status": "success",
                "vector_count": self.vector_store._index.ntotal,
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

            # Measure retrieval time 
            VECTOR_RETRIEVAL_LATENCY.observe(time.time() - retrieval_start_time)

            if stream: 
                # Setup streaming
                streaming_handler = StreamingCallbackHandler()
                from langchain.callbacks.manager import CallbackManager
                callbacks = CallbackManager([streaming_handler])
                
                # Create LLM with streaming
                llm = self._create_llm_with_callbacks(callbacks=[callbacks])
                
                # Create chain
                combine_docs_chain = create_stuff_documents_chain(
                    llm=llm,
                    prompt=self.visa_prompt
                )
                
                chain = create_retrieval_chain(
                    retriever=retriever,
                    combine_docs_chain=combine_docs_chain
                )
                
                # Execute the chain
                response = chain.invoke({"question": query})
                
                # Add streaming info to response
                response["streaming_handler"] = streaming_handler
            

            else:
                # Create LLM without streaming
                llm = self._create_llm_with_callbacks()
                
                # Create chain
                combine_docs_chain = create_stuff_documents_chain(
                    llm=llm,
                    prompt=self.visa_prompt
                )
                
                chain = create_retrieval_chain(
                    retriever=retriever,
                    combine_docs_chain=combine_docs_chain
                )
                
                # Execute the chain
                response = chain.invoke({"question": query})


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
                "answer": "I'm encountered an error while processing your question. Please try again.", 
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
        
        # Create async queue for streaming
        queue = asyncio.Queue()
        
        # Setup async streaming handler
        streaming_handler = AsyncStreamingCallbackHandler(queue)
        from langchain.callbacks.manager import AsyncCallbackManager
        callbacks = AsyncCallbackManager([streaming_handler])

        # Create chain with async streaming 

        try:
            start_time = time.time()
            
            # Process documents retrieval first (this isn't streamed)
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            docs = retriever.get_relevant_documents(query)
            
            # Format documents for context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create LLM with streaming callbacks
            llm = self._create_llm_with_callbacks(callbacks=[callbacks])
            
            # First token to show retrieval is complete
            yield "Searching visa regulations...\n\n"
            
            # Start generation in background
            task = asyncio.create_task(
                llm.ainvoke(
                    self.visa_prompt.format(
                        question=query,
                        context=context
                    )
                )
            )

            # Stream tokens as they arrive
            while True:
                # Wait for new tokens with timeout
                try:
                    token = await asyncio.wait_for(queue.get(), timeout=0.1)
                    yield token
                    queue.task_done()
                except asyncio.TimeoutError:
                    # Check if generation is complete
                    if task.done():
                        break
            
            # Ensure task completes
            await task
            
            # Log performance metrics
            elapsed_time = time.time() - start_time
            logger.info(f"Streaming query processed in {elapsed_time:.2f} seconds")
            QUERY_LATENCY.observe(elapsed_time)

            QUERY_COUNT.labels(status="completed", query_type="streaming").inc()

        except Exception as e:

            # Track error properly 
            QUERY_ERRORS.labels(error_type = type(e).__name__).inc()
            QUERY_COUNT.labels(status="error", query_type="streaming").inc()

            logger.error(f"Error streaming response: {e}")
            yield f"\n I'm sorry, an error occurred: {str(e)}"
        


