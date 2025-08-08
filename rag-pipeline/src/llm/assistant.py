"""
OPT-RAG Assistant Implementation

This module contains the core implementation of the OPT-RAG International Student
Visa Assistant. It leverages RunPod's serverless infrastructure via an
OpenAI-compatible API for efficient and scalable model inference.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, AsyncIterator, List, Union
from pathlib import Path
from threading import Event

from opentelemetry import trace
from openai import AsyncOpenAI
from langchain_core.prompts import PromptTemplate

from src.retriever.vector_store import load_vector_store, build_vector_store
from src.document_processor.pipeline import process_documents
from langfuse.model import Trace, Generation

logger = logging.getLogger("opt_rag.assistant")
# tracer = get_tracer("opt_rag.assistant")


class OPTRagAssistant:
    """
    OPT-RAG International Student Visa Assistant using RunPod's OpenAI-compatible API.
    """

    def __init__(
        self,
        vector_store_path: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the OPT-RAG Assistant."""
        self.settings = get_settings()
        self.vector_store_path = Path(vector_store_path)
        self.model_params = model_kwargs or {}
        
        # Load vector store on initialization
        self.vector_store = self._load_vector_store()

        if not self.settings.runpod_api_key or not self.settings.runpod_endpoint_id:
            raise ValueError("RunPod API key and endpoint ID must be configured.")

        self.client = AsyncOpenAI(
            api_key=self.settings.runpod_api_key,
            base_url=f"https://api.runpod.ai/v2/{self.settings.runpod_endpoint_id}/openai/v1",
        )

        logger.info(f"Initialized RunPod assistant for endpoint: {self.settings.runpod_endpoint_id}")
        logger.info(f"Using model: {self.settings.runpod_model_name}")

    def _load_vector_store(self):
        """Load the FAISS vector store if it exists, or create an empty one."""
        # with tracer.start_as_current_span("load_vector_store") as span:
        #     span.set_attribute("vector_store_path", str(self.vector_store_path))
        try:
            return load_vector_store(
                vector_store_path=self.vector_store_path,
                force_reload=False
            )
        except (FileNotFoundError, RuntimeError) as e:
            # span.set_attribute("creating_new_vector_store", True)
            logger.warning(f"Vector store not found at {self.vector_store_path}, creating new one")
            
            # Create directory if it doesn't exist
            self.vector_store_path.mkdir(parents=True, exist_ok=True)

            # Return empty vector store 
            return build_vector_store(
                chunks=["This is a dummy chunk."],
                vector_store_path=self.vector_store_path,
            )

    def _create_messages(self, context: str, question: str) -> List[Dict[str, str]]:
        """Create the messages payload for the OpenAI API."""
        system_prompt = """
       You are an expert assistant for international student visa questions in the United States.
        Your role is to answer user questions based *only* on the provided context.
        ## ROLE AND GUIDELINES
        - ONLY provide information that is explicitly supported by the context below.
        - DO NOT make any claims or assertions that aren't directly supported by the provided context.
        - Focus specifically on visa-related issues: OPT applications, CPT authorization, study/work permits, and visa status questions.
        - If information is not available in the context, clearly state "Based on the provided context, I don't have specific information about that."
        - NEVER fabricate information or provide speculative advice on visa matters.
        - When answering, always check if your response contradicts any information in the context - if it does, defer to the context.
        - If the context indicates no documentation is available, be honest about this limitation.
        - NEVER pretend to have information that isn't in the context.
        - Always indicate the source of information in your responses.
        - Avoid legal advice; clarify when questions require consultation with immigration attorneys.
        - Do not suggest contacting a specific institution or program office (e.g., "contact the University of Oregon program office") unless the user has specified their affiliation. Instead, provide general advice like "you should contact your university's international student office."
        - DO NOT prefix your response with "A:", "Assistant:", or any similar prefix - just provide the answer directly.
        - If the context says there is no information on the topic, clearly state this and don't try to answer the question.
        - DO NOT repeat the question in your answer or include "and how does it work" or similar phrases.
        - DO NOT include any prefixes like "A:" or "Assistant:" anywhere in your response, not just at the beginning.
        - Your response should NOT contain multiple answers or repetitions - just provide a single coherent answer.
        """

        user_prompt = f"""
        ## CONTEXT
        {context}

        ## USER INFORMATION
        Primary concern: Visa and immigration matters

        ## QUESTION
        {question}
        """

        return [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]

    def _clean_response(self, response: str) -> str:
        """Clean up the response from the LLM."""
        return response.replace("</s>", "").strip()

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
        # with tracer.start_as_current_span("add_documents") as span:
        try:
            # Convert to list if a single path
            source_path = [file_path] if isinstance(file_path, str) else file_path
            
            # span.set_attribute("num_documents", len(source_path))
            # span.set_attribute("document_type", document_type or "unknown")
            
            start_time = time.time()
            
            # Process documents
            processed_info = await process_documents(
                source_path=source_path, 
                vector_store_path=self.vector_store_path,
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
            )
            
            # If no chunks were generated, return error
            if not processed_info["chunks"]:
                # span.set_status(trace.Status(trace.StatusCode.ERROR))
                # span.set_attribute("error", "No chunks generated")
                return {"status": "error", "message": "No chunks were generated from the documents"}
            
            # Add to vector store
            self.vector_store = build_vector_store(
                chunks=processed_info["chunks"],
                vector_store_path=self.vector_store_path,
            )
            
            # Update vector count metric
            # if hasattr(self.vector_store, "index"):
            #     VECTOR_COUNT.set(self.vector_store.index.ntotal)
            
            processing_time = time.time() - start_time
            
            # Create response
            return {
                "status": "success",
                "document_count": len(source_path),
                "chunk_count": len(processed_info["chunks"]),
                "processing_time": processing_time
            }
            
        except Exception as e:
            # span.set_status(trace.Status(trace.StatusCode.ERROR))
            # span.record_exception(e)
            logger.error(f"Failed to add documents: {e}")
            return {"status": "error", "message": str(e)}
    
    def remove_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """Remove documents from the vector store.
        
        Args:
            document_ids: List of document IDs to remove
            
        Returns:
            Dictionary with status and information
        """
        # with tracer.start_as_current_span("remove_documents") as span:
        #     span.set_attribute("document_ids", str(document_ids))
        # Not implemented yet
        return {"status": "error", "message": "Document removal not implemented yet"}
    
    def list_documents(self) -> Dict[str, Any]:
        """List documents in the vector store.
        
        Returns:
            Dictionary with document information
        """
        # with tracer.start_as_current_span("list_documents"):
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
    
    async def answer_question(self, query: str, trace: Trace, generation: Generation) -> Dict[str, Any]:
        """Answer a question using the RAG pipeline."""
        # with tracer.start_as_current_span("answer_question") as span:
        try:
            # QUERY_COUNT.labels(status="started", query_type="standard").inc()
            start_time = time.time()
            
            # Get relevant documents from vector store
            retrieval_start = time.time()
            relevant_docs = self.vector_store.similarity_search(query, k=3)
            retrieval_time = time.time() - retrieval_start
            # VECTOR_RETRIEVAL_LATENCY.observe(retrieval_time)
            
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            generation.end(
                output=context,
                metadata={
                    "retrieval_time": retrieval_time,
                    "document_count": len(relevant_docs)
                }
            )

            print("Context", context)
            
            messages = self._create_messages(context=context, question=query)

            response = await self.client.chat.completions.create(
                model=self.settings.runpod_model_name,
                messages=messages,
                stream=False,
                **self.model_params,
            )
            
            response_text = response.choices[0].message.content or ""
            response_text = self._clean_response(response_text)

            response_time = time.time() - start_time
            # QUERY_LATENCY.observe(response_time)

            generation.end(output=response_text)
            trace.update(output=response_text)
            
            return {
                "answer": response_text,
                "sources": [{"content": doc.page_content, "metadata": doc.metadata} for doc in relevant_docs],
                "metadata": {
                    "response_time": response_time,
                    "model": self.settings.runpod_model_name
                }
            }
            
        except Exception as e:
            # span.set_status(trace.Status(trace.StatusCode.ERROR))
            # span.record_exception(e)
            # QUERY_ERRORS.labels(error_type=type(e).__name__).inc()
            generation.end(level='ERROR', status_message=str(e))
            trace.update(level='ERROR', status_message=str(e))
            raise

    async def astream_response(self, query: str, cancel_event: Optional[Event] = None, trace: Optional[Trace] = None, generation: Optional[Generation] = None) -> AsyncIterator[str]:
        """Stream response to a question asynchronously."""
        # with tracer.start_as_current_span("astream_response") as span:
        #     span.set_attribute("query", query)
            
        start_time = time.time()
        # status = "success"
        # error_type = None
        
        try:
            # Record query count
            # QUERY_COUNT.labels(status="started", query_type="streaming").inc()
            
            # Get relevant documents
            # with tracer.start_as_current_span("retrieve_context"):
            if cancel_event and cancel_event.is_set():
                logger.info("Generation cancelled before context retrieval")
                # status = "cancelled"
                # QUERY_COUNT.labels(status="cancelled", query_type="streaming").inc()
                return
                
            retrieval_start = time.time()
            relevant_docs = self.vector_store.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            print("Context", context)

            # VECTOR_RETRIEVAL_LATENCY.observe(time.time() - retrieval_start)

            if generation:
                generation.end(
                    output=context,
                    metadata={
                        "retrieval_time": time.time() - retrieval_start,
                        "document_count": len(relevant_docs)
                    }
                )
            
            messages = self._create_messages(context=context, question=query)
            
            stream = await self.client.chat.completions.create(
                model=self.settings.runpod_model_name,
                messages=messages,
                stream=True,
                **self.model_params,
            )
            
            full_response = ""
            async for chunk in stream:
                if cancel_event and cancel_event.is_set():
                    logger.info("Generation cancelled")
                    # status = "cancelled"
                    break
                
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield content
            
            # processing_time = time.time() - start_time
            # QUERY_LATENCY.observe(processing_time)
            # QUERY_COUNT.labels(status=status, query_type="streaming").inc()
            if trace:
                trace.update(output=full_response)
            
        except Exception as e:
            # status = "error"
            # error_type = type(e).__name__
            
            # span.set_status(trace.Status(trace.StatusCode.ERROR))
            # span.record_exception(e)
            if generation:
                generation.end(level='ERROR', status_message=str(e))
            if trace:
                trace.update(level='ERROR', status_message=str(e))
            
            logger.error(f"Error processing streaming query: {e}", exc_info=True)
            
            # QUERY_LATENCY.observe(time.time() - start_time)
            # QUERY_COUNT.labels(status="error", query_type="streaming").inc()
            # QUERY_ERRORS.labels(error_type=error_type).inc()
            
            yield f"Error: {str(e)}"


