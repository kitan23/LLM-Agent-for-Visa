"""
OPT-RAG Assistant Implementation

This module contains the core implementation of the OPT-RAG International Student
Visa Assistant. It leverages RunPod's serverless infrastructure via an
OpenAI-compatible API for efficient and scalable model inference.
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, AsyncIterator, List, Union, Protocol
from pathlib import Path
from threading import Event

from openai import AsyncOpenAI

from src.retriever.vector_store import (
    load_vector_store,
    build_vector_store,
    append_to_vector_store,
    DUMMY_CHUNK,
)
from src.document_processor.pipeline import process_documents
from src.utils.config import get_settings

logger = logging.getLogger("opt_rag.assistant")

DUMMY_TEXTS = {DUMMY_CHUNK, "dummy text", "This is a dummy chunk."}


class TraceLike(Protocol):
    """Structural type for Langfuse trace objects."""

    def update(self, **kwargs: Any) -> Any: ...


class GenerationLike(Protocol):
    """Structural type for Langfuse generation objects."""

    def end(self, **kwargs: Any) -> Any: ...


class OPTRagAssistant:
    """
    OPT-RAG International Student Visa Assistant.

    Uses RunPod's OpenAI-compatible API as the primary LLM provider and falls
    back to the OpenAI API when RunPod is not configured or fails.
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

        self.client: Optional[AsyncOpenAI] = None
        self.fallback_client: Optional[AsyncOpenAI] = None

        if self.settings.runpod_api_key and self.settings.runpod_endpoint_id:
            self.client = AsyncOpenAI(
                api_key=self.settings.runpod_api_key,
                base_url=f"https://api.runpod.ai/v2/{self.settings.runpod_endpoint_id}/openai/v1",
            )
            logger.info(f"Initialized RunPod assistant for endpoint: {self.settings.runpod_endpoint_id}")
            logger.info(f"Using model: {self.settings.runpod_model_name}")
        else:
            logger.warning("RunPod API key / endpoint ID not configured; primary LLM unavailable.")

        if self.settings.openai_api_key:
            self.fallback_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
            logger.info(f"OpenAI fallback configured with model: {self.settings.openai_model_name}")

        if self.client is None and self.fallback_client is None:
            logger.error(
                "No LLM provider configured. Set RUNPOD_API_KEY/RUNPOD_ENDPOINT_ID "
                "or OPENAI_API_KEY to enable answering."
            )

    @property
    def is_configured(self) -> bool:
        """True when at least one LLM provider is available."""
        return self.client is not None or self.fallback_client is not None

    def _load_vector_store(self):
        """Load the FAISS vector store if it exists, or create an empty one."""
        try:
            return load_vector_store(
                vector_store_path=self.vector_store_path,
                force_reload=False
            )
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning(f"Vector store not found at {self.vector_store_path}, creating new one")

            # Create directory if it doesn't exist
            self.vector_store_path.mkdir(parents=True, exist_ok=True)

            # Return empty vector store
            return build_vector_store(
                chunks=[DUMMY_CHUNK],
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

        New chunks are merged into the existing knowledge base so previously
        uploaded documents are preserved.

        Args:
            file_path: Path to the file or list of file paths
            document_type: Optional type of document for metadata
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks

        Returns:
            Dictionary with information about the added documents
        """
        try:
            # Convert to list if a single path
            source_path = [file_path] if isinstance(file_path, str) else file_path

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
                return {"status": "error", "message": "No chunks were generated from the documents"}

            # Tag chunks with the provided document type for later filtering
            if document_type:
                for chunk in processed_info["chunks"]:
                    chunk.metadata["document_type"] = document_type

            # Merge the new chunks into the existing store instead of replacing it
            self.vector_store = append_to_vector_store(
                existing_store=self.vector_store,
                new_chunks=processed_info["chunks"],
                vector_store_path=self.vector_store_path,
            )

            processing_time = time.time() - start_time

            return {
                "status": "success",
                "document_count": len(source_path),
                "chunk_count": len(processed_info["chunks"]),
                "total_vectors": int(self.vector_store.index.ntotal),
                "processing_time": processing_time
            }

        except Exception as e:
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
        # Get documents from vector store
        documents = []
        document_count = 0

        if hasattr(self.vector_store, "docstore"):
            document_count = len(self.vector_store.docstore._dict)

            # Extract unique documents (removing chunks)
            unique_docs: Dict[str, Dict[str, Any]] = {}
            for doc_id, doc in self.vector_store.docstore._dict.items():
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    doc_source = doc.metadata["source"]
                    doc_type = doc.metadata.get("document_type", "unknown")
                    if doc_source not in unique_docs:
                        unique_docs[doc_source] = {
                            "source": doc_source,
                            "document_type": doc_type,
                            "chunk_count": 0
                        }
                    elif (
                        unique_docs[doc_source]["document_type"] == "unknown"
                        and doc_type != "unknown"
                    ):
                        # Prefer a known type if an earlier chunk lacked one
                        unique_docs[doc_source]["document_type"] = doc_type
                    unique_docs[doc_source]["chunk_count"] += 1

            documents = list(unique_docs.values())

        return {
            "status": "success",
            "document_count": len(documents),
            "total_chunks": document_count,
            "documents": documents
        }

    def _provider_chain(self) -> List[tuple]:
        """Return available LLM providers as (client, model_name) tuples, primary first."""
        providers = []
        if self.client is not None:
            providers.append((self.client, self.settings.runpod_model_name))
        if self.fallback_client is not None:
            providers.append((self.fallback_client, self.settings.openai_model_name))
        return providers

    async def _generate_completion(self, messages: List[Dict[str, str]], stream: bool):
        """Generate a chat completion trying each configured provider in order.

        Raises:
            RuntimeError: When no provider is configured or all providers fail.
        """
        providers = self._provider_chain()
        if not providers:
            raise RuntimeError(
                "No LLM provider configured. Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID "
                "(or OPENAI_API_KEY for fallback) in the environment."
            )

        last_error: Optional[Exception] = None
        for client, model_name in providers:
            try:
                return await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=stream,
                    **self.model_params,
                )
            except Exception as e:
                last_error = e
                logger.warning(f"LLM provider {model_name} failed: {e}; trying next provider")

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def _extract_delta_content(self, chunk: Any) -> Optional[str]:
        """Safely extract delta content from a streaming chunk."""
        try:
            if not chunk.choices:
                return None
            return chunk.choices[0].delta.content
        except (AttributeError, IndexError):
            return None
    
    async def answer_question(
        self,
        query: str,
        trace: Optional[TraceLike] = None,
        generation: Optional[GenerationLike] = None,
    ) -> Dict[str, Any]:
        """Answer a question using the RAG pipeline."""
        try:
            start_time = time.time()

            # Get relevant documents from vector store
            retrieval_start = time.time()
            relevant_docs = self.vector_store.similarity_search(query, k=3)
            retrieval_time = time.time() - retrieval_start

            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            if generation:
                generation.end(
                    output=context,
                    metadata={
                        "retrieval_time": retrieval_time,
                        "document_count": len(relevant_docs)
                    }
                )

            messages = self._create_messages(context=context, question=query)

            completion = await self._generate_completion(messages, stream=False)

            response_text = completion.choices[0].message.content or ""
            response_text = self._clean_response(response_text)

            response_time = time.time() - start_time

            if trace:
                trace.update(output=response_text)

            return {
                "answer": response_text,
                "sources": [{"content": doc.page_content, "metadata": doc.metadata} for doc in relevant_docs],
                "metadata": {
                    "response_time": response_time,
                    "model": self.settings.runpod_model_name if self.client else self.settings.openai_model_name
                }
            }

        except Exception as e:
            logger.error(f"Error answering question: {e}", exc_info=True)
            if generation:
                generation.end(level='ERROR', status_message=str(e))
            if trace:
                trace.update(level='ERROR', status_message=str(e))
            raise

    async def astream_response(
        self,
        query: str,
        cancel_event: Optional[Event] = None,
        trace: Optional[TraceLike] = None,
        generation: Optional[GenerationLike] = None,
    ) -> AsyncIterator[str]:
        """Stream response to a question asynchronously."""
        start_time = time.time()

        try:
            # Check for early cancellation
            if cancel_event and cancel_event.is_set():
                logger.info("Generation cancelled before context retrieval")
                return

            retrieval_start = time.time()
            relevant_docs = self.vector_store.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            if generation:
                generation.end(
                    output=context,
                    metadata={
                        "retrieval_time": time.time() - retrieval_start,
                        "document_count": len(relevant_docs)
                    }
                )

            messages = self._create_messages(context=context, question=query)

            stream = await self._generate_completion(messages, stream=True)

            full_response = ""
            async for chunk in stream:
                if cancel_event and cancel_event.is_set():
                    logger.info("Generation cancelled")
                    break

                content = self._extract_delta_content(chunk)
                if content:
                    full_response += content
                    yield content

            if trace:
                trace.update(output=full_response)

        except Exception as e:
            if generation:
                generation.end(level='ERROR', status_message=str(e))
            if trace:
                trace.update(level='ERROR', status_message=str(e))

            logger.error(f"Error processing streaming query: {e}", exc_info=True)

            yield f"Error: {str(e)}"
