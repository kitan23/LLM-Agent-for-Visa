"""
OPT-RAG Assistant Implementation

This module contains the core implementation of the OPT-RAG International Student 
Visa Assistant, handling model loading, vector store management, and query processing.
"""


import logging
import time
import asyncio
from typing import Dict, Any, Optional, AsyncIterator, Iterator
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from llm.callbacks import StreamingCallbackHandler, AsyncStreamingCallbackHandler
from utils.metrics import QUERY_LATENCY, QUERY_COUNT, MODEL_LOAD_TIME
from retriever.vector_store import load_vector_store

logger = logging.getLogger("opt_rag.assistant")


class OPTRagAssistant: 
    """OPT-RAG International Student Visa Assistant using RAG architecture."""

    def __init__(
        self, 
        model_path: str, 
        vector_store_path: str, 
        device: Optional[str] = None
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

        # Detect hardware if not specified
        self.device = device or self._detect_hardware()

        logger.info(f"Using device: {self.device}")

        # initialize components 
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.vector_store = load_vector_store(
            vector_store_path = self.vector_store_path
            device=self.device
        )

        # Store prompt template
        self.visa_prompt = self._create_prompt_template()

    


