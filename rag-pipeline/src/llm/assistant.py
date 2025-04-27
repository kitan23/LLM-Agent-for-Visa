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

