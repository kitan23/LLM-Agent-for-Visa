"""
Prometheus metrics for OPT-RAG.

This module defines Prometheus metrics for monitoring the performance
and usage of the OPT-RAG application.
"""

from prometheus_client import Counter, Gauge, Histogram, Info

# Application info metric (for version tracking)
APP_INFO = Info(
    "opt_rag_info", 
    "Information about the OPT-RAG application"
)

# Query metrics
QUERY_COUNT = Counter(
    "opt_rag_query_count",   # metric name
    "Total number of queries processed",  # metric description
    ["status"],  # labels
)

QUERY_LATENCY = Histogram(
    "opt_rag_query_latency_seconds",
    "Query processing time in seconds",
    buckets = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

QUERY_ERRORS = Counter(
    "opt_rag_query_errors", 
    "Number of query errors"
)

# Model metrics
MODEL_LOAD_TIME = Histogram(
    "opt_rag_model_load_time_seconds", 
    "Time to load the LLM model in seconds",
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
)

MEMORY_USAGE = Gauge(
    "opt_rag_memory_usage_bytes", 
    "Memory usage in bytes"
)

GPU_MEMORY_USAGE = Gauge(
    "opt_rag_gpu_memory_usage_bytes", 
    "GPU memory usage in bytes"
)

# Vector store metrics
VECTOR_RETRIEVAL_LATENCY = Histogram(
    "opt_rag_vector_retrieval_latency_seconds", 
    "Vector retrieval time in seconds",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

VECTOR_COUNT = Gauge(
    "opt_rag_vector_count", 
    "Number of vectors in the vector store"
)


# Function to initialize metrics with application info
def initialize_metrics(app_version, model_name):
    """Initialize metric labels and set application info.
    
    Args:
        app_version: Version of the application
        model_name: Name of the LLM model being used
    """
    APP_INFO.info({"version": app_version, "model": model_name})