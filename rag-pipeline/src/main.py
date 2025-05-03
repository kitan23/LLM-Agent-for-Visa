"""
OPT-RAG International Student Visa Assistant - FastAPI Application

This module serves as the entrypoint for the FastAPI application that powers
the OPT-RAG International Student Visa Assistant.
"""

import logging
import os
from typing import Dict, Any, Optional, List
import uvicorn
from fastapi import FastAPI, Request, Response, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, make_asgi_app

from src.llm.assistant import OPTRagAssistant
from src.utils.logging import setup_logging
from src.utils.metrics import initialize_metrics, APP_INFO
from src.utils.config import Settings, get_settings
from src.utils.tracing import setup_jaeger_tracing, get_tracer

# Configure logging 
setup_logging()
logger = logging.getLogger("opt_rag.main")

# Application info
APP_VERSION = "1.0.0"
APP_MODEL_NAME = "Qwen2.5-1.5b-instruct"

# Initialize FastAPI application 
app = FastAPI(
    title = "OPT-RAG API", 
    description = "International Student Visa Assistant API", 
    version = "1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()

# Setup Jaeger tracing
service_name = "opt-rag-service"
logger.info(f"Setting up tracing for service: {service_name}")
tracer_provider = setup_jaeger_tracing(app, service_name=service_name)
logger.info("Tracing setup complete")

# Add OpenTelemetry instrumentation 
# Note: We don't need this line as FastAPIInstrumentor is already initialized in setup_jaeger_tracing
# FastAPIInstrumentor.instrument_app(app)

# Add metrics endpoint using Prometheus ASGI app
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Initialize the OPT-RAG assistant 
assistant = None 

@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup"""
    global assistant 
    logger.info("Initializing OPT-RAG Assistant")

    # Initialize metrics with application info
    initialize_metrics(APP_VERSION, APP_MODEL_NAME)

    try: 
        model_path = os.environ.get("MODEL_PATH", settings.model_path)
        vector_store_path = os.environ.get("VECTOR_STORE_PATH", settings.vector_store_path)
        device = os.environ.get("DEVICE", settings.device)
        
        logger.info(f"Initializing OPT-RAG Assistant with model at {model_path}")
        logger.info(f"Using vector store at {vector_store_path}")
        
        assistant = OPTRagAssistant(
            model_path=model_path, 
            vector_store_path=vector_store_path, 
            device=device
        )
        logger.info("OPT-RAG Assistant initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OPT-RAG assistant {e}")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "OPT-RAG API is running"}

class QueryRequest(BaseModel):
    """Query request model."""
    question: str 

# Support both GET with query params and POST with JSON body
@app.post("/api/query")
async def query_post(request: QueryRequest):
    """Standard query endpoint that returns complete response (POST)"""
    if not assistant: 
        return {"error": "OPT-RAG Assistant not initialized"}
    
    # Create a span for this operation
    tracer = get_tracer()
    with tracer.start_as_current_span("query_post_operation"):
        logger.info(f"Received query: {request.question}")
        result = assistant.answer_question(request.question)

        return {
            "answer": result["answer"], 
            "processing_time": result["processing_time"]
        }

@app.get("/query", response_model=Dict[str, Any])
async def query_get(q: str = Query(..., description="Query text")):
    """Answer a question using OPT-RAG (GET)."""
    if not assistant:
        return {"error": "OPT-RAG Assistant not initialized"}
    
    # Create a span for this operation
    tracer = get_tracer()
    with tracer.start_as_current_span("query_get_operation"):
        logger.info(f"Received query: {q}")
        return assistant.answer_question(q)

@app.post("/api/query/stream")
async def stream_query_post(request: QueryRequest):
    """Streaming query endpoint (POST)."""
    if not assistant:
        return {"error": "OPT-RAG Assistant not initialized"}
    
    # Create a span for this operation
    tracer = get_tracer()
    with tracer.start_as_current_span("stream_query_post_operation"):
        logger.info(f"Received streaming query: {request.question}")

        async def generate():
            async for token in assistant.astream_response(request.question):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
    
        return StreamingResponse(
            generate(), 
            media_type="text/event-stream"
        )

@app.get("/stream")
async def stream_query_get(q: str = Query(..., description="Query text")):
    """Stream an answer using OPT-RAG (GET)."""
    if not assistant:
        return {"error": "OPT-RAG Assistant not initialized"}
    
    # Create a span for this operation
    tracer = get_tracer()
    with tracer.start_as_current_span("stream_query_get_operation"):
        logger.info(f"Received streaming query: {q}")
        
        return StreamingResponse(
            assistant.astream_response(q),
            media_type="text/event-stream"
        )

@app.post("/documents", response_model=Dict[str, Any])
async def add_documents(document_paths: List[str], document_type: Optional[str] = None):
    """Add documents to the vector store."""
    if not assistant:
        return {"error": "OPT-RAG Assistant not initialized"}
    
    # Create a span for this operation
    tracer = get_tracer()
    with tracer.start_as_current_span("add_documents_operation"):
        result = await assistant.add_documents(
            file_path=document_paths,
            document_type=document_type
        )
        return result

@app.get("/documents", response_model=Dict[str, Any])
async def list_documents():
    """List documents in the vector store."""
    if not assistant:
        return {"error": "OPT-RAG Assistant not initialized"}
    
    # Create a span for this operation
    tracer = get_tracer()
    with tracer.start_as_current_span("list_documents_operation"):
        return assistant.list_documents()

@app.get("/metrics/summary", response_model=Dict[str, Any])
async def metrics_summary():
    """Get a summary of metrics."""
    return {
        "query_count": "Available at /metrics",
        "query_latency": "Available at /metrics",
        "vector_count": "Available at /metrics",
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    if assistant:
        return {"status":"healthy"}
    return {"status": "unhealthy", "reason":"Assistant is not initialized"}

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )