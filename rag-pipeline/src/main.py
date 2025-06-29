"""
OPT-RAG International Student Visa Assistant - FastAPI Application

This module serves as the entrypoint for the FastAPI application that powers
the OPT-RAG International Student Visa Assistant.
"""

from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")
import logging
import os
from typing import Dict, Any, Optional, List
import uvicorn
from fastapi import FastAPI, Request, Response, Query, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, make_asgi_app
import shutil
from pathlib import Path
import json
import uuid
from threading import Event
from fastapi import HTTPException
import time
import asyncio

from src.llm.assistant import OPTRagAssistant
from src.utils.logging import setup_logging
from src.utils.metrics import initialize_metrics, APP_INFO
from src.utils.config import Settings, get_settings
from src.utils.tracing import setup_jaeger_tracing, get_tracer

# Set up detailed logging for streaming-related modules
logging.getLogger("opt_rag.assistant").setLevel(logging.DEBUG)
logging.getLogger("opt_rag.callbacks").setLevel(logging.DEBUG)
logging.getLogger("opt_rag.main").setLevel(logging.DEBUG)

# Configure logging 
setup_logging()
logger = logging.getLogger("opt_rag.main")

# Application info
APP_VERSION = "1.0.0"
APP_MODEL_NAME = "Qwen2.5-1.5b-instruct"

# Initialize FastAPI application 
app = FastAPI(
    title = "Visa RAG Assistant", 
    description = "International Student Visa Assistant API", 
    version = "1.0.0"
)

# Create API router with prefix
api_router = FastAPI(
    title = "OPT-RAG API Routes",
    description = "API routes for OPT-RAG",
    version = "1.0.0"
)

# Add CORS middleware to main app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add CORS middleware to API router
api_router.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()

# Setup Jaeger tracing
service_name = "opt-rag-service"
logger.info(f"Setting up tracing for service: {service_name}")
# tracer_provider = setup_jaeger_tracing(app, service_name=service_name)
logger.info("Tracing setup complete")

# Add OpenTelemetry instrumentation 
# Note: We don't need this line as FastAPIInstrumentor is already initialized in setup_jaeger_tracing
# FastAPIInstrumentor.instrument_app(app)

# Add metrics endpoint using Prometheus ASGI app
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Initialize the OPT-RAG assistant 
assistant = None 

# Add this after the app initialization
# Dictionary to store active generation tasks and their cancellation events
active_generations = {}

@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup."""
    global assistant 
    logger.info("Initializing OPT-RAG Assistant")

    # Initialize metrics with application info
    initialize_metrics(APP_VERSION, APP_MODEL_NAME)

    try:
        # Get vector store path
        vector_store_path = os.environ.get("VECTOR_STORE_PATH", settings.vector_store_path)
        logger.info(f"Using vector store at {vector_store_path}")
        
        # Configure RunPod model parameters
        model_kwargs = {
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        
        # Initialize assistant
        assistant = OPTRagAssistant(
            vector_store_path=vector_store_path,
            model_kwargs=model_kwargs
        )
        logger.info("OPT-RAG Assistant initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize OPT-RAG assistant: {e}")
        raise

# Add routes to both the main app and the API router
# This maintains backward compatibility while also supporting /api/* routes

# Root endpoint
@app.get("/", response_model=Dict[str, str])
@api_router.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "OPT-RAG API is running"}

class QueryRequest(BaseModel):
    """Query request model."""
    question: str 

# Support both GET with query params and POST with JSON body
@app.post("/api/query")
@api_router.post("/query")
async def query_post(request: QueryRequest):
    """Standard query endpoint that returns complete response (POST)"""
    if not assistant: 
        return {"error": "OPT-RAG Assistant not initialized"}
    
    # Create a span for this operation
    tracer = get_tracer()
    with tracer.start_as_current_span("query_post_operation"):
        logger.info(f"Received query: {request.question}")
        result = await assistant.answer_question(request.question)

        return {
            "answer": result["answer"], 
            "processing_time": result["metadata"]["response_time"]
        }

@app.get("/query", response_model=Dict[str, Any])
@api_router.get("/query", response_model=Dict[str, Any])
async def query_get(q: str = Query(..., description="Query text")):
    """Answer a question using OPT-RAG (GET)."""
    if not assistant:
        return {"error": "OPT-RAG Assistant not initialized"}
    
    # Create a span for this operation
    tracer = get_tracer()
    with tracer.start_as_current_span("query_get_operation"):
        logger.info(f"Received query: {q}")
        return await assistant.answer_question(q)

# Create a model for the cancellation request
class CancelRequest(BaseModel):
    request_id: str

@app.post("/api/cancel")
@api_router.post("/cancel")
async def cancel_generation(request: CancelRequest):
    """Cancel an ongoing generation task.
    
    Args:
        request: A CancelRequest containing the request_id to cancel
    """
    request_id = request.request_id
    logger.info(f"Received cancellation request for generation {request_id}")
    
    if request_id in active_generations:
        # Set the cancellation event
        active_generations[request_id].set()
        logger.info(f"Cancellation event set for generation {request_id}")
        
        # Sleep briefly to allow the cancellation to propagate
        await asyncio.sleep(0.1)
        
        return {
            "status": "success", 
            "message": f"Generation {request_id} cancellation requested",
            "cancelled": True
        }
    else:
        logger.warning(f"Attempted to cancel unknown generation ID: {request_id}")
        return {
            "status": "error",
            "message": f"Generation ID {request_id} not found",
            "cancelled": False
        }

@app.post("/api/query/stream")
@api_router.post("/query/stream")
async def stream_query_post(request: QueryRequest):
    """Streaming query endpoint (POST)."""
    if not assistant:
        return {"error": "OPT-RAG Assistant not initialized"}
    
    # Create a request ID and cancellation event
    request_id = str(uuid.uuid4())
    cancel_event = Event()
    active_generations[request_id] = cancel_event
    
    # Create a span for this operation
    tracer = get_tracer()
    with tracer.start_as_current_span("stream_query_post_operation"):
        logger.info(f"Received streaming query: {request.question} (ID: {request_id})")

        async def generate():
            token_count = 0
            full_response = ""
            
            try:
                # Send request ID first
                yield f"data: {json.dumps({'request_id': request_id})}\n\n"
                
                # Pass the question and cancel event to the assistant
                stream_iter = assistant.astream_response(request.question, cancel_event=cancel_event)
                
                while True:
                    try:
                        # Wait for the next token with a 10-second timeout
                        token = await asyncio.wait_for(anext(stream_iter), 10)
                        
                        if cancel_event.is_set():
                            logger.info(f"Generation {request_id} was cancelled by client request")
                            yield f"data: {json.dumps({'status': 'cancelled'})}\n\n"
                            break

                        token_count += 1
                        full_response += token
                        response_data = {"token": token, "full_response": full_response}
                        yield f"data: {json.dumps(response_data)}\n\n"

                    except asyncio.TimeoutError:
                        logger.debug("Sending heartbeat to keep connection alive")
                        yield ": heartbeat\n\n"
                    except StopAsyncIteration:
                        break # The stream is finished
                
                if not cancel_event.is_set():
                    logger.info(f"Stream complete for request {request_id}. Sent {token_count} tokens.")
                
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Error in SSE generation for request {request_id}: {e}", exc_info=True)
                error_data = {"error": f"An unexpected error occurred: {e}"}
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                # Clean up the active generation
                if request_id in active_generations:
                    logger.info(f"Cleaning up resources for generation {request_id}")
                    del active_generations[request_id]
    
        logger.info("Returning StreamingResponse")
        return StreamingResponse(
            generate(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Prevent proxy buffering
                "X-Request-ID": request_id  # Include the request ID in response headers
            }
        )

@app.get("/stream")
@api_router.get("/stream")
async def stream_query_get(q: str = Query(..., description="Query text")):
    """Stream an answer using OPT-RAG (GET)."""
    if not assistant:
        return {"error": "OPT-RAG Assistant not initialized"}
    
    # Create a span for this operation
    tracer = get_tracer()
    with tracer.start_as_current_span("stream_query_get_operation"):
        logger.info(f"Received streaming query: {q}")
        
        async def generate():
            token_count = 0
            logger.info("Starting SSE stream generation for GET request")
            try:
                async for token in assistant.astream_response(q):
                    token_count += 1
                    if token_count % 10 == 0:
                        logger.info(f"Streaming GET: sent {token_count} SSE events")
                    # Properly format for SSE, escape any newlines in the token
                    escaped_token = json.dumps(token)
                    yield f"data: {escaped_token}\n\n"
                logger.info(f"GET stream complete. Sent {token_count} tokens.")
                # Signal completion
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error in GET SSE generation: {str(e)}")
                error_json = json.dumps({"error": str(e)})
                yield f"data: {error_json}\n\n"
                yield "data: [DONE]\n\n"
        
        logger.info("Returning GET StreamingResponse")
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )

@app.post("/documents", response_model=Dict[str, Any])
@api_router.post("/documents", response_model=Dict[str, Any])
async def add_documents(
    file: UploadFile = File(...),
    document_type: Optional[str] = Form(None)
):
    """Add a document to the vector store.
    
    This endpoint accepts file uploads via multipart/form-data.
    """
    if not assistant:
        return {"error": "OPT-RAG Assistant not initialized"}
    
    # Create a span for this operation
    tracer = get_tracer()
    with tracer.start_as_current_span("add_documents_operation"):
        # Create temp directory if it doesn't exist
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Save the uploaded file temporarily
        file_path = temp_dir / file.filename
        
        try:
            # Save uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process the document
            logger.info(f"Processing uploaded document: {file.filename}")
            result = await assistant.add_documents(
                file_path=[str(file_path)],
                document_type=document_type
            )
            
            # Return success response
            return result
        
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {"status": "error", "message": str(e)}
        
        finally:
            # Clean up temporary file
            if file_path.exists():
                file_path.unlink()

@app.get("/documents", response_model=Dict[str, Any])
@api_router.get("/documents", response_model=Dict[str, Any])
async def list_documents():
    """List documents in the vector store."""
    if not assistant:
        return {"error": "OPT-RAG Assistant not initialized"}
    
    # Create a span for this operation
    tracer = get_tracer()
    with tracer.start_as_current_span("list_documents_operation"):
        return assistant.list_documents()

@app.get("/metrics/summary", response_model=Dict[str, Any])
@api_router.get("/metrics/summary", response_model=Dict[str, Any])
async def metrics_summary():
    """Get a summary of metrics."""
    return {
        "query_count": "Available at /metrics",
        "query_latency": "Available at /metrics",
        "vector_count": "Available at /metrics",
    }

@app.get("/health")
@api_router.get("/health")
async def health():
    """Health check endpoint."""
    if assistant:
        return {"status":"healthy"}
    return {"status": "unhealthy", "reason":"Assistant is not initialized"}

# Mount the API router at /api prefix
app.mount("/api", api_router)

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )