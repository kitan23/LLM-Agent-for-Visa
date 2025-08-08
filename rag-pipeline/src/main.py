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
from langfuse import Langfuse

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

# Setup Langfuse
langfuse = Langfuse()

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

    try:
        # Get vector store path
        vector_store_path = os.environ.get("VECTOR_STORE_PATH", "./vector_store")
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
    
    trace = langfuse.trace(
        name = "rag-pipeline",
        user_id = "user@example.com", # Replace with actual user id
        metadata = {
            "query": request.question
        }
    )

    generation = trace.generation(
        name = "answer-generation",
        model = APP_MODEL_NAME,
        input = request.question,
        metadata = {
            "interface": "api-post"
        }
    )
    
    result = await assistant.answer_question(request.question, trace=trace, generation=generation)

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
    
    trace = langfuse.trace(
        name = "rag-pipeline",
        user_id = "user@example.com", # Replace with actual user id
        metadata = {
            "query": q
        }
    )

    generation = trace.generation(
        name = "answer-generation",
        model = APP_MODEL_NAME,
        input = q,
        metadata = {
            "interface": "api-get"
        }
    )

    return await assistant.answer_question(q, trace=trace, generation=generation)

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
    
    trace = langfuse.trace(
        name = "rag-pipeline-stream",
        user_id = "user@example.com", # Replace with actual user id
        metadata = {
            "query": request.question
        }
    )

    generation = trace.generation(
        name = "answer-generation-stream",
        model = APP_MODEL_NAME,
        input = request.question,
        metadata = {
            "interface": "api-post-stream"
        }
    )

    async def generate():
        token_count = 0
        full_response = ""
        
        try:
            # Send request ID first
            yield f"data: {json.dumps({'request_id': request_id})}\n\n"
            
            # Pass the question and cancel event to the assistant
            stream_iter = assistant.astream_response(request.question, cancel_event=cancel_event, trace=trace, generation=generation)
            
            async for token in stream_iter:
                if cancel_event.is_set():
                    logger.info(f"Stream {request_id} cancelled by client")
                    break
                
                token_count += 1
                full_response += token
                
                # Format for SSE
                yield f"data: {json.dumps({'token': token})}\n\n"

            # Signal completion
            yield "data: [DONE]\n\n"
            
        except asyncio.CancelledError:
            logger.info(f"Stream {request_id} was cancelled.")
        except Exception as e:
            logger.error(f"Error in stream {request_id}: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            generation.end(output=full_response)
            langfuse.flush()
            # Clean up active generation
            if request_id in active_generations:
                del active_generations[request_id]
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/stream")
@api_router.get("/stream")
async def stream_query_get(q: str = Query(..., description="Query text")):
    """Stream an answer using OPT-RAG (GET)."""
    if not assistant:
        return {"error": "OPT-RAG Assistant not initialized"}
    
    trace = langfuse.trace(
        name = "rag-pipeline-stream",
        user_id = "user@example.com", # Replace with actual user id
        metadata = {
            "query": q
        }
    )

    generation = trace.generation(
        name = "answer-generation-stream",
        model = APP_MODEL_NAME,
        input = q,
        metadata = {
            "interface": "api-get-stream"
        }
    )
        
    async def generate():
        token_count = 0
        full_response = ""
        logger.info("Starting SSE stream generation for GET request")
        try:
            async for token in assistant.astream_response(q, trace=trace, generation=generation):
                token_count += 1
                full_response += token
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
        finally:
            generation.end(output=full_response)
            langfuse.flush()
    
    logger.info("Returning GET StreamingResponse")
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

@app.post("/cancel/{request_id}")
async def cancel_stream(request_id: str):
    """Cancel a running streaming request."""
    if request_id in active_generations:
        active_generations[request_id].set()
        logger.info(f"Cancellation requested for stream {request_id}")
        return {"status": "cancellation_requested"}
    return {"status": "not_found"}

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
    # This is now a legacy endpoint. Metrics are pushed to Langfuse.
    return {
        "status": "Metrics are now reported to Langfuse.",
        "query_count": "N/A",
        "query_latency": "N/A",
        "vector_count": "N/A",
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