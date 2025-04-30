"""
OPT-RAG International Student Visa Assistant - FastAPI Application

This module serves as the entrypoint for the FastAPI application that powers
the OPT-RAG International Student Visa Assistant.
"""

import logging
import argparse
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from llm.assistant import OPTRagAssistant
from utils.logging import setup_logging
from utils.metrics import initialize_metrics, APP_INFO
from utils.config import Settings, get_settings
from pydantic_settings import BaseSettings

# class Settings(BaseSettings):
#     model_path: str = ""
#     vector_store: str = ""
#     device: str = ""




# Configure logging 
# setup_logging()
logger = logging.getLogger("opt_rag.main")


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
# settings = Settings()

# Add OpenTelemetry instrumentation 
FastAPIInstrumentor.instrument_app(app)

# Initialize the OPT-RAG assistant 
assistant = None 

@app.on_event("startup")
async def startup_event():
    """Initialize resources on application startup"""
    global assistant 
    logger.info("Initializing OPT-RAG Assistant")

    # Initialize metrics with application info
    initialize_metrics(APP_INFO.version, APP_INFO.model_name)

    try: 

        # assistant = OPTRagAssistant(
        #     model_path = settings.model_path, 
        #     vector_store = settings.vector_store_path, 
        #     device = settings.device
        # )
        logger.info("OPT-RAG Assistant initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OPT-RAG assistant {e}")

@app.get("/metrics")
def metrics():
    """Endpoint for exposing the Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

class QueryRequest(BaseModel):
    """Query request model."""
    question: str 

@app.post("/api/query")
async def query(request: QueryRequest):
    """Standard query endpoint that returns complete response"""
    if not assistant: 
        return {"error": "OPT-RAG Assistant not initialized"}
    
    logger.info(f"Received query: {request.question}")
    result = assistant.answer_question(request.question)

    return {
        "answer": result["answer"], 
        "processing_time": result["processing_time"]
    }

@app.post("/api/query/stream")
async def stream_query(request: QueryRequest):
    """Streaming query endpoint."""
    if not assistant:
        return {"error": "OPT-RAG Assistant not initialized"}
    
    logger.info(f"Received streaming query: {request.question}")

    async def generate():
        async for token in assistant.astream_reponse(request.question):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(), 
        media_type="text/event-stream"
    )

@app.get("/health")
async def health():
    """Health check endpoint."""

    if assistant:
        return {"status":"healthy"}
    return {"status": "unhealthy", "reason":"Assistant is not initialized"}