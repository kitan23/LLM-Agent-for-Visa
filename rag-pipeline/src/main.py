"""
OPT-RAG International Student Visa Assistant - FastAPI Application

This module serves as the entrypoint for the FastAPI application that powers
the OPT-RAG International Student Visa Assistant.
"""

import logging
import argparse
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from llm.assistant import OPTRagAssistant
from utils.logging import setup_logging
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
    try: 
        # assistant = OPTRagAssistant(
        #     model_path = settings.model_path, 
        #     vector_store = settings.vector_store_path, 
        #     device = settings.device
        # )
        logger.info("OPT-RAG Assistant initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OPT-RAG assistant {e}")


class QueryRequest(BaseModel):
    """Query request model."""
    question: str 


@app.get("/health")
async def health():
    """Health check endpoint."""

    if assistant:
        return {"status":"healthy"}
    return {"status": "unhealthy", "reason":"Assistant is not initialized"}