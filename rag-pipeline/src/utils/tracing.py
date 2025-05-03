"""
Distributed tracing configuration for OPT-RAG.

This module defines the tracing configuration using OpenTelemetry and OTLP.
It provides functions to set up and initialize tracing for the application.
"""

import logging
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

logger = logging.getLogger("opt_rag.tracing")

def setup_jaeger_tracing(app=None, service_name="opt-rag-service"):
    """Set up OTLP tracing for the application.
    
    Args:
        app: FastAPI application instance (optional)
        service_name: Name of the service for tracing
        
    Returns:
        The configured tracer provider
    """
    try:
        # Create a resource with service name and other attributes
        resource = Resource(attributes={
            SERVICE_NAME: service_name,
            "deployment.environment": "development"
        })
        
        # Create a TracerProvider with the service resource
        tracer_provider = TracerProvider(resource=resource)
        
        # Create an OTLP exporter
        otlp_endpoint = os.environ.get("OTLP_ENDPOINT", "http://localhost:4317")
        logger.info(f"Connecting to OTLP endpoint: {otlp_endpoint}")
        
        otlp_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True
        )
        
        # Add the exporter to the tracer provider
        tracer_provider.add_span_processor(
            BatchSpanProcessor(otlp_exporter)
        )
        
        # Set the global tracer provider
        trace.set_tracer_provider(tracer_provider)
        
        # Instrument FastAPI if an app is provided
        if app:
            logger.info(f"Instrumenting FastAPI app with service name: {service_name}")
            FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)
        
        # Instrument requests to trace outgoing HTTP calls
        RequestsInstrumentor().instrument(tracer_provider=tracer_provider)
        
        logger.info(f"OTLP tracing initialized for service: {service_name}")
        return tracer_provider
    
    except Exception as e:
        logger.error(f"Failed to initialize OTLP tracing: {e}")
        # Return default tracer provider if configuration fails
        return trace.get_tracer_provider()

def get_tracer(tracer_name="opt-rag-tracer"):
    """Get a tracer for creating spans.
    
    Args:
        tracer_name: Name of the tracer
        
    Returns:
        A configured tracer
    """
    return trace.get_tracer(tracer_name) 