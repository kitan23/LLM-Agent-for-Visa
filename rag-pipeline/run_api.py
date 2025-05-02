#!/usr/bin/env python3
"""
Starter script for the OPT-RAG API server.
Run this script from the root directory to start the API server.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Make sure we can import from the rag-pipeline directory
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Change to the project directory
os.chdir(BASE_DIR)

if __name__ == "__main__":
    print(f"Starting OPT-RAG API from {BASE_DIR}")
    print("Server will be available at http://localhost:8000")
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 