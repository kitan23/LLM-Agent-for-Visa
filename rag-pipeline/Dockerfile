# Use Python 3.10 as base image (more stable than 3.11 for some packages)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables to prevent Python from writing bytecode and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install dependencies with pip upgrade
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create directories if they don't exist
RUN mkdir -p /app/models /app/vector_store

# Copy the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"] 