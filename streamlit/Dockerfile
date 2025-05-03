# Use Python 3.10 as base image
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

# Create temp directory for file uploads
RUN mkdir -p /app/temp

# Copy the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 