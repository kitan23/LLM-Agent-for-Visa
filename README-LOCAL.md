# OPT-RAG Local Setup Guide

This guide explains how to run OPT-RAG locally using Docker Compose.

## Prerequisites

- Docker and Docker Compose installed
- Model files already downloaded in `rag-pipeline/models/qwen2.5-1.5b`

## Running the Application

1. Make sure you have model files in the correct location:
   ```
   rag-pipeline/models/qwen2.5-1.5b/
   ```

2. Start all services using Docker Compose:
   ```bash
   docker-compose up -d
   ```

3. To check if services are running:
   ```bash
   docker-compose ps
   ```

4. To view logs of a specific service:
   ```bash
   docker-compose logs -f api        # Backend logs
   docker-compose logs -f streamlit-ui # Frontend logs
   ```

## Accessing the Application

- Frontend UI: http://localhost:8501
- Backend API: http://localhost:8000/docs
- Monitoring:
  - Grafana: http://localhost:3000 (admin/admin)
  - Prometheus: http://localhost:9090
  - Jaeger: http://localhost:16686

## Stopping the Application

To stop all running services:
```bash
docker-compose down
```

## Troubleshooting

If you encounter any issues:

1. Check logs for errors:
   ```bash
   docker-compose logs
   ```

2. Restart a specific service:
   ```bash
   docker-compose restart api
   ```

3. Rebuild and restart a service:
   ```bash
   docker-compose build api
   docker-compose up -d api
   ``` 