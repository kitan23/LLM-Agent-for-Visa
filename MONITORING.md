# OPT-RAG Monitoring with Prometheus and Grafana

This guide explains how to set up monitoring for your OPT-RAG pipeline using Prometheus and Grafana.

## Prerequisites

- Docker and Docker Compose installed
- Python with FastAPI, Uvicorn, and Prometheus client

## Setup

1. Install the required Python packages:

```bash
pip install fastapi uvicorn prometheus-client
```

2. Start the API server:

```bash
python main.py 
```

This will:
- Start a FastAPI server on port 8000
- Load your OPT-RAG assistant
- Expose metrics at `/metrics`
- Provide API endpoints for querying and document management

3. Start Prometheus and Grafana using Docker Compose:

```bash
docker-compose up -d
```

## Accessing the Dashboards

- **Prometheus UI**: http://localhost:9090
- **Grafana**: http://localhost:3000 (login with admin/admin)

## Setting up Grafana

1. Log in to Grafana (http://localhost:3000) with:
   - Username: admin
   - Password: admin

2. Add Prometheus as a data source:
   - Go to Configuration > Data Sources > Add data source
   - Select Prometheus
   - URL: http://prometheus:9090
   - Click "Save & Test"

3. Import the dashboard:
   - Go to Create > Import
   - Upload the `dashboard.json` file or use dashboard ID
   - Select the Prometheus data source
   - Click "Import"

## Metrics Available

The OPT-RAG system exposes the following metrics:

- **QUERY_COUNT**: Counter for number of queries processed
- **QUERY_LATENCY**: Histogram for query processing time
- **MODEL_LOAD_TIME**: Histogram for model loading time
- **VECTOR_COUNT**: Gauge for the number of vectors in the store
- **VECTOR_RETRIEVAL_LATENCY**: Histogram for vector retrieval time
- **QUERY_ERRORS**: Counter for query errors by type

## Testing the Monitoring

To generate some traffic and metrics:

1. Make some API calls:

```bash
# Query endpoint
curl "http://localhost:8000/query?q=What%20is%20OPT%20and%20who%20is%20eligible%20for%20it%3F"

# Stream endpoint
curl "http://localhost:8000/stream?q=What%20is%20OPT%20and%20who%20is%20eligible%20for%20it%3F"

# Add document
curl -X POST "http://localhost:8000/documents" -H "Content-Type: application/json" -d '["examples/uscis_opt.pdf"]'
```

2. Watch the metrics update in Grafana

## Shutting Down

To stop the monitoring stack:

```bash
docker-compose down
``` 