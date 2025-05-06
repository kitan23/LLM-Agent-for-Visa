# OPT-RAG: International Student Visa Assistant

OPT-RAG is a Retrieval-Augmented Generation (RAG) application designed to help international students navigate visa-related issues, OPT applications, and other immigration concerns.

## Project Overview

The OPT-RAG application uses retrieval-augmented generation to provide accurate information by retrieving relevant content from official documentation and policies. The application consists of:

- **Backend (FastAPI)**: Processes documents, maintains vector store, and handles queries
- **Frontend (Streamlit)**: Provides user interface for interacting with the assistant
- **API Gateway (NGINX)**: Routes requests between services
- **Monitoring Stack**: Prometheus, Grafana, and Jaeger for observability

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- For cloud deployment: [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)

### Local Development

1. Clone the repository:
   ```
   git clone <repository-url>
   cd OPT-RAG
   ```

2. Start the application using Docker Compose:
   ```
   docker-compose up -d
   ```

3. Access the services:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000/docs
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000
   - Jaeger: http://localhost:16686

### Cloud Deployment (GKE)

#### Step 1: Install Google Cloud SDK

See [kubernetes/gcloud-install-guide.md](kubernetes/gcloud-install-guide.md) for detailed instructions on installing and setting up the Google Cloud SDK.

#### Step 2: Deploy to Google Kubernetes Engine

The simplest way to deploy the application to GKE is to use the provided script:

```bash
chmod +x kubernetes/deploy-to-gke.sh
./kubernetes/deploy-to-gke.sh -p YOUR_GCP_PROJECT_ID
```

For more detailed instructions, refer to [kubernetes/README.md](kubernetes/README.md).

## Project Structure

The project follows a modular structure:

- `rag-pipeline/`: Backend service with document processing and vector store
- `streamlit/`: Frontend UI for user interaction
- `nginx/`: API Gateway configuration
- `kubernetes/`: Kubernetes deployment manifests
- `prometheus/`, `grafana/`, `jaeger/`: Monitoring components

## Features

- Upload and process official immigration documents
- Ask questions in natural language about visa and immigration topics
- Get context-aware responses based on official documentation
- Reference sources used to generate answers
- Maintain conversation context

## Monitoring

The application includes a comprehensive monitoring stack:

- **Prometheus**: Collects metrics from all services
- **Grafana**: Visualizes metrics with customizable dashboards
- **Jaeger**: Distributed tracing for request flows

## Documentation

- [Project Structure](project-structure.md)
- [PRD (Product Requirements Document)](prd.md)
- [Kubernetes Deployment Guide](kubernetes/README.md)
- [NGINX Gateway](NGINX-GATEWAY.md)
- [Monitoring Setup](MONITORING.md)
- [GKE Deployment Guide](opt-rag-gke-deployment-guide.md)

## License

[MIT License](LICENSE)

## Disclaimer

This assistant provides information based on available documents. It is not a substitute for legal advice. 