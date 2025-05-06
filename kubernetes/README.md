# OPT-RAG Kubernetes Deployment Guide

This directory contains Kubernetes manifests and scripts for deploying the OPT-RAG application on Kubernetes, including Google Kubernetes Engine (GKE).

## Directory Structure

- `cloud/` - Kubernetes manifests for cloud deployment (GKE)
  - `backend.yaml` - Backend service deployment and PVCs
  - `frontend.yaml` - Streamlit frontend deployment
  - `ingress.yaml` - NGINX gateway and Ingress resources
  - `monitoring.yaml` - Prometheus, Grafana, and Jaeger deployments

- `deploy-to-gke.sh` - Deployment script for GKE

## Prerequisites

Before deploying to GKE, make sure you have the following:

1. **Google Cloud SDK**: Install from [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)
2. **Docker**: Install from [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
3. **kubectl**: Install from [https://kubernetes.io/docs/tasks/tools/](https://kubernetes.io/docs/tasks/tools/)
4. **Google Cloud account**: With sufficient permissions to create GKE clusters
5. **GCP Project**: With billing enabled and necessary APIs activated:
   - Kubernetes Engine API
   - Container Registry API
   - Cloud Storage API

## GKE Deployment Steps

### 1. Set Up Google Cloud SDK

If you haven't installed the Google Cloud SDK yet:

```bash
# macOS (with Homebrew)
brew install --cask google-cloud-sdk

# Verify installation
gcloud --version
```

Initialize gcloud configuration:

```bash
gcloud init
```

### 2. Run the Deployment Script

The easiest way to deploy the application is to use the provided script:

```bash
./deploy-to-gke.sh -p YOUR_GCP_PROJECT_ID
```

Script options:
- `-p, --project-id`: GCP project ID (required)
- `-c, --cluster-name`: GKE cluster name (default: opt-rag-cluster)
- `-z, --zone`: GCP zone (default: us-central1-a)
- `-r, --registry`: Container registry (default: gcr.io)
- `-v, --version`: Image version (default: v1)

### 3. Manual Deployment

If you prefer to deploy manually:

1. Build and push Docker images:
   ```bash
   # Authenticate Docker with GCR
   gcloud auth configure-docker
   
   # Build and push backend
   docker build -t gcr.io/YOUR_PROJECT_ID/opt-rag-backend:v1 ./rag-pipeline/
   docker push gcr.io/YOUR_PROJECT_ID/opt-rag-backend:v1
   
   # Build and push frontend
   docker build -t gcr.io/YOUR_PROJECT_ID/opt-rag-frontend:v1 ./streamlit/
   docker push gcr.io/YOUR_PROJECT_ID/opt-rag-frontend:v1
   
   # Build and push NGINX
   docker build -t gcr.io/YOUR_PROJECT_ID/opt-rag-gateway:v1 ./nginx/
   docker push gcr.io/YOUR_PROJECT_ID/opt-rag-gateway:v1
   ```

2. Create GKE cluster:
   ```bash
   gcloud container clusters create opt-rag-cluster \
     --project YOUR_PROJECT_ID \
     --zone us-central1-a \
     --num-nodes=3 \
     --machine-type=e2-standard-4
   ```

3. Update image references in Kubernetes manifests:
   ```bash
   # Use sed or manually update ${DOCKER_REGISTRY} in all YAML files
   sed -i 's/${DOCKER_REGISTRY}/gcr.io\/YOUR_PROJECT_ID/g' kubernetes/cloud/*.yaml
   ```

4. Apply Kubernetes manifests:
   ```bash
   kubectl create namespace opt-rag
   kubectl apply -f kubernetes/cloud/backend.yaml
   kubectl apply -f kubernetes/cloud/frontend.yaml
   kubectl apply -f kubernetes/cloud/monitoring.yaml
   kubectl apply -f kubernetes/cloud/ingress.yaml
   ```

## Accessing the Application

After deployment, you can access the application through the external IP assigned to the LoadBalancer:

```bash
# Get service external IPs
kubectl get svc -n opt-rag

# For Ingress (may take a few minutes to assign IP)
kubectl get ingress -n opt-rag
```

- Backend API: `http://EXTERNAL-IP:8000/docs`
- Frontend UI: `http://EXTERNAL-IP:8501`
- Monitoring:
  - Prometheus: Port-forward to access `kubectl port-forward -n opt-rag svc/prometheus 9090:9090`
  - Grafana: Port-forward to access `kubectl port-forward -n opt-rag svc/grafana 3000:3000`
  - Jaeger: Port-forward to access `kubectl port-forward -n opt-rag svc/jaeger 16686:16686`

## Cleanup

To delete the GKE resources:

```bash
# Delete the namespace and all resources in it
kubectl delete namespace opt-rag

# Delete the GKE cluster
gcloud container clusters delete opt-rag-cluster --zone us-central1-a
```

## Troubleshooting

- **Pod status issues**: Use `kubectl describe pod POD_NAME -n opt-rag` to get detailed information about pod status and errors.
- **Service connectivity**: Ensure services are correctly exposing ports with `kubectl get svc -n opt-rag` and `kubectl describe svc SERVICE_NAME -n opt-rag`.
- **Log inspection**: Check container logs with `kubectl logs POD_NAME -n opt-rag -c CONTAINER_NAME`.
- **Resource limitations**: Adjust resource requests and limits in deployment manifests if pods are failing due to resource constraints. 