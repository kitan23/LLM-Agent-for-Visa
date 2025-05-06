# OPT-RAG Helm Chart

This directory contains a Helm chart for deploying the OPT-RAG application on Kubernetes, either locally with Minikube or on Google Kubernetes Engine (GKE).

## Chart Structure

- `Chart.yaml`: Defines the chart metadata
- `values.yaml`: Contains default configuration values
- `templates/`: Directory containing Kubernetes manifest templates
  - `backend-deployment.yaml`: Backend deployment
  - `backend-service.yaml`: Backend service
  - `frontend-deployment.yaml`: Frontend deployment
  - `frontend-service.yaml`: Frontend service
  - `nginx-deployment.yaml`: NGINX gateway deployment
  - `nginx-service.yaml`: NGINX gateway service
  - `persistent-volumes.yaml`: Persistent volume claims
  - `namespace.yaml`: Namespace definition

## Prerequisites

- Kubernetes cluster (Minikube or GKE)
- Helm 3+ installed
- Docker installed
- For GKE: Google Cloud SDK installed and configured

## Deployment

### Option 1: Using the Deployment Script

The easiest way to deploy is using the provided script:

```bash
# For Minikube deployment
../helm-deploy.sh -e minikube

# For GCP deployment
../helm-deploy.sh -e gcp
```

### Option 2: Manual Deployment

#### For Minikube

1. Start Minikube:
   ```bash
   minikube start --cpus=4 --memory=7168
   ```

2. Set Minikube Docker environment:
   ```bash
   eval $(minikube docker-env)
   ```

3. Build Docker images:
   ```bash
   docker build -t opt-rag-backend:v1 -f ../../rag-pipeline/Dockerfile ../../rag-pipeline/
   docker build -t opt-rag-frontend:v1 -f ../../streamlit/Dockerfile ../../streamlit/
   docker build -t opt-rag-gateway:v1 -f ../../nginx/Dockerfile ../../nginx/
   ```

4. Create the namespace:
   ```bash
   kubectl create namespace opt-rag
   ```

5. Install the Helm chart:
   ```bash
   helm install opt-rag . \
     --namespace opt-rag \
     --set images.registry="" \
     --set services.backend.type=NodePort \
     --set services.frontend.type=NodePort \
     --set services.nginx.type=NodePort
   ```

#### For GKE

1. Set up Google Cloud SDK:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. Create GKE cluster (if you don't have one):
   ```bash
   gcloud container clusters create opt-rag-cluster \
     --zone us-central1-a \
     --num-nodes=3 \
     --machine-type=e2-standard-4
   ```

3. Configure Docker for GCR:
   ```bash
   gcloud auth configure-docker
   ```

4. Build and push Docker images:
   ```bash
   docker build -t gcr.io/YOUR_PROJECT_ID/opt-rag-backend:v1 -f ../../rag-pipeline/Dockerfile ../../rag-pipeline/
   docker push gcr.io/YOUR_PROJECT_ID/opt-rag-backend:v1
   
   docker build -t gcr.io/YOUR_PROJECT_ID/opt-rag-frontend:v1 -f ../../streamlit/Dockerfile ../../streamlit/
   docker push gcr.io/YOUR_PROJECT_ID/opt-rag-frontend:v1
   
   docker build -t gcr.io/YOUR_PROJECT_ID/opt-rag-gateway:v1 -f ../../nginx/Dockerfile ../../nginx/
   docker push gcr.io/YOUR_PROJECT_ID/opt-rag-gateway:v1
   ```

5. Create the namespace:
   ```bash
   kubectl create namespace opt-rag
   ```

6. Install the Helm chart:
   ```bash
   helm install opt-rag . \
     --namespace opt-rag \
     --set images.registry=gcr.io/YOUR_PROJECT_ID/ \
     --set services.backend.type=LoadBalancer \
     --set services.frontend.type=LoadBalancer \
     --set services.nginx.type=LoadBalancer
   ```

## Configuration

The following table lists the configurable parameters and their default values:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `images.registry` | Container registry | `""` |
| `images.backend.repository` | Backend image repository | `opt-rag-backend` |
| `images.backend.tag` | Backend image tag | `v1` |
| `images.frontend.repository` | Frontend image repository | `opt-rag-frontend` |
| `images.frontend.tag` | Frontend image tag | `v1` |
| `images.nginx.repository` | NGINX image repository | `opt-rag-gateway` |
| `images.nginx.tag` | NGINX image tag | `v1` |
| `replicaCount.backend` | Number of backend replicas | `1` |
| `replicaCount.frontend` | Number of frontend replicas | `1` |
| `replicaCount.nginx` | Number of NGINX replicas | `1` |
| `services.backend.type` | Backend service type | `NodePort` |
| `services.frontend.type` | Frontend service type | `NodePort` |
| `services.nginx.type` | NGINX service type | `NodePort` |
| `persistentVolume.models.size` | Size of models PVC | `5Gi` |
| `persistentVolume.vectorStore.size` | Size of vector store PVC | `1Gi` |

## Accessing the Application

### On Minikube

```bash
# Backend API
kubectl port-forward -n opt-rag svc/backend-service 8000:8000

# Frontend UI
kubectl port-forward -n opt-rag svc/frontend-service 8501:8501

# NGINX Gateway
kubectl port-forward -n opt-rag svc/nginx-service 8080:80
```

### On GKE

Get the external IPs:

```bash
kubectl get svc -n opt-rag
```

Access the services using their external IPs:
- Backend API: `http://<backend-external-ip>:8000/docs`
- Frontend UI: `http://<frontend-external-ip>:8501`
- NGINX Gateway: `http://<nginx-external-ip>:80`

## Uninstalling the Chart

```bash
helm uninstall opt-rag -n opt-rag
```

This will remove all the Kubernetes resources associated with the chart. 