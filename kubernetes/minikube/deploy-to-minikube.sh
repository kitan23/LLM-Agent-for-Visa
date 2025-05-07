#!/bin/bash

# Exit on any error
set -e

# Set Minikube's Docker environment variables so that Docker builds go to Minikube's registry
echo "Setting Minikube Docker environment..."
eval $(minikube docker-env)

echo "Building Docker images within Minikube..."

# Build backend image
echo "Building backend image..."
docker build -t opt-rag-backend:v1 -f rag-pipeline/Dockerfile rag-pipeline/

# Build frontend image
echo "Building frontend image..."
docker build -t opt-rag-frontend:v1 -f streamlit/Dockerfile streamlit/

# Build NGINX gateway image
echo "Building NGINX gateway image..."
docker build -t opt-rag-gateway:v1 -f nginx/Dockerfile nginx/

# Set the Docker registry to point to the local Minikube registry
# This is for local testing, so we use the local images
export DOCKER_REGISTRY=""

# Replace placeholders in Kubernetes YAML files
echo "Replacing placeholders in Kubernetes manifests..."
for file in kubernetes/minikube/*.yaml; do
  if [ -f "$file" ]; then
    echo "Processing $file..."
    sed "s|\${DOCKER_REGISTRY}/||g" $file > "${file}.temp"
    mv "${file}.temp" $file
  fi
done

# Apply Kubernetes manifests
echo "Creating namespace..."
kubectl apply -f kubernetes/minikube/namespace.yaml

echo "Creating persistent volumes..."
kubectl apply -f kubernetes/minikube/persistent-volumes.yaml

echo "Applying deployment..."
kubectl apply -f kubernetes/minikube/deployment.yaml

echo "Applying services..."
kubectl apply -f kubernetes/minikube/backend-service.yaml
kubectl apply -f kubernetes/minikube/frontend-service.yaml

echo "Deployment completed successfully!"
echo "To check the status of the pods, run: kubectl get pods -n opt-rag"
echo "To port-forward to the backend: kubectl port-forward -n opt-rag svc/backend-service 8000:8000"
echo "To port-forward to the frontend: kubectl port-forward -n opt-rag svc/frontend-service 8501:8501" 