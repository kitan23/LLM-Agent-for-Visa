#!/bin/bash

# Exit on any error
set -e

# Default values
PROJECT_ID=""
CLUSTER_NAME="opt-rag-cluster"
ZONE="us-central1-a"
REGISTRY="gcr.io"
VERSION="v1"
NAMESPACE="opt-rag"

# Display help message
function show_help {
  echo "Usage: $0 -p [project-id] [options]"
  echo ""
  echo "Options:"
  echo "  -p, --project-id    GCP project ID (required)"
  echo "  -c, --cluster-name  Name of GKE cluster (default: opt-rag-cluster)"
  echo "  -z, --zone          GCP zone (default: us-central1-a)"
  echo "  -r, --registry      Container registry (default: gcr.io)"
  echo "  -v, --version       Version tag (default: v1)"
  echo "  -h, --help          Show this help message"
  echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -p|--project-id)
      PROJECT_ID="$2"
      shift
      shift
      ;;
    -c|--cluster-name)
      CLUSTER_NAME="$2"
      shift
      shift
      ;;
    -z|--zone)
      ZONE="$2"
      shift
      shift
      ;;
    -r|--registry)
      REGISTRY="$2"
      shift
      shift
      ;;
    -v|--version)
      VERSION="$2"
      shift
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Check if project ID is provided
if [ -z "$PROJECT_ID" ]; then
  echo "Error: GCP project ID is required"
  show_help
  exit 1
fi

# Set full registry path
DOCKER_REGISTRY="${REGISTRY}/${PROJECT_ID}"

echo "=== OPT-RAG GKE Deployment ==="
echo "Project ID:    $PROJECT_ID"
echo "Cluster Name:  $CLUSTER_NAME"
echo "Zone:          $ZONE"
echo "Registry:      $DOCKER_REGISTRY"
echo "Version:       $VERSION"
echo "=========================="

# Check for gcloud, docker and kubectl
command -v gcloud >/dev/null 2>&1 || { echo "Error: gcloud CLI not found. Please install Google Cloud SDK."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Error: docker not found. Please install Docker."; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "Error: kubectl not found. Please install kubectl."; exit 1; }

# Configure docker to use gcloud credentials
echo "Configuring Docker to use gcloud credentials..."
gcloud auth configure-docker

# Build and push images
echo "Building and pushing Docker images..."

# Backend
echo "Building backend image..."
docker build -t ${DOCKER_REGISTRY}/opt-rag-backend:${VERSION} -f rag-pipeline/Dockerfile rag-pipeline/
echo "Pushing backend image..."
docker push ${DOCKER_REGISTRY}/opt-rag-backend:${VERSION}

# Frontend
echo "Building frontend image..."
docker build -t ${DOCKER_REGISTRY}/opt-rag-frontend:${VERSION} -f streamlit/Dockerfile streamlit/
echo "Pushing frontend image..."
docker push ${DOCKER_REGISTRY}/opt-rag-frontend:${VERSION}

# NGINX Gateway
echo "Building NGINX gateway image..."
docker build -t ${DOCKER_REGISTRY}/opt-rag-gateway:${VERSION} -f nginx/Dockerfile nginx/
echo "Pushing NGINX gateway image..."
docker push ${DOCKER_REGISTRY}/opt-rag-gateway:${VERSION}

# Create or get GKE cluster
echo "Checking if GKE cluster $CLUSTER_NAME exists..."
if gcloud container clusters list --project $PROJECT_ID --filter="name=$CLUSTER_NAME" --format="value(name)" | grep -q "^${CLUSTER_NAME}$"; then
  echo "Cluster $CLUSTER_NAME already exists. Retrieving credentials..."
  gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE --project $PROJECT_ID
else
  echo "Creating GKE cluster $CLUSTER_NAME..."
  gcloud container clusters create $CLUSTER_NAME \
    --project $PROJECT_ID \
    --zone $ZONE \
    --num-nodes=3 \
    --machine-type=e2-standard-4 \
    --enable-autoscaling \
    --min-nodes=3 \
    --max-nodes=5
  
  # Get cluster credentials
  echo "Getting credentials for the new cluster..."
  gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE --project $PROJECT_ID
fi

# Create namespace if it doesn't exist
echo "Creating namespace $NAMESPACE if it doesn't exist..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Replace placeholders in Kubernetes YAML files
echo "Replacing placeholders in Kubernetes manifests..."
for file in kubernetes/cloud/*.yaml; do
  echo "Processing $file..."
  sed "s|\${DOCKER_REGISTRY}|${DOCKER_REGISTRY}|g" $file > "${file}.temp"
  mv "${file}.temp" $file
done

# Apply Kubernetes manifests
echo "Applying Kubernetes manifests..."
kubectl apply -f kubernetes/cloud/backend.yaml
kubectl apply -f kubernetes/cloud/frontend.yaml
kubectl apply -f kubernetes/cloud/monitoring.yaml
kubectl apply -f kubernetes/cloud/ingress.yaml

echo "Deployment completed successfully!"
echo "To check the status of the pods, run: kubectl get pods -n $NAMESPACE"
echo "To get the external IP for the services, run: kubectl get svc -n $NAMESPACE" 