#!/bin/bash

# Exit on any error
set -e

# Default values
ENVIRONMENT="minikube"  # minikube or gcp
NAMESPACE="opt-rag"
RELEASE_NAME="opt-rag"

# Display help message
function show_help {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -e, --environment    Target environment (minikube or gcp, default: minikube)"
  echo "  -n, --namespace      Kubernetes namespace (default: opt-rag)"
  echo "  -r, --release        Helm release name (default: opt-rag)"
  echo "  -h, --help           Show this help message"
  echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -e|--environment)
      ENVIRONMENT="$2"
      shift
      shift
      ;;
    -n|--namespace)
      NAMESPACE="$2"
      shift
      shift
      ;;
    -r|--release)
      RELEASE_NAME="$2"
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

# Validate environment
if [[ "$ENVIRONMENT" != "minikube" && "$ENVIRONMENT" != "gcp" ]]; then
  echo "Error: Environment must be either 'minikube' or 'gcp'"
  show_help
  exit 1
fi

echo "=== OPT-RAG Helm Deployment ==="
echo "Environment: $ENVIRONMENT"
echo "Namespace:   $NAMESPACE"
echo "Release:     $RELEASE_NAME"
echo "================================"

# Ensure the namespace exists
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# If deploying to Minikube, set Docker environment
if [[ "$ENVIRONMENT" == "minikube" ]]; then
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
  
  # Deploy with Helm
  echo "Deploying to Minikube with Helm..."
  helm upgrade --install $RELEASE_NAME kubernetes/helm-chart \
    --namespace $NAMESPACE \
    --set images.registry="" \
    --set services.backend.type=NodePort \
    --set services.frontend.type=NodePort \
    --set services.nginx.type=NodePort
    
elif [[ "$ENVIRONMENT" == "gcp" ]]; then
  # For GCP deployment, we need to specify the GCP project ID
  read -p "Enter your GCP project ID: " PROJECT_ID
  
  if [[ -z "$PROJECT_ID" ]]; then
    echo "Error: GCP project ID is required for GCP deployment"
    exit 1
  fi
  
  REGISTRY="gcr.io/$PROJECT_ID/"
  
  # Configure Docker for GCR
  echo "Configuring Docker to use GCP credentials..."
  gcloud auth configure-docker
  
  echo "Building and pushing Docker images to GCR..."
  
  # Build and push backend image
  echo "Building and pushing backend image..."
  docker build -t ${REGISTRY}opt-rag-backend:v1 -f rag-pipeline/Dockerfile rag-pipeline/
  docker push ${REGISTRY}opt-rag-backend:v1
  
  # Build and push frontend image
  echo "Building and pushing frontend image..."
  docker build -t ${REGISTRY}opt-rag-frontend:v1 -f streamlit/Dockerfile streamlit/
  docker push ${REGISTRY}opt-rag-frontend:v1
  
  # Build and push NGINX gateway image
  echo "Building and pushing NGINX gateway image..."
  docker build -t ${REGISTRY}opt-rag-gateway:v1 -f nginx/Dockerfile nginx/
  docker push ${REGISTRY}opt-rag-gateway:v1
  
  # Deploy with Helm
  echo "Deploying to GCP with Helm..."
  helm upgrade --install $RELEASE_NAME kubernetes/helm-chart \
    --namespace $NAMESPACE \
    --set images.registry=$REGISTRY \
    --set services.backend.type=LoadBalancer \
    --set services.frontend.type=LoadBalancer \
    --set services.nginx.type=LoadBalancer
fi

echo "Deployment completed successfully!"
echo "To check the status of the pods, run: kubectl get pods -n $NAMESPACE"

if [[ "$ENVIRONMENT" == "minikube" ]]; then
  echo "To access the services on Minikube:"
  echo "Backend API: kubectl port-forward -n $NAMESPACE svc/backend-service 8000:8000"
  echo "Frontend UI: kubectl port-forward -n $NAMESPACE svc/frontend-service 8501:8501"
  echo "NGINX Gateway: kubectl port-forward -n $NAMESPACE svc/nginx-service 8080:80"
elif [[ "$ENVIRONMENT" == "gcp" ]]; then
  echo "To get the external IPs on GCP:"
  echo "kubectl get svc -n $NAMESPACE"
fi 