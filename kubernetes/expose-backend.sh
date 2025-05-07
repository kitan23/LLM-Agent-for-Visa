#!/bin/bash

# Exit on any error
set -e

# Colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
NAMESPACE="opt-rag"

echo -e "${BLUE}=== OPT-RAG Backend Exposure Script ===${NC}"
echo "This script will change the backend service type to LoadBalancer"
echo "to make it accessible from outside the Kubernetes cluster."

# Check for kubectl
command -v kubectl >/dev/null 2>&1 || { 
  echo -e "${RED}Error: kubectl not found. Please install kubectl.${NC}"; 
  exit 1; 
}

# Check if kubectl is configured to the right context
echo "Checking kubectl connection..."
CURRENT_CONTEXT=$(kubectl config current-context)
echo "Current Kubernetes context: $CURRENT_CONTEXT"
read -p "Is this the correct context for your cloud deployment? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Please set the correct kubectl context and try again."
  exit 1
fi

# Apply the backend service changes
echo -e "${BLUE}Applying backend service changes...${NC}"
kubectl apply -f kubernetes/cloud/backend.yaml

# Wait for LoadBalancer to get external IP
echo -e "${BLUE}Waiting for LoadBalancer to be provisioned...${NC}"
echo "This might take a minute or two..."

# Loop until external IP is assigned
for i in {1..30}; do
  BACKEND_IP=$(kubectl get svc -n $NAMESPACE backend -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
  if [ -n "$BACKEND_IP" ]; then
    echo -e "${GREEN}Backend service is now exposed at: http://${BACKEND_IP}:8000${NC}"
    break
  fi
  echo -n "."
  sleep 5
done

if [ -z "$BACKEND_IP" ]; then
  echo -e "${RED}\nTimeout waiting for external IP.${NC}"
  echo "Check service status with: kubectl get svc -n $NAMESPACE"
  exit 1
fi

echo
echo -e "${GREEN}Setup completed successfully!${NC}"
echo "To run the frontend locally with cloud backend:"
echo "./streamlit/run-local-with-cloud-backend.sh"
echo
echo "Test the backend API directly with:"
echo "curl http://${BACKEND_IP}:8000/health" 