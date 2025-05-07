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

echo -e "${BLUE}=== OPT-RAG Local Frontend with Cloud Backend ===${NC}"
echo "This script will help you run the Streamlit frontend locally"
echo "while connecting to the backend deployed in the cloud."

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

# Get external IP of the backend service
echo -e "${BLUE}Getting backend service IP...${NC}"
BACKEND_IP=$(kubectl get svc -n $NAMESPACE backend -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

if [ -z "$BACKEND_IP" ]; then
  echo -e "${RED}Backend service doesn't have an external IP yet.${NC}"
  echo "You need to change the service type to LoadBalancer first."
  echo "Apply the changes with: kubectl apply -f kubernetes/cloud/backend.yaml"
  exit 1
fi

# Set API_URL environment variable to point to cloud backend
export API_URL="http://${BACKEND_IP}:8000"
echo -e "${GREEN}Backend service found at: ${API_URL}${NC}"
echo "Setting API_URL environment variable accordingly."

# Run the Streamlit app
echo -e "${BLUE}Running Streamlit app...${NC}"
echo "Starting Streamlit frontend pointing to cloud backend at: ${API_URL}"
echo "Press Ctrl+C to stop the application"

cd streamlit
if [ -f "requirements.txt" ]; then
  echo "Installing Python dependencies..."
  pip install -r requirements.txt
fi

echo -e "${GREEN}Starting Streamlit...${NC}"
streamlit run app.py 