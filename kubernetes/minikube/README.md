# OPT-RAG Minikube Deployment Guide

This guide explains how to deploy the OPT-RAG application on a local Kubernetes cluster using Minikube.

## Prerequisites

Before you begin, make sure you have the following tools installed:

- [Docker](https://docs.docker.com/get-docker/)
- [Minikube](https://minikube.sigs.k8s.io/docs/start/)
- [kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl)
- [Helm](https://helm.sh/docs/intro/install/) (optional, for Helm deployment)

## Option 1: Direct Deployment with kubectl

### 1. Start Minikube

Start Minikube with sufficient resources:

```bash
minikube start --cpus=4 --memory=7168
```

### 2. Deploy to Minikube

Use the provided script to deploy the application:

```bash
./deploy-to-minikube.sh
```

This script will:
1. Set the Minikube Docker environment
2. Build Docker images within Minikube
3. Create necessary Kubernetes resources (namespace, PVCs, deployment, services)

### 3. Access the Application

Once deployed, you can access the application by port-forwarding:

```bash
# Backend API
kubectl port-forward -n opt-rag svc/backend-service 8000:8000

# Frontend UI
kubectl port-forward -n opt-rag svc/frontend-service 8501:8501
```

Then open your browser and navigate to:
- Backend API: http://localhost:8000/docs
- Frontend UI: http://localhost:8501

## Option 2: Helm Chart Deployment

For a more automated deployment with Helm:

```bash
../helm-deploy.sh -e minikube
```

## Cleaning Up

To clean up the resources:

```bash
# Delete the namespace (this will delete all resources in the namespace)
kubectl delete namespace opt-rag

# Stop Minikube
minikube stop

# Optional: Delete the Minikube cluster
minikube delete
```

## Troubleshooting

### Check Pod Status

```bash
kubectl get pods -n opt-rag
kubectl describe pod -n opt-rag <pod-name>
```

### View Logs

```bash
kubectl logs -n opt-rag <pod-name> -c <container-name>
```

Where:
- `<pod-name>` is the name of the pod (get it from `kubectl get pods -n opt-rag`)
- `<container-name>` is either "backend" or "frontend"

### Common Issues

1. **Out of Memory**: If pods are terminated due to OOM issues, increase the memory allocated to Minikube:
   ```bash
   minikube stop
   minikube start --cpus=4 --memory=10240
   ```

2. **Image Pull Errors**: Make sure you're building images within Minikube's Docker environment:
   ```bash
   eval $(minikube docker-env)
   ```

3. **Persistent Volume Issues**: If PVCs are stuck in "Pending" state, check the storage class:
   ```bash
   kubectl get sc
   ``` 