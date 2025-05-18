# ☁️ OPT-RAG Cloud Deployment Guide (GKE & Helm-Ready)

## Overview

This guide will help you deploy the OPT-RAG stack (FastAPI backend + LLM, Streamlit frontend, NGINX gateway, and monitoring) on **Google Kubernetes Engine (GKE)** or locally with Minikube.  
It covers:
- Building and pushing images
- Persistent storage for models/vector store
- Efficient resource allocation
- Load balancing and ingress
- Monitoring (Prometheus, Grafana, Jaeger)
- Using Helm for easier upgrades and config
- **Using Google Artifact Registry (recommended, replaces Container Registry)**

---

## 1. Prerequisites

- **Google Cloud SDK** (`gcloud`)
- **kubectl**
- **Helm 3+**
- **Docker**
- GCP project with GKE and Artifact Registry enabled

---

## 2. Set Up Google Artifact Registry

**Create a Docker repository in Artifact Registry:**
```bash
gcloud artifacts repositories create opt-rag-docker-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for OPT-RAG images"
```

**Configure Docker to authenticate with Artifact Registry:**
```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

---

## 3. Build & Push Docker Images (Artifact Registry)

**Build and push images:**
```bash
# Set variables
PROJECT_ID=$(gcloud config get-value project)
REGION=us-central1
REPO=opt-rag-docker-repo

# Backend
docker build --platform linux/amd64 -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/opt-rag-backend:v1 -f rag-pipeline/Dockerfile rag-pipeline/
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/opt-rag-backend:v1

# Frontend
docker build --platform linux/amd64 -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/opt-rag-frontend:v1 -f streamlit/Dockerfile streamlit/
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/opt-rag-frontend:v1

# NGINX Gateway
docker build --platform linux/amd64 -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/opt-rag-gateway:v1 -f nginx/Dockerfile nginx/
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO/opt-rag-gateway:v1
```

> **Important:** The `--platform linux/amd64` flag ensures compatibility with GKE nodes, which typically run on x86_64 architecture. This is especially important if you're building on Apple Silicon (M1/M2/M3) Macs or other ARM-based systems.

---

## 4. Create GKE Cluster

```bash
gcloud container clusters create opt-rag-cluster \
  --zone=us-central1-a \
  --num-nodes=1 \
  --machine-type=e2-standard-2 \
  --disk-type=pd-standard \
  --disk-size=50 \
  --enable-ip-alias
```

---

## 5. Deploy with Helm (Recommended)

**Navigate to the Helm chart:**
```bash
cd kubernetes/helm-chart
```

**Create namespace:**
```bash
kubectl create namespace opt-rag
```

**Install the chart:**
```bash
helm upgrade -i opt-rag . \
    --namespace opt-rag \
    --set images.registry=us-central1-docker.pkg.dev/$PROJECT_ID/$REPO/ \
    --set services.backend.type=LoadBalancer \
    --set services.frontend.type=LoadBalancer \
    --set services.nginx.type=LoadBalancer
```

**(Optional) Upgrade config:**
- Edit `values.yaml` for resource requests, PVC sizes, or image tags.
- Upgrade with:  
  `helm upgrade opt-rag . -n opt-rag`

---

## 6. Persistent Storage

- **Models and vector store** are mounted as PVCs.
- Default sizes: 5Gi (models), 1Gi (vector store).  
  Adjust in `values.yaml` or the manifest if needed.
- GKE will provision persistent disks automatically.

---

## 6a. Populating Your Model and Vector Store PVCs

Your backend expects the model files in `/app/models` and the vector store in `/app/vector_store` inside the pod. In your repo, these are:
- **Model files:** `rag-pipeline/models/`
- **Vector store files:** `rag-pipeline/vector_store/`

You must copy these files into the PVCs before your backend will work. Here are two recommended methods:

### **Option 1: Copy from Google Cloud Storage (GCS) to PVC (Recommended for large files or remote work)**

1. **Upload your files to a GCS bucket:**
   ```bash
   gsutil cp rag-pipeline/models/* gs://YOUR_BUCKET/models/
   gsutil cp rag-pipeline/vector_store/* gs://YOUR_BUCKET/vector_store/
   ```

2. **Create a copy pod to transfer from GCS to the PVCs:**
   Save as `model-copy.yaml`:
   ```yaml
   apiVersion: v1
   kind: Pod
   metadata:
     name: model-copy
     namespace: opt-rag
   spec:
     restartPolicy: Never
     containers:
     - name: copy
       image: google/cloud-sdk:slim
       command: ["/bin/sh", "-c"]
       args:
         - |
           gsutil -m cp -r gs://YOUR_BUCKET/models/* /models/ && \
           gsutil -m cp -r gs://YOUR_BUCKET/vector_store/* /vector_store/
       volumeMounts:
       - name: models
         mountPath: /models
       - name: vector-store
         mountPath: /vector_store
     volumes:
     - name: models
       persistentVolumeClaim:
         claimName: models-pvc
     - name: vector-store
       persistentVolumeClaim:
         claimName: vector-store-pvc
   ```

   Apply and run:
   ```bash
   kubectl apply -f model-copy.yaml
   kubectl logs model-copy -n opt-rag  # Wait for completion
   kubectl delete pod model-copy -n opt-rag
   ```

### **Option 2: Copy Directly from Your Local Machine to the PVC (Quick for small/medium files)**

1. **Start a temporary pod that mounts the PVCs:**
   Save as `pvc-access.yaml`:
   ```yaml
   apiVersion: v1
   kind: Pod
   metadata:
     name: pvc-access
     namespace: opt-rag
   spec:
     containers:
     - name: shell
       image: ubuntu
       command: ["/bin/bash", "-c", "sleep 3600"]
       volumeMounts:
       - name: models
         mountPath: /models
       - name: vector-store
         mountPath: /vector_store
     volumes:
     - name: models
       persistentVolumeClaim:
         claimName: models-pvc
     - name: vector-store
       persistentVolumeClaim:
         claimName: vector-store-pvc
   ```
   Apply:
   ```bash
   kubectl apply -f pvc-access.yaml
   ```

2. **Copy files from your local machine:**
   ```bash
   kubectl cp rag-pipeline/models/. opt-rag/pvc-access:/models
   kubectl cp rag-pipeline/vector_store/. opt-rag/pvc-access:/vector_store
   ```

3. **Delete the pod after copying:**
   ```bash
   kubectl delete pod pvc-access -n opt-rag
   ```

---

## 7. Accessing the Application

**Get external IPs:**
```bash
kubectl get svc -n opt-rag
```
- **Backend API:** `http://<backend-external-ip>:8000/docs`
- **Frontend UI:** `http://<frontend-external-ip>:8501`
- **NGINX Gateway:** `http://<nginx-external-ip>:80` (recommended entrypoint)

---

## 8. Ingress (Optional, for custom domains/TLS)

- Use the provided `ingress.yaml` or `simple-ingress.yaml` in `kubernetes/cloud/` for GKE Ingress.
- Edit annotations for timeouts and SSL as needed.
- Apply with:
  ```bash
  kubectl apply -f kubernetes/cloud/ingress.yaml -n opt-rag
  ```

---

## 9. Monitoring

- Deploy monitoring stack:
  ```bash
  kubectl apply -f kubernetes/cloud/monitoring.yaml -n opt-rag
  ```
- Port-forward to access dashboards:
  ```bash
  kubectl port-forward -n opt-rag svc/grafana 3000:3000
  kubectl port-forward -n opt-rag svc/prometheus 9090:9090
  kubectl port-forward -n opt-rag svc/jaeger 16686:16686
  ```

---

## 10. Troubleshooting & Best Practices

- **Resource Requests:**  
  For LLMs, ensure backend has enough RAM (3–6Gi for 1.5B models).  
  Adjust in `values.yaml` or manifest if OOM errors occur.
- **PVCs:**  
  GKE will auto-provision disks. If you see `Pending` PVCs, check your storage class and quotas.
- **Timeouts:**  
  LLM inference can be slow. NGINX and Ingress timeouts are increased in the provided configs.
- **Health Checks:**  
  Liveness/readiness probes are set for all services.
- **Scaling:**  
  Increase `replicaCount` in `values.yaml` for frontend/backend as needed.
- **Logs:**  
  Use `kubectl logs` to debug pods.

---

## 11. Cleanup

```bash
helm uninstall opt-rag -n opt-rag
kubectl delete namespace opt-rag
gcloud container clusters delete opt-rag-cluster --zone=us-central1-a
```

---

## 12. Advanced: Manual YAML Deployment

If you prefer not to use Helm, you can apply the manifests in `kubernetes/cloud/` directly:
```bash
kubectl create namespace opt-rag
kubectl apply -f kubernetes/cloud/backend.yaml
kubectl apply -f kubernetes/cloud/frontend.yaml
kubectl apply -f kubernetes/cloud/nginx-gateway-fixed.yaml
kubectl apply -f kubernetes/cloud/nginx-configmap.yaml
kubectl apply -f kubernetes/cloud/monitoring.yaml
kubectl apply -f kubernetes/cloud/ingress.yaml
```
**Edit image names and PVCs as needed.**

---

## 13. What is Artifact Registry? What is a Model Registry?

**Artifact Registry** is Google Cloud's modern, universal package manager for storing and managing container images (Docker), language packages, and other build artifacts. It replaces the older Container Registry (`gcr.io`).
- **Use Artifact Registry** to store and pull your Docker images for Kubernetes deployments.
- Images are referenced as: `REGION-docker.pkg.dev/PROJECT-ID/REPO/IMAGE:TAG`

**Model Registry** (not used directly in this guide) is a specialized system for storing, versioning, and managing machine learning models (e.g., Vertex AI Model Registry, MLflow Model Registry). It is for tracking ML models, not general Docker images or code artifacts.

**Summary:**
- **Artifact Registry:** For Docker images, Python/Java packages, etc. (used in this guide)
- **Model Registry:** For ML model binaries, metadata, and versioning (not required for this deployment)

---

## 14. What is Helm and When Should I Use It?

**Helm** is the package manager for Kubernetes. It lets you:
- Template and reuse Kubernetes manifests (like Docker Compose for K8s)
- Upgrade, rollback, and manage releases easily
- Parameterize deployments (e.g., change image tags, resource sizes, replica counts)
- Avoid copy-pasting YAML for every environment

**Use Helm if:**
- You want to easily upgrade or rollback deployments
- You want to deploy to multiple environments (dev, staging, prod)
- You want to keep your manifests DRY and maintainable

**If you use the raw YAML in `cloud/`,** you must manually edit image names, resource requests, and PVC sizes as needed. Helm makes this much easier and less error-prone.

---

## 15. Do I Need to Change Anything in `@cloud`?

- **If using Helm:** No changes needed in `cloud/`—use the Helm chart and edit `values.yaml` for config.
- **If using raw YAML:**
  - Update `image:` fields to match your pushed images (e.g., `us-central1-docker.pkg.dev/YOUR_PROJECT_ID/opt-rag-docker-repo/opt-rag-backend:v1`)
  - Adjust `resources:` (CPU/memory) for your backend if you get OOM errors
  - Adjust PVC sizes if you need more space for models or vector store
  - Check that `storageClassName` matches your GKE setup (usually `standard`)

---

**Need a custom `values.yaml` or want to automate CI/CD? Let me know!**
This guide is now tailored to your codebase and GKE best practices.
