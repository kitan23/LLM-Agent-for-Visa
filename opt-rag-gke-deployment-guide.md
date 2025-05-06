# ☁️ Cloud Deployment Using Kubernetes (GKE Ready)

## 📌 Overview

This guide explains how to deploy the **OPT-RAG** project to a local Kubernetes cluster for development and testing, and then to **Google Kubernetes Engine (GKE)** for production. The deployment includes:

* ✅ Containerizing backend (FastAPI), frontend (Streamlit), and API Gateway (NGINX)
* ✅ Local deployment using **Minikube** for testing
* ✅ Cloud deployment using **GKE** with autoscaling, persistent storage, and load balancers
* ✅ Monitoring via Prometheus, Grafana, and Jaeger
* ✅ CI/CD enabled via Jenkins

---

## 📁 Directory Context

Your project structure is modular and well-suited for container orchestration:

* **rag-pipeline/** – FastAPI backend + vector store + LLM
* **streamlit/** – UI frontend
* **nginx/** – API gateway for routing
* **grafana/**, **prometheus/**, **jaeger/** – Monitoring
* **kubernetes/** – Organized Kubernetes manifests for `local/` and `cloud/`

---

## 1. 📦 Build & Push Docker Images

First, log in to Docker Hub:

```bash
docker login
```

Then build and push each component:

```bash
# Backend
cd rag-pipeline
docker build -t your-dockerhub/opt-rag-backend:v1 .
docker push your-dockerhub/opt-rag-backend:v1

# Frontend
cd ../streamlit
docker build -t your-dockerhub/opt-rag-frontend:v1 .
docker push your-dockerhub/opt-rag-frontend:v1

# NGINX Gateway
cd ../nginx
docker build -t your-dockerhub/opt-rag-gateway:v1 .
docker push your-dockerhub/opt-rag-gateway:v1
```

Optionally push monitoring components:

```bash
cd ../grafana
... (repeat for prometheus and jaeger)
```

---

## 2. 🧪 Local Deployment with Minikube

### 📥 Prerequisites

Install:

* [kubectl](https://kubernetes.io/docs/tasks/tools/)
* [minikube](https://minikube.sigs.k8s.io/docs/start/)
* [helm](https://helm.sh/docs/intro/install/)

### 🏗️ Start Cluster

```bash
minikube start --cpus=4 --memory=7168
```

### 🛠️ Apply Manifests

```bash
kubectl create namespace opt-rag
cd kubernetes/local
kubectl apply -f backend.yaml
kubectl apply -f frontend.yaml
kubectl apply -f ingress.yaml
kubectl apply -f monitoring.yaml
```

### 🔍 Access Services

Port forward services:

```bash
kubectl port-forward -n opt-rag svc/backend 8000:8000
kubectl port-forward -n opt-rag svc/frontend 8501:8501
```

---

## 3. ☁️ Deploy on Google Cloud (GKE)

### 📌 Prerequisites

Install and set up:

* [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
* Enable GKE + Artifact Registry

### 🧱 Create GKE Cluster

```bash
gcloud container clusters create opt-rag-cluster \
  --num-nodes=3 \
  --zone=us-central1-a \
  --enable-ip-alias
```

### 📤 Push Images to Artifact Registry (Optional)

Tag and push images:

```bash
docker tag opt-rag-backend gcr.io/your-project-id/opt-rag-backend:v1
docker push gcr.io/your-project-id/opt-rag-backend:v1
```

### 🛠️ Deploy to GKE

```bash
cd kubernetes/cloud
kubectl create namespace opt-rag
kubectl apply -f backend.yaml
kubectl apply -f frontend.yaml
kubectl apply -f ingress.yaml
kubectl apply -f monitoring.yaml
```

### 🌐 Access via External IP

```bash
kubectl get svc -n opt-rag
```

Use `EXTERNAL-IP` from LoadBalancer to access:

* Backend: `http://EXTERNAL-IP:8000/docs`
* Frontend: `http://EXTERNAL-IP:8501`
* Full App via Ingress or NGINX: `http://EXTERNAL-IP`

---

## 4. 📊 Monitoring Setup

* **Prometheus** collects metrics from FastAPI
* **Grafana** visualizes dashboards
* **Jaeger** traces document upload and query latency

After deployment:

```bash
kubectl port-forward -n opt-rag svc/prometheus 9090:9090
kubectl port-forward -n opt-rag svc/grafana 3000:3000
kubectl port-forward -n opt-rag svc/jaeger 16686:16686
```

---

## 🧹 Cleanup

```bash
kubectl delete namespace opt-rag
minikube stop && minikube delete
```

---

## 🧠 Notes

* Use GPU-enabled node pool if hosting large models (e.g., LLaMA, Mistral)
* Attach PersistentVolumeClaim for FAISS index and uploaded PDFs
* Enable auto-scaling for production workloads
* Set `type: LoadBalancer` in `cloud/*service.yaml` for external access

---

Let me know if you want the actual Kubernetes or Helm files generated next.
