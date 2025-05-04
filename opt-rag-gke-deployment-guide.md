# 📦 OPT-RAG Deployment Guide on Google Kubernetes Engine (GKE)

This guide outlines step-by-step instructions to deploy the OPT-RAG backend (FastAPI + Qwen2.5 LLM + FAISS vector store) to Google Kubernetes Engine (GKE).

---

## 🚧 Prerequisites

1. **Install Required Tools**
   ```bash
   gcloud auth login
   gcloud config set project <your-gcp-project-id>
   gcloud components install kubectl
   gcloud components install beta
   ```

2. **Enable Required APIs**
   ```bash
   gcloud services enable container.googleapis.com
   gcloud services enable artifactregistry.googleapis.com
   ```

3. **Install Docker & Kubernetes**
   - [Docker Desktop](https://www.docker.com/products/docker-desktop)
   - [kubectl CLI](https://kubernetes.io/docs/tasks/tools/)

---

## 🐳 Step 1: Build and Push Docker Image to Google Artifact Registry

1. **Create Artifact Registry**
   ```bash
   gcloud artifacts repositories create opt-rag-backend        --repository-format=docker        --location=us-central1        --description="OPT-RAG backend container"
   ```

2. **Build Docker Image**
   ```bash
   docker build -t us-central1-docker.pkg.dev/<your-project-id>/opt-rag-backend/opt-rag:latest .
   ```

3. **Push Docker Image**
   ```bash
   docker push us-central1-docker.pkg.dev/<your-project-id>/opt-rag-backend/opt-rag:latest
   ```

---

## ☸️ Step 2: Create GKE Cluster

```bash
gcloud container clusters create opt-rag-cluster     --zone=us-central1-a     --num-nodes=3     --enable-autoscaling --min-nodes=1 --max-nodes=5     --enable-ip-alias
```

```bash
gcloud container clusters get-credentials opt-rag-cluster --zone us-central1-a
```

---

## 📄 Step 3: Deploy Kubernetes Manifests

### 1. `deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opt-rag
spec:
  replicas: 1
  selector:
    matchLabels:
      app: opt-rag
  template:
    metadata:
      labels:
        app: opt-rag
    spec:
      containers:
      - name: opt-rag-container
        image: us-central1-docker.pkg.dev/<your-project-id>/opt-rag-backend/opt-rag:latest
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: faiss-data
          mountPath: /app/vector_store
      volumes:
      - name: faiss-data
        persistentVolumeClaim:
          claimName: faiss-pvc
```

### 2. `service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: opt-rag-service
spec:
  type: LoadBalancer
  selector:
    app: opt-rag
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
```

### 3. `pvc.yaml`

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: faiss-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

---

## 🚀 Deploy to Kubernetes

```bash
kubectl apply -f pvc.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

---

## 🌍 Access Your Backend

```bash
kubectl get service opt-rag-service
```

Copy the external IP, and access your FastAPI docs at:
```
http://<EXTERNAL_IP>/docs
```

---

## ✅ Optional Add-ons

- Use `GPU` nodes: Add `--accelerator="type=nvidia-tesla-t4,count=1"` to `gcloud container clusters create`
- Set up monitoring: [Grafana + Prometheus + Jaeger](https://github.com/hieunq95/tiny-llm-agent)
- Add HTTPS & domain: Use [Ingress + Google-managed certificate](https://cloud.google.com/kubernetes-engine/docs/how-to/managed-certs)

---