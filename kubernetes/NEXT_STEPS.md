# Next Steps for GKE Deployment

## What Has Been Set Up

1. ✅ Created Kubernetes manifests for GKE in `kubernetes/cloud/`:
   - Backend service, deployment, and persistent volumes
   - Frontend service and deployment
   - NGINX gateway and Ingress
   - Monitoring stack (Prometheus, Grafana, Jaeger)

2. ✅ Created a deployment script `kubernetes/deploy-to-gke.sh` to automate the deployment process

3. ✅ Added documentation:
   - Main Kubernetes README (`kubernetes/README.md`)
   - Google Cloud SDK installation guide (`kubernetes/gcloud-install-guide.md`)
   - Updated project README.md with cloud deployment information

## Next Steps

### 1. Complete Xcode Command Line Tools Installation

The Xcode Command Line Tools installation has been initiated. Please:
- Complete the installation by following the dialog prompts
- Once installation is complete, restart your terminal

### 2. Install Google Cloud SDK

```bash
brew install --cask google-cloud-sdk
```

After installation, verify it's working:

```bash
gcloud --version
```

### 3. Initialize Google Cloud SDK

```bash
gcloud init
```

This will guide you through:
- Logging in to your Google account
- Selecting a project
- Setting default compute region and zone

### 4. Install kubectl

```bash
gcloud components install kubectl
```

### 5. Create a GCP Project (if you don't have one)

You can create a project via the [Google Cloud Console](https://console.cloud.google.com/).

Alternatively, use the command line:

```bash
gcloud projects create YOUR_PROJECT_ID --name="OPT-RAG Project"
```

### 6. Enable Billing

Make sure billing is enabled for your project through the [Google Cloud Console](https://console.cloud.google.com/billing).

### 7. Enable Required APIs

```bash
gcloud services enable container.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 8. Run the Deployment Script

```bash
./kubernetes/deploy-to-gke.sh -p YOUR_PROJECT_ID
```

This script will:
- Build and push Docker images to Google Container Registry
- Create a GKE cluster if it doesn't exist
- Deploy the application components to the cluster

### 9. Access the Application

After deployment completes, get the external IP:

```bash
kubectl get svc -n opt-rag
```

You can access:
- Frontend UI: http://EXTERNAL-IP:8501
- API Gateway: http://EXTERNAL-IP

## Troubleshooting

If you encounter any issues:

1. **Docker image errors**: Check Docker is running and your GCP credentials are set up
   ```bash
   gcloud auth configure-docker
   ```

2. **GKE permissions**: Ensure your GCP account has sufficient permissions
   ```bash
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member=user:YOUR_EMAIL \
     --role=roles/container.admin
   ```

3. **Check Pod Status**:
   ```bash
   kubectl get pods -n opt-rag
   kubectl describe pod POD_NAME -n opt-rag
   ```

4. **View Logs**:
   ```bash
   kubectl logs POD_NAME -n opt-rag -c CONTAINER_NAME
   ```

5. **Resource constraints**: If pods are failing to start due to insufficient resources, adjust the resource requests and limits in the Kubernetes manifests.

For more information, refer to the [`kubernetes/README.md`](kubernetes/README.md) guide. 