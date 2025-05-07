# Setting up Local Models with OPT-RAG

This guide explains how to run the OPT-RAG application with local models while keeping other components in Kubernetes.

## Quick Start

1. Create a local directory to store models:
   ```bash
   sudo mkdir -p /opt/models
   sudo chmod 777 /opt/models
   ```

2. Choose one of the following options:

   **Option A: Use existing models (if you already have them)**
   ```bash
   ./kubernetes/setup-local-models.sh --use-existing
   ```
   
   **Option B: Download from Hugging Face (requires authentication):**
   
   a. Get a Hugging Face token (required for Qwen models):
   - Go to https://huggingface.co/ and sign up/login
   - Visit your profile -> Settings -> Access Tokens 
   - Create a new token with "read" access
   - Copy the token (starts with "hf_...")
   
   b. Pre-download the model:
   ```bash
   ./kubernetes/setup-local-models.sh --predownload --token "hf_your_token_here"
   ```

3. Deploy the application to Kubernetes:
   ```bash
   kubectl apply -f kubernetes/cloud/
   ```

## How It Works

- The backend pod will mount the host directory `/opt/models` into the container
- If the model doesn't exist in this directory, it will be automatically downloaded
- All inference happens locally on your machine rather than in the Kubernetes cluster
- The rest of the application (frontend, monitoring, etc.) runs in the cluster

## Using Different Models

To use a different model:

```bash
./kubernetes/setup-local-models.sh --model-id "your-preferred-model/name" --model-dir "custom-name" --predownload
```

## Minikube Users

If you're using Minikube:

```bash
./kubernetes/setup-local-models.sh --minikube --predownload
```

## Troubleshooting

- Check pod logs: `kubectl logs -n opt-rag deployment/backend`
- Verify model directory permissions: `ls -la /opt/models`
- For permission issues: `sudo chmod -R 777 /opt/models` 