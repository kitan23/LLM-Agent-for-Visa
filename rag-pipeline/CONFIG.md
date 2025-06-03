# OPT-RAG Configuration Guide

## Overview

Your OPT-RAG system now supports **two modes of operation**:

1. **API-based LLM Mode** (Recommended for cloud deployment)
2. **Local Model Mode** (Original implementation)

## Quick Start

### 🌐 API Mode (Recommended)

```bash
# Set environment variables
export USE_API_LLM=true
export LLM_API_PROVIDER=openai
export LLM_API_KEY=your-openai-api-key-here
export LLM_API_MODEL=gpt-4o-mini
export VECTOR_STORE_PATH=/app/vector_store

# Start the service
python run_api.py
```

### 🏠 Local Model Mode

```bash
# Set environment variables  
export USE_API_LLM=false
export MODEL_PATH=/app/models/qwen2.5-1.5b
export DEVICE=cpu
export VECTOR_STORE_PATH=/app/vector_store

# Start the service
python run_api.py
```

## Environment Variables

### Mode Selection

| Variable | Values | Description |
|----------|--------|-------------|
| `USE_API_LLM` | `true`/`false` | Choose API mode (`true`) or local mode (`false`, default) |

### API Mode Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_PROVIDER` | `openai` | API provider (`openai`) |
| `LLM_API_KEY` | Required | Your API key |
| `LLM_API_MODEL` | `gpt-4o-mini` | Model to use |
| `LLM_API_BASE_URL` | Optional | Custom API endpoint |

### Local Mode Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/models/qwen2.5-1.5b` | Path to local model |
| `DEVICE` | `cpu` | Device (`cpu`, `cuda`, `mps`, `auto`) |

### Common Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_STORE_PATH` | `/app/vector_store` | Path to vector database |
| `OTLP_ENDPOINT` | `http://jaeger:4317` | Tracing endpoint |

## Kubernetes Deployment

### API Mode (Recommended for Cloud)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-service
spec:
  template:
    spec:
      containers:
      - name: backend
        image: your-image:latest
        env:
        - name: USE_API_LLM
          value: "true"
        - name: LLM_API_PROVIDER
          value: "openai"
        - name: LLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-api-secret
              key: api-key
        - name: LLM_API_MODEL
          value: "gpt-4o-mini"
        - name: VECTOR_STORE_PATH
          value: "/app/vector_store"
        resources:
          requests:
            cpu: "100m"
            memory: "256Mi"
          limits:
            cpu: "500m"
            memory: "1Gi"
        volumeMounts:
        - name: vector-store-volume
          mountPath: /app/vector_store
```

### Local Mode (High Memory Requirements)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-service
spec:
  template:
    spec:
      containers:
      - name: backend
        image: your-image:latest
        env:
        - name: USE_API_LLM
          value: "false"
        - name: MODEL_PATH
          value: "/app/models/qwen2.5-1.5b"
        - name: DEVICE
          value: "cpu"
        - name: VECTOR_STORE_PATH
          value: "/app/vector_store"
        resources:
          requests:
            cpu: "1000m"
            memory: "8Gi"
          limits:
            cpu: "2000m"
            memory: "16Gi"
        volumeMounts:
        - name: models-volume
          mountPath: /app/models
        - name: vector-store-volume
          mountPath: /app/vector_store
```

## Mode Comparison

| Feature | API Mode | Local Mode |
|---------|----------|------------|
| **Memory Requirements** | Low (256MB - 1GB) | High (8GB - 16GB) |
| **Startup Time** | Fast (< 10s) | Slow (1-5 min) |
| **Model Quality** | Latest GPT-4/Claude | Qwen2.5-1.5b |
| **Cost** | Pay per request | One-time setup |
| **Privacy** | Data sent to API | Fully private |
| **Scalability** | Unlimited | Limited by hardware |
| **Offline Operation** | No | Yes |

## API Keys Setup

### OpenAI API Key

1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create a new secret key
3. Set the environment variable:
   ```bash
   export LLM_API_KEY=sk-your-key-here
   ```

### Kubernetes Secret

```bash
kubectl create secret generic llm-api-secret \
  --from-literal=api-key=sk-your-key-here \
  -n opt-rag
```

## Testing Both Modes

### Test API Mode
```bash
export USE_API_LLM=true
export LLM_API_KEY=your-key
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is OPT?"}'
```

### Test Local Mode
```bash
export USE_API_LLM=false
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is OPT?"}'
```

## Switching Between Modes

You can switch between modes **without code changes** by just changing the `USE_API_LLM` environment variable and restarting the service.

## Troubleshooting

### API Mode Issues
- **Error: API key not found**: Set `LLM_API_KEY` environment variable
- **Error: Rate limit**: Upgrade your OpenAI plan or use different model
- **Error: Model not found**: Check `LLM_API_MODEL` value

### Local Mode Issues
- **Error: OOMKilled**: Increase memory limits to 16Gi+
- **Error: Model not found**: Ensure model files are in `MODEL_PATH`
- **Error: CUDA out of memory**: Switch to CPU mode (`DEVICE=cpu`)

## Cost Optimization

### API Mode
- Use `gpt-4o-mini` for cost efficiency
- Implement caching for repeated queries
- Monitor usage on OpenAI dashboard

### Local Mode
- Use CPU for cost savings (slower)
- Use GPU nodes only when needed
- Consider smaller models for development 