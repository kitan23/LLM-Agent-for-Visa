# Testing Streamlit Frontend Locally with Cloud Backend

This guide explains how to test your local Streamlit frontend with the OPT-RAG backend deployed in Kubernetes (GKE).

## Prerequisites

- Working Kubernetes deployment of the OPT-RAG backend
- `kubectl` configured with access to your cluster
- Python and Streamlit installed locally

## Setup Instructions

### 1. Expose the Backend Service

The backend service needs to be exposed with a LoadBalancer to make it accessible from outside the Kubernetes cluster.

```bash
# Run the expose-backend.sh script
./kubernetes/expose-backend.sh
```

This script will:
- Change the backend service type from ClusterIP to LoadBalancer
- Wait for an external IP to be assigned
- Display the external IP and port for accessing the backend

### 2. Run the Local Frontend

Once you have the backend exposed, you can run the local frontend pointing to the cloud backend:

```bash
# Run the local frontend script
./streamlit/run-local-with-cloud-backend.sh
```

This script will:
- Retrieve the backend service IP from Kubernetes
- Set the API_URL environment variable accordingly
- Install any required Python dependencies
- Start the Streamlit application

## Configuring Timeouts

If you encounter timeout issues, you can adjust the timeout settings using environment variables:

```bash
# Set longer timeouts for health checks and API queries
export HEALTH_CHECK_TIMEOUT=60  # seconds
export QUERY_TIMEOUT=180  # seconds

# Then run the frontend
streamlit run streamlit/app.py
```

## Troubleshooting

### Connection Issues

If you can't connect to the backend:

1. **Verify External IP**:
   ```bash
   kubectl get svc -n opt-rag backend
   ```

2. **Check Firewall Rules**:
   - GKE automatically creates firewall rules for LoadBalancer services
   - Check GCP Console > VPC Network > Firewall for rules allowing port 8000

3. **Test Direct Connection**:
   ```bash
   curl -v http://<BACKEND_IP>:8000/health
   ```

4. **Check Backend Logs**:
   ```bash
   kubectl logs -n opt-rag -l component=backend
   ```

### Performance Issues

If the backend responds slowly:

1. **Increase Resource Limits**:
   - Edit `kubernetes/cloud/backend.yaml` to increase CPU and memory
   - Apply changes with `kubectl apply -f kubernetes/cloud/backend.yaml`

2. **Verify Backend Pod Status**:
   ```bash
   kubectl describe pod -n opt-rag -l component=backend
   ```

3. **Check Backend Metrics**:
   ```bash
   kubectl top pod -n opt-rag
   ```

## Next Steps

After verifying that the frontend works correctly with the cloud backend, you can:

1. Deploy the updated frontend to Kubernetes
2. Run performance tests with more users
3. Monitor the backend performance in production 