# OPT-RAG Troubleshooting Guide

This document provides solutions for common issues when deploying and running the OPT-RAG application.

## Connection Issues

### Frontend Unable to Connect to Backend

**Symptom**: Frontend shows error: `Cannot connect to API: HTTPConnectionPool(host='backend', port=8000): Read timed out. (read timeout=5)`

**Solutions**:

1. **Increase timeout in health check**:
   - The default timeout in the frontend health check (5 seconds) may be too short
   - If using the local frontend, edit `streamlit/app.py` and increase the timeout in the health check:
     ```python
     # Change from
     response = requests.get(f"{API_URL}/health", timeout=5)
     # To
     response = requests.get(f"{API_URL}/health", timeout=30)
     ```

2. **Test backend accessibility**:
   - Ensure the backend service is reachable:
     ```bash
     # For in-cluster backend
     kubectl exec -it -n opt-rag $(kubectl get pod -n opt-rag -l component=frontend -o name | head -n 1) -- curl -v backend:8000/health
     
     # For external backend
     curl -v http://<BACKEND_IP>:8000/health
     ```

3. **Check backend resources**:
   - Backend may be resource-constrained:
     ```bash
     kubectl describe pod -n opt-rag -l component=backend
     kubectl top pod -n opt-rag -l component=backend
     ```

4. **Check backend logs**:
   ```bash
   kubectl logs -n opt-rag -l component=backend
   ```

## Performance Issues

### Slow Backend Response

**Symptom**: Backend takes too long to respond to queries

**Solutions**:

1. **Increase backend resources**:
   - Edit `kubernetes/cloud/backend.yaml` to increase resources:
     ```yaml
     resources:
       requests:
         cpu: "500m"  # Increase from 250m
         memory: "2Gi"  # Increase from 1.5Gi
       limits:
         cpu: "2"  # Increase from 1
         memory: "4Gi"  # Increase from 3Gi
     ```
   - Apply changes:
     ```bash
     kubectl apply -f kubernetes/cloud/backend.yaml
     ```

2. **Check model size and device**:
   - Consider using a smaller model or GPU acceleration:
     ```yaml
     env:
     - name: MODEL_PATH
       value: "/app/models/qwen2.5-0.5b"  # Use smaller model
     - name: DEVICE
       value: "cuda"  # Change from "cpu" to "cuda" if GPU is available
     ```

3. **Add liveness and readiness probes** to ensure Kubernetes can monitor the backend:
   ```yaml
   livenessProbe:
     httpGet:
       path: /health
       port: 8000
     initialDelaySeconds: 60
     periodSeconds: 30
     timeoutSeconds: 10
   readinessProbe:
     httpGet:
       path: /health
       port: 8000
     initialDelaySeconds: 30
     periodSeconds: 15
     timeoutSeconds: 10
   ```

## Testing Local Frontend with Cloud Backend

To test your local frontend with a cloud-deployed backend:

1. Expose the backend service:
   ```bash
   ./kubernetes/expose-backend.sh
   ```

2. Run the local frontend pointing to the cloud backend:
   ```bash
   ./streamlit/run-local-with-cloud-backend.sh
   ```

3. Make direct API calls to test:
   ```bash
   curl http://<BACKEND_IP>:8000/health
   curl -X POST http://<BACKEND_IP>:8000/api/query -H "Content-Type: application/json" -d '{"question":"Tell me about OPT visa"}'
   ```

## Networking Issues

### Ingress Not Working

**Symptom**: Unable to access application through Ingress

**Solutions**:

1. **Check Ingress status**:
   ```bash
   kubectl get ingress -n opt-rag
   kubectl describe ingress -n opt-rag
   ```

2. **Verify NGINX controller**:
   ```bash
   kubectl get pods -n ingress-nginx
   kubectl logs -n ingress-nginx -l app.kubernetes.io/component=controller
   ```

3. **Test services directly**:
   ```bash
   # Port forward to test locally
   kubectl port-forward -n opt-rag svc/backend 8000:8000
   kubectl port-forward -n opt-rag svc/frontend 8501:8501
   ```

## Storage Issues

### PVC Not Binding

**Symptom**: PVCs stay in Pending state

**Solutions**:

1. **Check PVC status**:
   ```bash
   kubectl get pvc -n opt-rag
   kubectl describe pvc -n opt-rag models-pvc
   ```

2. **Check storage class**:
   ```bash
   kubectl get storageclass
   ```

3. **Use cloud provider storage class**:
   - For GKE: `standard-rwo` or `premium-rwo`
   - For AKS: `managed-premium` or `default`
   - For EKS: `gp2` or `gp3` 