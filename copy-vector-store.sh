#!/bin/bash

NAMESPACE="opt-rag"
PVC_NAME="vector-store-pvc"
POD_NAME="vectorstore-copier"
LOCAL_PATH="./rag-pipeline/vector_store/"

echo "==> Step 1: Create a temporary pod to mount the PVC..."
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
  namespace: ${NAMESPACE}
spec:
  containers:
  - name: busybox
    image: busybox
    command: ["sleep", "3600"]
    volumeMounts:
    - name: vector-store-volume
      mountPath: /vector_store
  volumes:
  - name: vector-store-volume
    persistentVolumeClaim:
      claimName: ${PVC_NAME}
EOF

echo "==> Step 2: Waiting for pod '${POD_NAME}' to be in 'Running' state..."
while true; do
  status=$(kubectl get pod ${POD_NAME} -n ${NAMESPACE} -o jsonpath='{.status.phase}')
  [[ "$status" == "Running" ]] && break
  echo "  → Current status: $status... waiting..."
  sleep 2
done

echo "==> Step 3: Copying local vector store to the pod PVC..."
kubectl cp ${LOCAL_PATH}. ${NAMESPACE}/${POD_NAME}:/vector_store/

echo "==> Step 4: Restarting backend deployment to load new vector store..."
kubectl rollout restart deployment backend -n ${NAMESPACE}

echo "==> Step 5: Cleaning up temporary pod..."
kubectl delete pod ${POD_NAME} -n ${NAMESPACE}

echo "✅ Vector store upload complete and backend restarted." 