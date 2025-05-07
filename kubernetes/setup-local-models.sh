#!/bin/bash
# Script to set up local models for OPT-RAG Kubernetes deployment

set -e  # Exit on error

# Default values
MODEL_ID="Qwen/Qwen2.5-1.5B-Chat"
HOST_DIR="/opt/models"
MODEL_DIR="qwen2.5-1.5b"
PREDOWNLOAD=false
MINIKUBE=false
HF_TOKEN=""
USE_EXISTING=false
EXISTING_MODELS_DIR="./rag-pipeline/models"

# Display help
function show_help {
    echo "Usage: $0 [options]"
    echo "Setup local models for OPT-RAG Kubernetes deployment"
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -m, --model-id ID         HuggingFace model ID (default: $MODEL_ID)"
    echo "  -d, --host-dir DIR        Host directory to store models (default: $HOST_DIR)"
    echo "  -o, --model-dir NAME      Output directory name (default: $MODEL_DIR)"
    echo "  -p, --predownload         Pre-download the model instead of letting the pod do it"
    echo "  -k, --minikube            Set up for Minikube environment"
    echo "  -t, --token TOKEN         Hugging Face token for downloading gated models"
    echo "  -e, --use-existing        Use existing models directory (default: $EXISTING_MODELS_DIR)"
    echo "  -x, --existing-dir DIR    Path to existing models directory (default: $EXISTING_MODELS_DIR)"
    echo ""
    echo "Example:"
    echo "  $0 --model-id 'Qwen/Qwen2.5-0.5B-Chat' --predownload --token 'hf_...' "
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit 0
            ;;
        -m|--model-id)
            MODEL_ID="$2"
            shift
            shift
            ;;
        -d|--host-dir)
            HOST_DIR="$2"
            shift
            shift
            ;;
        -o|--model-dir)
            MODEL_DIR="$2"
            shift
            shift
            ;;
        -p|--predownload)
            PREDOWNLOAD=true
            shift
            ;;
        -k|--minikube)
            MINIKUBE=true
            shift
            ;;
        -t|--token)
            HF_TOKEN="$2"
            shift
            shift
            ;;
        -e|--use-existing)
            USE_EXISTING=true
            shift
            ;;
        -x|--existing-dir)
            EXISTING_MODELS_DIR="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "Setting up local models for OPT-RAG..."
echo "Model ID: $MODEL_ID"
echo "Host directory: $HOST_DIR"
echo "Model directory: $MODEL_DIR"

# Check for pip and python
if ! command -v python &> /dev/null; then
    echo "Python not found. Please install Python 3.8 or higher."
    exit 1
fi

# Create host directory
if [ "$MINIKUBE" = true ]; then
    echo "Setting up for Minikube environment..."
    if ! command -v minikube &> /dev/null; then
        echo "Minikube not found. Please install Minikube first."
        exit 1
    fi
    
    echo "Creating directory in Minikube VM..."
    minikube ssh "sudo mkdir -p $HOST_DIR && sudo chmod 777 $HOST_DIR"
    
    if [ "$PREDOWNLOAD" = true ]; then
        echo "For Minikube, you need to manually mount the directory and download the model."
        echo "Run the following commands:"
        echo ""
        echo "  # Start a minikube mount in a separate terminal"
        echo "  minikube mount \$LOCAL_DIR:$HOST_DIR"
        echo ""
        echo "  # In your local machine, download the model"
        echo "  python3 -m venv temp-env"
        echo "  source temp-env/bin/activate"
        echo "  pip install huggingface_hub"
        
        if [ -n "$HF_TOKEN" ]; then
            echo "  # Download with token"
            echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID', local_dir='./\$LOCAL_DIR/$MODEL_DIR', token='$HF_TOKEN', local_dir_use_symlinks=False)\""
        else
            echo "  # Download without token (may fail for gated models)"
            echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID', local_dir='./\$LOCAL_DIR/$MODEL_DIR', local_dir_use_symlinks=False)\""
        fi
        
        echo "  deactivate"
        echo "  rm -rf temp-env"
    fi
else
    echo "Creating host directory..."
    sudo mkdir -p $HOST_DIR
    sudo chmod 777 $HOST_DIR
    
    if [ "$USE_EXISTING" = true ]; then
        echo "Using existing models from $EXISTING_MODELS_DIR"
        
        # Check if the source directory exists
        if [ -d "$EXISTING_MODELS_DIR" ]; then
            MODELS_FOUND=false
            
            # Check if specific model directory exists
            if [ -d "$EXISTING_MODELS_DIR/$MODEL_DIR" ]; then
                echo "Found model directory: $EXISTING_MODELS_DIR/$MODEL_DIR"
                
                # Create symbolic link or copy to host directory
                echo "Creating symbolic link from $EXISTING_MODELS_DIR/$MODEL_DIR to $HOST_DIR/$MODEL_DIR"
                ln -sf "$(pwd)/$EXISTING_MODELS_DIR/$MODEL_DIR" "$HOST_DIR/$MODEL_DIR"
                MODELS_FOUND=true
            elif [ -d "$EXISTING_MODELS_DIR" ]; then
                # Check if models are directly in the directory
                echo "Using entire models directory: $EXISTING_MODELS_DIR"
                
                # Create symbolic link for the whole directory
                echo "Creating symbolic link from $EXISTING_MODELS_DIR to $HOST_DIR/$MODEL_DIR"
                ln -sf "$(pwd)/$EXISTING_MODELS_DIR" "$HOST_DIR/$MODEL_DIR"
                MODELS_FOUND=true
            fi
            
            if [ "$MODELS_FOUND" = false ]; then
                echo "Warning: Could not find model directory in $EXISTING_MODELS_DIR"
                echo "Will try to download model if predownload is set or let the pod download it"
            fi
        else
            echo "Error: Existing models directory not found: $EXISTING_MODELS_DIR"
            exit 1
        fi
    elif [ "$PREDOWNLOAD" = true ]; then
        echo "Pre-downloading model (this may take some time)..."
        
        # Check for Python3 and different versions
        if command -v python3 &> /dev/null; then
            PYTHON_CMD="python3"
        elif command -v python &> /dev/null; then
            PYTHON_CMD="python"
        else
            echo "No Python executable found. Please install Python 3.8 or higher."
            exit 1
        fi
        
        # Create a temporary virtual environment
        echo "Creating a temporary virtual environment for huggingface_hub..."
        TEMP_VENV="/tmp/opt-rag-venv"
        $PYTHON_CMD -m venv $TEMP_VENV
        
        # Activate the virtual environment
        source $TEMP_VENV/bin/activate
        
        # Install huggingface_hub in the virtual environment
        pip install huggingface_hub
        
        # Download the model with the virtual environment Python
        echo "Downloading the model (this may take a while)..."
        
        if [ -n "$HF_TOKEN" ]; then
            echo "Using provided Hugging Face token for download"
            python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID', local_dir='$HOST_DIR/$MODEL_DIR', token='$HF_TOKEN', local_dir_use_symlinks=False)"
        else
            echo "No Hugging Face token provided. Some models like Qwen may require authentication."
            echo "Attempting to download without token..."
            python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID', local_dir='$HOST_DIR/$MODEL_DIR', local_dir_use_symlinks=False)"
        fi
        
        # Deactivate the virtual environment
        deactivate
        
        # Clean up
        echo "Cleaning up temporary virtual environment..."
        rm -rf $TEMP_VENV
        
        echo "Model downloaded successfully to $HOST_DIR/$MODEL_DIR"
    fi
fi

# Get the script directory - macOS compatible way
echo "Updating Kubernetes backend.yaml with model settings..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
    SED_CMD="sed -i ''"
else
    # Linux
    SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
    SED_CMD="sed -i"
fi

BACKEND_YAML="$SCRIPT_DIR/cloud/backend.yaml"

if [ -f "$BACKEND_YAML" ]; then
    # Create a backup
    cp "$BACKEND_YAML" "$BACKEND_YAML.bak"
    
    # Update the MODEL_ID in the yaml file - macOS compatible
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|value: \"Qwen/Qwen.*\"|value: \"$MODEL_ID\"|g" "$BACKEND_YAML"
        sed -i '' "s|value: \"/app/models/.*\"|value: \"/app/models/$MODEL_DIR\"|g" "$BACKEND_YAML"
        sed -i '' "s|path: /opt/models|path: $HOST_DIR|g" "$BACKEND_YAML"
    else
        sed -i "s|value: \"Qwen/Qwen.*\"|value: \"$MODEL_ID\"|g" "$BACKEND_YAML"
        sed -i "s|value: \"/app/models/.*\"|value: \"/app/models/$MODEL_DIR\"|g" "$BACKEND_YAML"
        sed -i "s|path: /opt/models|path: $HOST_DIR|g" "$BACKEND_YAML"
    fi
    
    echo "Updated $BACKEND_YAML successfully."
else
    echo "Warning: Could not find $BACKEND_YAML to update."
    echo "You will need to manually update the MODEL_ID and MODEL_PATH in your Kubernetes deployment."
fi

echo ""
echo "Setup complete! Next steps:"
if [ "$USE_EXISTING" = true ] && [ "$MODELS_FOUND" = true ]; then
    echo "1. Existing models have been linked to $HOST_DIR/$MODEL_DIR"
elif [ "$PREDOWNLOAD" = true ]; then
    echo "1. The model has been downloaded to $HOST_DIR/$MODEL_DIR"
else
    echo "1. The model will be downloaded automatically when you deploy the application"
fi
echo "2. Deploy your Kubernetes application with: kubectl apply -f kubernetes/cloud/"
echo "3. Check pod status with: kubectl get pods -n opt-rag"
echo ""
echo "For more details, see the documentation in kubernetes/README.md" 