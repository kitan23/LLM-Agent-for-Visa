#!/bin/bash

# This script uploads all PDF documents from the examples folder to the OPT-RAG vector store

API_URL="http://localhost:8000/documents"
EXAMPLES_DIR="./rag-pipeline/examples"

echo "Starting to upload documents to vector store..."

# Function to upload a file
upload_file() {
    local file_path=$1
    local file_name=$(basename "$file_path")
    
    echo "Uploading $file_name..."
    
    # Extract document type from filename (assuming the first part of filename is the document type)
    local doc_type="${file_name%%_*}"
    
    # Upload file using curl
    response=$(curl -s -X POST \
        -F "file=@$file_path" \
        -F "document_type=$doc_type" \
        "$API_URL")
    
    # Check if upload was successful
    if [[ "$response" == *"successfully"* || "$response" == *"success"* ]]; then
        echo "✅ Successfully uploaded $file_name"
    else
        echo "❌ Failed to upload $file_name. Response:"
        echo "$response"
    fi
    
    # Small pause to avoid overwhelming the server
    sleep 1
}

# Find all PDF files in the examples directory and upload them
for pdf_file in "$EXAMPLES_DIR"/*.pdf; do
    if [ -f "$pdf_file" ]; then
        upload_file "$pdf_file"
    fi
done

echo "Document upload process completed."
echo "You can verify the documents in the vector store by calling: curl -s http://localhost:8000/documents | jq" 