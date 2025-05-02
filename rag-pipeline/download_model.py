#!/usr/bin/env python3
"""
Model downloader for OPT-RAG project.

This script downloads smaller models from Hugging Face for local inference.
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

def download_model(model_id, output_dir, local_files_only=False):
    """Download a model from Hugging Face Hub.
    
    Args:
        model_id: ID of the model on Hugging Face (e.g., 'Qwen/Qwen2.5-0.5B-Instruct')
        output_dir: Directory to save the model
        local_files_only: If True, only use local files (no download)
    """
    print(f"Downloading model: {model_id}")
    print(f"Destination: {output_dir}")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            local_files_only=local_files_only
        )
        print(f"Model downloaded successfully to {output_dir}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

def main():
    """Main function to parse arguments and download model."""
    parser = argparse.ArgumentParser(description="Download models for OPT-RAG")
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model ID on Hugging Face Hub"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save the model"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Use only local files (no download)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    model_name = args.model.split("/")[-1].lower()
    model_path = output_path / model_name
    os.makedirs(model_path, exist_ok=True)
    
    # Download the model
    download_model(
        model_id=args.model,
        output_dir=str(model_path),
        local_files_only=args.local_only
    )

if __name__ == "__main__":
    main() 