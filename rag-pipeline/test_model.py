#!/usr/bin/env python3
"""
Simple model testing script for OPT-RAG.

This script tests the basic functionality of the model without the full RAG pipeline.
"""

import os
import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer

def test_model(model_path, query, device=None):
    """Test basic model functionality.
    
    Args:
        model_path: Path to the model
        query: Input text for the model
        device: Device to use (cuda, mps, cpu, or None for auto-detection)
    """
    start_time = time.time()
    
    # Detect device if not specified
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    print(f"Using device: {device}")
    print(f"Loading model from {model_path}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32
    )
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Simple visa query prompt
    prompt = f"""You are an expert in international student visas. Please answer the following question:

Question: {query}

Answer:"""
    
    print("\nGenerating text...")
    print("-" * 50)
    
    # Use Hugging Face's built-in TextStreamer
    streamer = TextStreamer(tokenizer, skip_special_tokens=True)
    
    # Tokenize the input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    
    # Generate text with streaming
    generation_start = time.time()
    
    # Generate with streamer
    _ = model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    
    print("\n" + "-" * 50)
    
    generation_time = time.time() - generation_start
    print(f"\nGeneration completed in {generation_time:.2f} seconds")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test model generation")
    parser.add_argument(
        "--model", 
        type=str, 
        default="models/qwen2.5-0.5b",
        help="Path to the model"
    )
    parser.add_argument(
        "--query", 
        type=str, 
        default="What is OPT and who is eligible for it?",
        help="Query to test"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cuda", "mps", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    test_model(
        model_path=args.model,
        query=args.query,
        device=args.device
    )

if __name__ == "__main__":
    main() 