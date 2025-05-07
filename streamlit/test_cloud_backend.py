#!/usr/bin/env python3
"""
Test script for OPT-RAG backend API.

Usage:
    python test_cloud_backend.py --url http://backend-ip:8000
"""

import argparse
import json
import time
import requests
import sys

def test_health(base_url, timeout=30):
    """Test the health endpoint."""
    url = f"{base_url}/health"
    print(f"Testing health endpoint: {url}")
    
    try:
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        elapsed = time.time() - start_time
        
        print(f"Status: {response.status_code}")
        print(f"Response time: {elapsed:.2f} seconds")
        print(f"Response body: {response.text}")
        
        if response.status_code == 200:
            print("✅ Health check successful")
            return True
        else:
            print("❌ Health check failed")
            return False
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_query(base_url, query="What is OPT?", timeout=60):
    """Test the query endpoint with a sample question."""
    url = f"{base_url}/api/query"
    print(f"\nTesting query endpoint: {url}")
    print(f"Query: \"{query}\"")
    
    try:
        start_time = time.time()
        response = requests.post(
            url,
            json={"question": query},
            timeout=timeout
        )
        elapsed = time.time() - start_time
        
        print(f"Status: {response.status_code}")
        print(f"Response time: {elapsed:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print("\nResponse:")
            print(f"Answer: {result.get('answer', 'No answer')[:200]}...")
            print(f"Processing time: {result.get('processing_time', 'N/A')} seconds")
            print("✅ Query successful")
            return True
        else:
            print(f"Response: {response.text}")
            print("❌ Query failed")
            return False
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_stream(base_url, query="Tell me about F1 visa", timeout=60):
    """Test the streaming endpoint with a sample question."""
    url = f"{base_url}/api/query/stream"
    print(f"\nTesting streaming endpoint: {url}")
    print(f"Query: \"{query}\"")
    
    try:
        start_time = time.time()
        with requests.post(
            url,
            json={"question": query},
            stream=True,
            timeout=timeout
        ) as response:
            elapsed = time.time() - start_time
            print(f"Initial response time: {elapsed:.2f} seconds")
            
            if response.status_code == 200:
                print("\nStreaming response:")
                for i, line in enumerate(response.iter_lines()):
                    if line and line.startswith(b'data: '):
                        data = line[6:].decode('utf-8')
                        if data == "[DONE]":
                            break
                        print(data, end="", flush=True)
                        if i >= 5:  # Just show first few tokens
                            print("...")
                            break
                
                elapsed = time.time() - start_time
                print(f"\nTotal streaming time: {elapsed:.2f} seconds")
                print("✅ Streaming query successful")
                return True
            else:
                print(f"Response: {response.text}")
                print("❌ Streaming query failed")
                return False
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Run the tests."""
    parser = argparse.ArgumentParser(description='Test OPT-RAG backend API')
    parser.add_argument('--url', type=str, required=True, help='Base URL of the backend API (e.g., http://ip:8000)')
    parser.add_argument('--timeout', type=int, default=60, help='Request timeout in seconds')
    parser.add_argument('--test', type=str, choices=['health', 'query', 'stream', 'all'], default='all', 
                        help='Which test to run')
    parser.add_argument('--question', type=str, default="What is OPT?", 
                        help='Custom question to ask')
    
    args = parser.parse_args()
    
    # Remove trailing slash if present
    base_url = args.url.rstrip('/')
    
    print(f"Testing OPT-RAG backend at: {base_url}")
    print(f"Timeout: {args.timeout} seconds")
    print("-" * 50)
    
    if args.test == 'health' or args.test == 'all':
        health_ok = test_health(base_url, args.timeout)
        if not health_ok and args.test == 'all':
            print("\n⚠️ Health check failed, skipping other tests")
            sys.exit(1)
    
    if args.test == 'query' or args.test == 'all':
        test_query(base_url, args.question, args.timeout)
    
    if args.test == 'stream' or args.test == 'all':
        test_stream(base_url, args.question, args.timeout)

if __name__ == "__main__":
    main() 