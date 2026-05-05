#!/usr/bin/env python3
"""
Test script for Ollama integration
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.ollama_helper import get_ollama_helper
from src.utils.gemini_helper import get_backend_status, generate_text

def test_ollama():
    print("=== Testing Ollama Integration ===")
    
    # Test 1: Check Ollama helper
    print("\n1. Testing Ollama Helper:")
    helper = get_ollama_helper()
    print(f"   Available: {helper.available}")
    print(f"   Base URL: {helper.base_url}")
    print(f"   Models: {helper.list_models()}")
    
    if helper.available:
        # Test 2: Generate text with Ollama
        print("\n2. Testing text generation:")
        test_prompt = "Write a brief summary of machine learning in 2-3 sentences."
        result = helper.generate_text(test_prompt)
        print(f"   Result: {result[:200]}..." if len(result) > 200 else f"   Result: {result}")
    
    # Test 3: Check backend status
    print("\n3. Backend Status:")
    status = get_backend_status()
    for backend, info in status.items():
        print(f"   {backend}: {info}")
    
    # Test 4: Test auto backend selection
    print("\n4. Testing auto backend selection:")
    auto_result = generate_text("What is artificial intelligence?")
    print(f"   Auto result: {auto_result[:200]}..." if len(auto_result) > 200 else f"   Auto result: {auto_result}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_ollama()
