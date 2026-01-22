#!/usr/bin/env python3
"""
Test script for MCP server tools.
Tests the tools directly without going through MCP protocol.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server.model_client import get_model_client
from mcp_server.config import Config


def test_query_syntax():
    """Test syntax query."""
    print("Testing query_cobol_syntax...")
    client = get_model_client()
    
    response = client.query(
        "What is the syntax for PERFORM VARYING in COBOL?",
        temperature=0.7
    )
    print(f"Response: {response}\n")
    return response


def test_query_compiler():
    """Test compiler query."""
    print("Testing query_cobol_compiler...")
    client = get_model_client()
    
    response = client.query(
        "How should I implement symbol table lookups in my COBOL compiler?",
        temperature=0.5,
        max_tokens=1024
    )
    print(f"Response: {response}\n")
    return response


def test_get_example():
    """Test code example generation."""
    print("Testing get_cobol_example...")
    client = get_model_client()
    
    response = client.query(
        "Generate COBOL code example in modern style for: file handling with sequential access",
        temperature=0.3,
        max_tokens=2048
    )
    print(f"Response: {response}\n")
    return response


def main():
    """Run all tests."""
    print("=" * 60)
    print("MCP Server Tool Tests")
    print("=" * 60)
    print(f"Model path: {Config.get_model_path()}")
    print(f"Base model: {Config.BASE_MODEL}")
    print("=" * 60)
    print()
    
    # Check if model exists
    if not Config.get_model_path().exists():
        print(f"ERROR: Model path does not exist: {Config.get_model_path()}")
        print("Please ensure the fine-tuned model is available.")
        return
    
    try:
        # Test each tool
        test_query_syntax()
        test_query_compiler()
        test_get_example()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
