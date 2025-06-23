#!/usr/bin/env python3
"""
Test script to verify ChromaDB Pydantic deprecation warning fix
"""

import warnings
import sys
import os

# Capture warnings
warnings.filterwarnings('error', category=DeprecationWarning)

try:
    print("Testing ChromaDB import and basic functionality...")
    
    # Import ChromaDB
    import chromadb
    from chromadb.config import Settings
    print("✓ ChromaDB imported successfully")
    
    # Test basic ChromaDB functionality
    client = chromadb.Client()
    print("✓ ChromaDB client created successfully")
    
    # Create a test collection
    collection = client.create_collection("test_collection")
    print("✓ Test collection created successfully")
    
    # Add some test data
    collection.add(
        documents=["This is a test document"],
        metadatas=[{"source": "test"}],
        ids=["test_id_1"]
    )
    print("✓ Test document added successfully")
    
    # Query the collection
    results = collection.query(
        query_texts=["test"],
        n_results=1
    )
    print("✓ Query executed successfully")
    
    # Clean up
    client.delete_collection("test_collection")
    print("✓ Test collection deleted successfully")
    
    print("\n🎉 SUCCESS: No Pydantic deprecation warnings detected!")
    print(f"ChromaDB version: {chromadb.__version__}")
    
    # Check Pydantic version
    import pydantic
    print(f"Pydantic version: {pydantic.__version__}")
    
except DeprecationWarning as e:
    print(f"\n❌ DEPRECATION WARNING DETECTED: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    sys.exit(1)
