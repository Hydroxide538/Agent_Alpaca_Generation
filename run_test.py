#!/usr/bin/env python3
"""
Simple script to run the Ollama-only test with proper environment setup
"""

import os
import subprocess
import sys

# Set the placeholder OpenAI API key to prevent embedchain from failing
os.environ['OPENAI_API_KEY'] = 'placeholder_key_not_used'

# Run the test
try:
    result = subprocess.run([sys.executable, 'test_ollama_only.py'], 
                          capture_output=False, 
                          text=True)
    sys.exit(result.returncode)
except Exception as e:
    print(f"Error running test: {e}")
    sys.exit(1)
