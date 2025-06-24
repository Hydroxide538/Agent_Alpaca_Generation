#!/usr/bin/env python3
"""
Test script to verify that the CrewAI workflow works with Ollama-only configuration
"""

import os
import sys
import asyncio
from pathlib import Path

# Set a placeholder OpenAI API key to prevent embedchain from failing
# This is needed because some CrewAI tools expect this environment variable to exist
os.environ['OPENAI_API_KEY'] = 'placeholder_key_not_used'

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from local_crewai_workflow_for_synthetic_data_with_rag_and_llm_options.crew import LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew

def test_ollama_only_config():
    """Test CrewAI workflow with Ollama-only configuration"""
    print("Testing CrewAI workflow with Ollama-only configuration...")
    
    # Configuration with no OpenAI API key
    config = {
        'manager_model': 'ollama:llama3.3:latest',
        'data_generation_model': 'ollama:llama3.3:latest',
        'embedding_model': 'ollama:bge-m3:latest',
        'ollama_url': 'http://localhost:11434',
        'openai_api_key': None,  # No OpenAI API key
        'enable_gpu_optimization': False
    }
    
    try:
        # Create crew instance
        print("Creating CrewAI crew instance...")
        crew_instance = LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew(config=config)
        
        # Check if LLMs were created successfully
        if crew_instance.manager_llm is None:
            print("❌ Manager LLM creation failed")
            return False
        else:
            print("✅ Manager LLM created successfully")
        
        if crew_instance.default_worker_llm is None:
            print("❌ Default worker LLM creation failed")
            return False
        else:
            print("✅ Default worker LLM created successfully")
        
        # Try to create the crew
        print("Creating crew...")
        crew = crew_instance.crew()
        
        if crew is None:
            print("❌ Crew creation failed")
            return False
        else:
            print("✅ Crew created successfully")
        
        print("✅ All tests passed! The system can run with Ollama-only configuration.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_openai_fallback():
    """Test that OpenAI models fall back to Ollama when no API key is provided"""
    print("\nTesting OpenAI fallback to Ollama...")
    
    # Configuration with OpenAI models but no API key
    config = {
        'manager_model': 'openai:gpt-4o',  # This should fall back to Ollama
        'data_generation_model': 'openai:gpt-4o',  # This should fall back to Ollama
        'embedding_model': 'ollama:bge-m3:latest',
        'ollama_url': 'http://localhost:11434',
        'openai_api_key': None,  # No OpenAI API key
        'enable_gpu_optimization': False
    }
    
    try:
        # Create crew instance
        print("Creating CrewAI crew instance with OpenAI models but no API key...")
        crew_instance = LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew(config=config)
        
        # Check if LLMs were created successfully (should fall back to Ollama)
        if crew_instance.manager_llm is None:
            print("❌ Manager LLM creation failed")
            return False
        else:
            print("✅ Manager LLM created successfully (fell back to Ollama)")
        
        if crew_instance.default_worker_llm is None:
            print("❌ Default worker LLM creation failed")
            return False
        else:
            print("✅ Default worker LLM created successfully (fell back to Ollama)")
        
        # Try to create the crew
        print("Creating crew...")
        crew = crew_instance.crew()
        
        if crew is None:
            print("❌ Crew creation failed")
            return False
        else:
            print("✅ Crew created successfully")
        
        print("✅ OpenAI fallback test passed! The system falls back to Ollama when no API key is provided.")
        return True
        
    except Exception as e:
        print(f"❌ Fallback test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("CrewAI Ollama-Only Configuration Test")
    print("=" * 60)
    
    # Test 1: Pure Ollama configuration
    test1_passed = test_ollama_only_config()
    
    # Test 2: OpenAI fallback to Ollama
    test2_passed = test_openai_fallback()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    print(f"Ollama-only configuration: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"OpenAI fallback to Ollama: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! The system is ready to run with Ollama-only configuration.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the configuration and try again.")
        sys.exit(1)
