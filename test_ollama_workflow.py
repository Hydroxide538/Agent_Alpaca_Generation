#!/usr/bin/env python3
"""
Test script to verify Ollama workflow functionality without OpenAI API key
"""

import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.models import WorkflowConfig, DocumentInfo
from backend.llm_manager import LLMManager
from local_crewai_workflow_for_synthetic_data_with_rag_and_llm_options.crew import LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew

async def test_ollama_workflow():
    """Test the Ollama workflow configuration"""
    
    print("🧪 Testing Ollama Workflow Configuration")
    print("=" * 50)
    
    # First, get available models from Ollama
    llm_manager = LLMManager()
    available_models = await llm_manager.get_available_models("http://host.docker.internal:11434")
    
    # Select appropriate models from available ones
    data_gen_model = None
    embedding_model = None
    reranking_model = None
    
    # Look for suitable data generation models (general purpose LLMs)
    data_gen_candidates = ["mistral-small3.2", "llama3.3", "deepseek-r1", "devstral"]
    for candidate in data_gen_candidates:
        if any(candidate in model for model in available_models):
            # Find the exact model name
            for model in available_models:
                if candidate in model:
                    data_gen_model = f"ollama:{model}"
                    break
            break
    
    # Look for embedding models
    embedding_candidates = ["snowflake-arctic-embed2", "bge-m3"]
    for candidate in embedding_candidates:
        if any(candidate in model for model in available_models):
            for model in available_models:
                if candidate in model:
                    embedding_model = f"ollama:{model}"
                    break
            break
    
    # Look for reranking models
    reranking_candidates = ["bge-m3"]
    for candidate in reranking_candidates:
        if any(candidate in model for model in available_models):
            for model in available_models:
                if candidate in model:
                    reranking_model = f"ollama:{model}"
                    break
            break
    
    if not data_gen_model:
        print("❌ No suitable data generation model found in Ollama")
        print(f"Available models: {available_models}")
        return
    
    if not embedding_model:
        print("❌ No suitable embedding model found in Ollama")
        print(f"Available models: {available_models}")
        return
    
    # Create a test configuration with dynamically selected Ollama models
    config = WorkflowConfig(
        data_generation_model=data_gen_model,
        embedding_model=embedding_model,
        reranking_model=reranking_model,
        openai_api_key=None,  # Explicitly no OpenAI API key
        ollama_url="http://host.docker.internal:11434",
        enable_gpu_optimization=True,
        documents=[],
        workflow_type="data_generation_only"
    )
    
    print(f"📋 Configuration:")
    print(f"   Data Generation Model: {config.data_generation_model}")
    print(f"   Embedding Model: {config.embedding_model}")
    print(f"   OpenAI API Key: {'✅ Provided' if config.openai_api_key else '❌ Not provided (using Ollama)'}")
    print(f"   Ollama URL: {config.ollama_url}")
    print()
    
    # Test 1: LLM Manager model testing
    print("🔍 Test 1: Testing LLM Manager")
    llm_manager = LLMManager()
    
    try:
        test_results = await llm_manager.test_models(config)
        
        for model_type, result in test_results.items():
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"   {model_type}: {status} - {result.message}")
            if not result.success and result.error:
                print(f"      Error: {result.error}")
    
    except Exception as e:
        print(f"   ❌ LLM Manager test failed: {str(e)}")
    
    print()
    
    # Test 2: CrewAI Crew initialization
    print("🔍 Test 2: Testing CrewAI Crew Initialization")
    
    try:
        # Convert config to crew format
        crew_config = {
            "data_generation_model": config.data_generation_model,
            "embedding_model": config.embedding_model,
            "reranking_model": config.reranking_model,
            "openai_api_key": config.openai_api_key,
            "ollama_url": config.ollama_url,
            "enable_gpu_optimization": config.enable_gpu_optimization
        }
        
        crew_instance = LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew(config=crew_config)
        
        # Check if LLMs were initialized
        if crew_instance.data_generation_llm is not None:
            print("   ✅ Data generation LLM initialized successfully")
        else:
            print("   ❌ Data generation LLM failed to initialize")
            
        if crew_instance.embedding_llm is not None:
            print("   ✅ Embedding LLM initialized successfully")
        else:
            print("   ❌ Embedding LLM failed to initialize")
            
        print("   ✅ CrewAI crew created without requiring OpenAI API key")
        
    except Exception as e:
        print(f"   ❌ CrewAI crew initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test 3: Configuration validation
    print("🔍 Test 3: Testing Configuration Validation")
    
    # Test with OpenAI model but no API key (should fail)
    openai_config = WorkflowConfig(
        data_generation_model="openai:gpt-3.5-turbo",
        embedding_model="openai:text-embedding-ada-002",
        openai_api_key=None,  # No API key
        ollama_url="http://localhost:11434"
    )
    
    try:
        openai_results = await llm_manager.test_models(openai_config)
        for model_type, result in openai_results.items():
            if not result.success and "OpenAI API key" in result.message:
                print(f"   ✅ Correctly rejected OpenAI model without API key: {model_type}")
            else:
                print(f"   ❌ Should have rejected OpenAI model without API key: {model_type}")
    except Exception as e:
        print(f"   ❌ OpenAI validation test failed: {str(e)}")
    
    print()
    print("🎉 Test completed!")
    print("=" * 50)
    print()
    print("💡 Next steps:")
    print("   1. Make sure Ollama is running on localhost:11434")
    print("   2. Ensure you have the required models pulled:")
    print(f"      - Data generation model: {data_gen_model.replace('ollama:', '') if data_gen_model else 'Not found'}")
    print(f"      - Embedding model: {embedding_model.replace('ollama:', '') if embedding_model else 'Not found'}")
    if reranking_model:
        print(f"      - Reranking model: {reranking_model.replace('ollama:', '')}")
    print("   3. Test the full workflow through the frontend")

if __name__ == "__main__":
    asyncio.run(test_ollama_workflow())
