import asyncio
import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.llm_manager import LLMManager
from backend.models import WorkflowConfig, DocumentInfo

async def test_workflow_model_setup():
    """Test the exact model setup process that the workflow manager uses"""
    print("Testing workflow model setup process...")
    
    # Create a mock workflow config similar to what would be used
    config = WorkflowConfig(
        data_generation_model="ollama:llama3.3:latest",  # Example data generation model
        embedding_model="ollama:bge-m3:latest",  # This is the failing model
        reranking_model=None,
        openai_api_key=None,
        ollama_url="http://host.docker.internal:11434",
        enable_gpu_optimization=True,
        documents=[],
        workflow_type="full"
    )
    
    print(f"Config:")
    print(f"  Data generation model: {config.data_generation_model}")
    print(f"  Embedding model: {config.embedding_model}")
    print(f"  Ollama URL: {config.ollama_url}")
    
    # Create LLM manager and test models
    llm_manager = LLMManager()
    
    try:
        print(f"\nTesting models...")
        test_results = await llm_manager.test_models(config)
        
        print(f"\nTest Results:")
        for model_type, result in test_results.items():
            status = "✓ SUCCESS" if result.success else "✗ FAILED"
            print(f"  {model_type}: {status}")
            print(f"    Message: {result.message}")
            if result.error:
                print(f"    Error: {result.error}")
            if result.response_time:
                print(f"    Response time: {result.response_time:.2f}ms")
            print()
        
        # Check if any failed
        failed_tests = [model_type for model_type, result in test_results.items() if not result.success]
        if failed_tests:
            print(f"❌ Failed tests: {', '.join(failed_tests)}")
            return False
        else:
            print(f"✅ All tests passed!")
            return True
            
    except Exception as e:
        print(f"❌ Exception during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_workflow_model_setup())
    if not success:
        sys.exit(1)
