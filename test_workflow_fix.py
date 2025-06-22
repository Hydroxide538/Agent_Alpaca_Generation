#!/usr/bin/env python3
"""
Quick test to verify the workflow manager fix
"""

import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from local_crewai_workflow_for_synthetic_data_with_rag_and_llm_options.crew import LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew

def test_workflow_fix():
    """Test that the workflow fix resolves the LLM failure"""
    
    print("🔧 Testing Workflow Manager Fix")
    print("=" * 40)
    
    # Create crew configuration with working models
    crew_config = {
        "data_generation_model": "ollama:mistral-small3.2:latest",
        "embedding_model": "ollama:snowflake-arctic-embed2:latest",
        "reranking_model": "ollama:bge-m3:latest",
        "openai_api_key": None,
        "ollama_url": "http://host.docker.internal:11434",
        "enable_gpu_optimization": True
    }
    
    print("🔍 Test 1: CrewAI Crew Creation")
    try:
        crew_instance = LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew(config=crew_config)
        print("   ✅ CrewAI crew instance created successfully")
        
        # Check if LLMs are properly initialized
        if crew_instance.data_generation_llm:
            print(f"   ✅ Data generation LLM: {crew_instance.data_generation_llm.model}")
        else:
            print("   ❌ Data generation LLM not initialized")
            return False
            
    except Exception as e:
        print(f"   ❌ CrewAI crew creation failed: {str(e)}")
        return False
    
    print()
    print("🔍 Test 2: Crew Kickoff (Quick Test)")
    try:
        # Prepare minimal inputs
        inputs = {
            "documents": [],  # Empty for quick test
            "workflow_type": "full",
            "data_generation_model": crew_config["data_generation_model"],
            "embedding_model": crew_config["embedding_model"],
            "reranking_model": crew_config["reranking_model"],
            "enable_gpu_optimization": False  # Disable for quick test
        }
        
        print("   🚀 Starting CrewAI workflow (this may take a moment)...")
        
        # This should now work without the "LLM Failed" error
        result = crew_instance.crew().kickoff(inputs=inputs)
        
        print("   ✅ CrewAI workflow completed without LLM failure!")
        print(f"   📄 Result type: {type(result)}")
        
        # Show a snippet of the result
        result_str = str(result)
        if len(result_str) > 200:
            print(f"   📄 Result preview: {result_str[:200]}...")
        else:
            print(f"   📄 Result: {result_str}")
            
        return True
        
    except Exception as e:
        print(f"   ❌ CrewAI workflow failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    print("🎉 Fix verification completed!")

if __name__ == "__main__":
    success = test_workflow_fix()
    if success:
        print("\n✅ The workflow manager fix appears to be working!")
        print("💡 You should now be able to run the full workflow without LLM failures.")
    else:
        print("\n❌ The fix needs more work. Check the error messages above.")
