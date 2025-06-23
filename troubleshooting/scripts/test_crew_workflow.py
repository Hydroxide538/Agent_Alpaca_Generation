#!/usr/bin/env python3
"""
Test script to verify CrewAI workflow execution with proper model configuration
"""

import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from local_crewai_workflow_for_synthetic_data_with_rag_and_llm_options.crew import LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew

def test_crew_workflow():
    """Test the CrewAI workflow execution"""
    
    print("🧪 Testing CrewAI Workflow Execution")
    print("=" * 50)
    
    # Create crew configuration with working models
    crew_config = {
        "data_generation_model": "ollama:mistral-small3.2:latest",
        "embedding_model": "ollama:snowflake-arctic-embed2:latest",
        "reranking_model": "ollama:bge-m3:latest",
        "openai_api_key": None,
        "ollama_url": "http://host.docker.internal:11434",
        "enable_gpu_optimization": True
    }
    
    print(f"📋 Configuration:")
    print(f"   Data Generation Model: {crew_config['data_generation_model']}")
    print(f"   Embedding Model: {crew_config['embedding_model']}")
    print(f"   Reranking Model: {crew_config['reranking_model']}")
    print(f"   Ollama URL: {crew_config['ollama_url']}")
    print()
    
    try:
        # Create CrewAI crew instance
        print("🔍 Creating CrewAI crew instance...")
        crew_instance = LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew(config=crew_config)
        
        # Check LLM initialization
        print("🔍 Checking LLM initialization...")
        if crew_instance.data_generation_llm is not None:
            print("   ✅ Data generation LLM initialized successfully")
            print(f"      Model: {crew_instance.data_generation_llm.model}")
        else:
            print("   ❌ Data generation LLM failed to initialize")
            return
            
        if crew_instance.embedding_llm is not None:
            print("   ✅ Embedding LLM initialized successfully")
            print(f"      Model: {crew_instance.embedding_llm.model}")
        else:
            print("   ❌ Embedding LLM failed to initialize")
            
        if crew_instance.reranking_llm is not None:
            print("   ✅ Reranking LLM initialized successfully")
            print(f"      Model: {crew_instance.reranking_llm.model}")
        else:
            print("   ⚠️  Reranking LLM not initialized (optional)")
        
        print()
        
        # Check agents
        print("🔍 Checking agent configuration...")
        try:
            # Get the crew to access agents
            crew = crew_instance.crew()
            agents = crew.agents
            for i, agent in enumerate(agents):
                print(f"   Agent {i+1}: {agent.role}")
                if agent.llm:
                    print(f"      LLM: {agent.llm.model}")
                else:
                    print("      ❌ No LLM assigned")
        except Exception as e:
            print(f"   ⚠️  Could not access agents: {str(e)}")
        
        print()
        
        # Prepare inputs for the crew
        inputs = {
            "documents": ["uploads/6183_Text_to_LoRA_Instant_Tran.pdf"],  # Use an existing document
            "data_generation_model": crew_config["data_generation_model"],
            "embedding_model": crew_config["embedding_model"],
            "reranking_model": crew_config["reranking_model"],
            "enable_gpu_optimization": crew_config["enable_gpu_optimization"]
        }
        
        print("🚀 Running CrewAI workflow...")
        print("   This may take a few minutes...")
        
        # Execute the crew workflow
        result = crew_instance.crew().kickoff(inputs=inputs)
        
        print("✅ CrewAI workflow completed successfully!")
        print(f"Result type: {type(result)}")
        print(f"Result: {str(result)[:500]}...")  # Show first 500 characters
        
        if hasattr(result, 'raw'):
            print(f"Raw result: {str(result.raw)[:500]}...")
        
    except Exception as e:
        print(f"❌ CrewAI workflow execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print()
    print("🎉 Test completed!")
    print("=" * 50)

if __name__ == "__main__":
    test_crew_workflow()
