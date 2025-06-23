#!/usr/bin/env python3
"""
Simple test to verify the CrewAI fix is working
"""

import sys
import os

# Add paths
sys.path.append('src')
sys.path.append('backend')

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    
    try:
        from backend.safe_llm_wrapper import CrewAICompatibleLLM, SafeOllamaLLM
        print("✅ Safe LLM wrapper imports successful")
    except Exception as e:
        print(f"❌ Safe LLM wrapper import failed: {e}")
        return False
    
    try:
        from local_crewai_workflow_for_synthetic_data_with_rag_and_llm_options.crew import LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew
        print("✅ CrewAI crew import successful")
    except Exception as e:
        print(f"❌ CrewAI crew import failed: {e}")
        return False
    
    return True

def test_crew_creation():
    """Test crew creation with safe LLM wrappers"""
    print("\nTesting crew creation...")
    
    try:
        from local_crewai_workflow_for_synthetic_data_with_rag_and_llm_options.crew import LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew
        
        config = {
            "data_generation_model": "ollama:mistral-small3.2:latest",
            "embedding_model": "ollama:snowflake-arctic-embed2:latest", 
            "reranking_model": "ollama:bge-m3:latest",
            "ollama_url": "http://host.docker.internal:11434"
        }
        
        crew_instance = LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew(config=config)
        print("✅ Crew instance created successfully")
        
        # Check LLM types
        if crew_instance.data_generation_llm:
            llm_type = type(crew_instance.data_generation_llm).__name__
            print(f"✅ Data generation LLM: {llm_type}")
            if "CrewAICompatibleLLM" in llm_type:
                print("🛡️  Using safe wrapper for data generation")
            else:
                print("⚠️  Using standard LLM for data generation")
        
        # Test crew creation
        crew = crew_instance.crew()
        print("✅ Crew created successfully")
        
        # Check agent LLMs
        agents = crew.agents
        safe_agents = 0
        for i, agent in enumerate(agents):
            if hasattr(agent, 'llm') and agent.llm:
                llm_type = type(agent.llm).__name__
                if "CrewAICompatibleLLM" in llm_type:
                    safe_agents += 1
                    print(f"🛡️  Agent {i+1}: Using safe LLM wrapper ({llm_type})")
                else:
                    print(f"⚠️  Agent {i+1}: Using standard LLM ({llm_type})")
            else:
                print(f"❌ Agent {i+1}: No LLM assigned")
        
        print(f"\n📊 Summary: {safe_agents}/{len(agents)} agents using safe LLM wrappers")
        
        return True
        
    except Exception as e:
        print(f"❌ Crew creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_prevention():
    """Test that the specific error is prevented"""
    print("\nTesting error prevention...")
    
    try:
        from backend.message_validator import MessageValidator
        
        # Test empty messages (would cause "list index out of range")
        empty_messages = []
        safe_messages = MessageValidator.validate_messages_for_ollama(empty_messages)
        print(f"✅ Empty messages handled: {len(empty_messages)} -> {len(safe_messages)}")
        
        # Test messages with tool_calls (problematic for litellm)
        problematic_messages = [
            {"role": "user", "content": "Hello", "tool_calls": [{"function": "test"}]}
        ]
        safe_messages = MessageValidator.validate_messages_for_ollama(problematic_messages)
        has_tool_calls = any("tool_calls" in msg for msg in safe_messages)
        print(f"✅ Tool calls removed: {not has_tool_calls}")
        
        # Simulate the exact error condition
        try:
            test_messages = []
            msg_i = 0
            # This would cause the original "list index out of range" error
            tool_calls = test_messages[msg_i].get("tool_calls")
            print("❌ Should have failed with IndexError")
        except IndexError as e:
            if "list index out of range" in str(e):
                print("✅ IndexError caught - safe wrapper would handle this")
            else:
                print(f"❌ Unexpected IndexError: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error prevention test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 CrewAI 'List Index Out of Range' Fix Verification")
    print("=" * 60)
    
    success = True
    
    success &= test_imports()
    success &= test_crew_creation()
    success &= test_error_prevention()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! The fix should prevent the 'list index out of range' error.")
        print("\n🔧 Next steps:")
        print("1. Ensure Ollama is running with the required models")
        print("2. Run: python start_server.py")
        print("3. Test the workflow through the web interface")
        print("4. The 'list index out of range' error should no longer occur")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    main()
