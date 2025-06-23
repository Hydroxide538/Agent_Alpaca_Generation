# Troubleshooting Test Scripts

This directory contains all the test scripts for debugging and verifying the CrewAI workflow system. These scripts have been organized from the root directory to provide better structure and integration with the troubleshooting interface.

## Available Test Scripts

### API and Connection Tests
- **test_api.py** - Basic API health testing for localhost and Docker Ollama connections
- **test_docker_ollama.py** - Test Ollama connection from Docker environment with detailed debugging

### Database Tests
- **test_chromadb_fix.py** - Test ChromaDB connection and vector database functionality

### CrewAI and Workflow Tests
- **test_crew_workflow.py** - Comprehensive CrewAI workflow execution testing with proper model configuration
- **test_crewai_fix.py** - Test CrewAI specific fixes and configurations
- **test_workflow_fix.py** - Quick test to verify workflow manager fix resolves LLM failures
- **test_workflow_integration.py** - Test complete workflow integration and end-to-end functionality
- **test_workflow_model.py** - Test workflow model setup and configuration
- **test_ollama_workflow.py** - Test Ollama workflow configuration with dynamic model selection

### LLM and Model Tests
- **test_litellm_fix.py** - Test LiteLLM integration fixes and configurations
- **test_model_debug.py** - Detailed model debugging for specific models like bge-m3
- **debug_llm_manager.py** - Comprehensive LLM manager debugging and diagnostics

### Data Generation Tests
- **test_improved_alpaca.py** - Test improved Alpaca format data generation with enhanced features

### Verification Tests
- **test_fix_verification.py** - Comprehensive test to verify all applied fixes are working correctly

## Usage

### Running Individual Tests
```bash
# From the project root directory
python troubleshooting/scripts/test_api.py
python troubleshooting/scripts/test_docker_ollama.py
# ... etc
```

### Integration with Troubleshooting Interface
These scripts are integrated with the main troubleshooting interface and can be executed through:
1. The web-based troubleshooting interface
2. The troubleshooting wiki system
3. Direct API calls to the troubleshooting endpoints

### Test Categories
- **API Tests** - Connection and health checks
- **Database Tests** - Vector database functionality
- **Workflow Tests** - CrewAI workflow execution
- **Model Tests** - LLM and embedding model functionality
- **Docker Tests** - Docker environment connectivity
- **Verification Tests** - Overall system verification
- **Alpaca Tests** - Data generation specific tests
- **LLM Tests** - Language model integration

## Test Integration Status
- ✅ **Integrated** - Tests that are fully integrated with the troubleshooting interface
- ⚪ **Standalone** - Tests that run independently but can be called from the interface

## Documentation
For detailed documentation about fixes and troubleshooting history, see the knowledge/wiki directory which contains comprehensive markdown documentation for all resolved issues.
