# Troubleshooting Test Scripts

This directory contains comprehensive test scripts for debugging and verifying the CrewAI Workflow Manager system. These scripts provide both standalone testing capabilities and integration with the web-based troubleshooting interface. All scripts are designed to work with the current system architecture including the Manager Agent system, LLM Shootout Arena, and advanced diagnostic capabilities.

## Script Categories

### Core System Tests

#### API and Connection Tests
- **test_api.py** - Basic API health testing for localhost and Docker Ollama connections
  - Tests backend health endpoints
  - Validates Ollama server connectivity
  - Checks model availability and accessibility
  - **Integration Status**: ✅ Fully integrated with web interface

- **test_docker_ollama.py** - Comprehensive Docker Ollama connection testing
  - Validates Docker container status and connectivity
  - Tests model availability within Docker environment
  - Verifies embedding model functionality
  - Includes connection timeout handling and error recovery
  - **Integration Status**: ✅ Fully integrated with web interface

### Database and Storage Tests

- **test_chromadb_fix.py** - ChromaDB connection and vector database functionality testing
  - Tests vector database initialization and connectivity
  - Validates embedding storage and retrieval operations
  - Checks ChromaDB configuration and performance
  - Essential for RAG system functionality validation
  - **Integration Status**: ⚪ Standalone (can be called from interface)

### CrewAI and Workflow Tests

- **test_crew_workflow.py** - Comprehensive CrewAI workflow execution testing
  - Tests complete CrewAI workflow with proper model configuration
  - Validates Manager Agent system integration
  - Tests hierarchical process management
  - Includes agent initialization and task execution validation
  - **Integration Status**: ✅ Fully integrated with web interface

- **test_crewai_fix.py** - CrewAI specific fixes and configuration testing
  - Tests CrewAI framework integration fixes
  - Validates agent and task configuration
  - Checks CrewAI-specific error handling
  - **Integration Status**: ⚪ Standalone (can be called from interface)

- **test_workflow_fix.py** - Quick workflow manager fix verification
  - Rapid test to verify workflow manager fixes resolve LLM failures
  - Validates workflow execution pipeline
  - Tests error handling and recovery mechanisms
  - **Integration Status**: ⚪ Standalone (can be called from interface)

- **test_workflow_integration.py** - End-to-end workflow integration testing
  - Tests complete workflow integration from start to finish
  - Validates data flow between all system components
  - Comprehensive system integration validation
  - **Integration Status**: ⚪ Standalone (can be called from interface)

- **test_workflow_model.py** - Workflow model setup and configuration testing
  - Tests workflow model configuration and compatibility
  - Validates model setup procedures
  - Checks model-workflow integration
  - **Integration Status**: ✅ Fully integrated with web interface

- **test_ollama_workflow.py** - Ollama workflow configuration with dynamic model selection
  - Tests Ollama-specific workflow configurations
  - Validates dynamic model selection and discovery
  - Tests Manager Agent integration with Ollama models
  - Includes OpenAI model rejection testing for Ollama workflows
  - **Integration Status**: ✅ Fully integrated with web interface

### LLM and Model Tests

- **test_model_debug.py** - Detailed model debugging for specific models
  - Deep analysis of individual models (default: bge-m3:latest)
  - Model type detection (embedding vs generation)
  - Functionality testing and performance metrics
  - Configurable model selection for targeted debugging
  - **Integration Status**: ✅ Fully integrated with web interface

- **debug_llm_manager.py** - Comprehensive LLM manager debugging and diagnostics
  - Full LLMManager system validation and testing
  - Model initialization and performance tracking validation
  - Error handling verification and response time measurement
  - Integration with model performance database
  - **Integration Status**: ✅ Fully integrated with web interface

- **test_litellm_fix.py** - LiteLLM integration fixes and configuration testing
  - Tests LiteLLM framework integration
  - Validates LiteLLM-specific configurations and fixes
  - Checks compatibility with current system architecture
  - **Integration Status**: ⚪ Standalone (can be called from interface)

### Advanced Evaluation Tests

- **enhanced_llm_evaluator.py** - Enhanced LLM evaluation with thinking model support
  - Advanced model evaluation with structured scoring
  - Support for thinking models and complex reasoning tasks
  - Document-based evaluation with comprehensive metrics
  - Integration with Manager Agent scoring system
  - **Integration Status**: ✅ Fully integrated with web interface

- **dynamic_llm_evaluator.py** - Dynamic LLM evaluation system
  - Real-time model evaluation and comparison
  - Dynamic model discovery and testing
  - Performance benchmarking across multiple criteria
  - **Integration Status**: ⚪ Standalone (can be called from interface)

- **evaluate_llms.py** - General LLM performance evaluation
  - Standardized LLM performance testing
  - Multi-model comparison and benchmarking
  - Performance metrics collection and analysis
  - **Integration Status**: ⚪ Standalone (can be called from interface)

### Data Generation Tests

- **test_improved_alpaca.py** - Improved Alpaca format data generation testing
  - Tests enhanced Alpaca data generation with structured extraction
  - Validates fact extraction, concept extraction, and Q&A generation
  - Tests quality improvements and advanced features
  - Integration with ImprovedAlpacaGenerator system
  - **Integration Status**: ⚪ Standalone (can be called from interface)

- **test_json_parser_fix.py** - JSON parsing robustness and error handling testing
  - Tests robust JSON extraction from LLM responses
  - Validates error recovery and format validation
  - Critical for reliable LLM response parsing
  - Tests RobustJSONParser functionality
  - **Integration Status**: ⚪ Standalone (can be called from interface)

### Verification Tests

- **test_fix_verification.py** - Comprehensive fix verification testing
  - Tests to verify all applied system fixes are working correctly
  - Regression testing for historical fixes
  - Comprehensive system stability validation
  - Ensures system integrity after updates or changes
  - **Integration Status**: ⚪ Standalone (can be called from interface)

## Integration Status Legend

- ✅ **Fully Integrated** - Tests that are completely integrated with the web troubleshooting interface
- ⚪ **Standalone** - Tests that run independently but can be called from the interface
- 🔄 **In Progress** - Tests currently being integrated
- ❌ **Not Integrated** - Tests that are standalone only

## Current System Integration

### Web Interface Integration
The following tests are fully integrated with the main troubleshooting interface:

1. **API Health Check** (`test_api.py`)
2. **Docker Ollama Test** (`test_docker_ollama.py`)
3. **Model Debug Test** (`test_model_debug.py`)
4. **Workflow Model Test** (`test_workflow_model.py`)
5. **LLM Manager Debug** (`debug_llm_manager.py`)
6. **CrewAI Workflow Test** (`test_crew_workflow.py`)
7. **Ollama Workflow Test** (`test_ollama_workflow.py`)
8. **Enhanced LLM Evaluation** (`enhanced_llm_evaluator.py`)

### Backend Integration
All tests integrate with the `TroubleshootingManager` class in `backend/troubleshooting.py`:

#### Key Integration Features:
- **Async Execution**: All tests run asynchronously with WebSocket progress updates
- **Real-time Logging**: Live log streaming with color-coded severity levels
- **Structured Results**: Standardized result formats with pass/fail indicators
- **Error Handling**: Comprehensive error handling with detailed stack traces
- **Performance Metrics**: Timing and performance data collection

#### WebSocket Integration:
- **Progress Updates**: Real-time progress streaming during test execution
- **Log Broadcasting**: Live log messages with timestamps and severity levels
- **Status Updates**: Visual progress indicators and completion status
- **Error Reporting**: Detailed error information with context

## Usage Instructions

### Running Tests from Web Interface

1. **Access Troubleshooting Interface**:
   - Click the "Troubleshooting" button in the main workflow control panel
   - The troubleshooting modal will open with all available tests

2. **Individual Test Execution**:
   - Click on specific test buttons (e.g., "API Health Check", "Model Debug")
   - Monitor real-time progress in the "Live Logs" view
   - Switch to "Test Results" view for structured analysis

3. **Batch Test Execution**:
   - Click "Run All Tests" to execute comprehensive diagnostics
   - Monitor overall progress and individual test status
   - Review consolidated results and export if needed

4. **Test Configuration**:
   - Configure test parameters (e.g., model names for debugging)
   - Use "Refresh Models" to update available model lists
   - Customize test settings based on your environment

### Running Tests from Command Line

```bash
# Navigate to project root directory
cd /path/to/local_crewai_workflow_for_synthetic_data_with_rag_and_llm_options

# Core system tests
python troubleshooting/scripts/test_api.py
python troubleshooting/scripts/test_docker_ollama.py

# Workflow tests
python troubleshooting/scripts/test_crew_workflow.py
python troubleshooting/scripts/test_workflow_model.py
python troubleshooting/scripts/test_ollama_workflow.py

# Model and LLM tests
python troubleshooting/scripts/test_model_debug.py
python troubleshooting/scripts/debug_llm_manager.py

# Advanced evaluation
python troubleshooting/scripts/enhanced_llm_evaluator.py
python troubleshooting/scripts/dynamic_llm_evaluator.py

# Data generation tests
python troubleshooting/scripts/test_improved_alpaca.py
python troubleshooting/scripts/test_json_parser_fix.py

# Comprehensive validation
python troubleshooting/scripts/test_fix_verification.py
```

### Running Tests via API

```bash
# API health check
curl -X POST http://localhost:8000/troubleshoot/api-health

# Docker Ollama test
curl -X POST http://localhost:8000/troubleshoot/docker-ollama

# Model debug test (with parameter)
curl -X POST "http://localhost:8000/troubleshoot/model-debug?model_name=llama3.3:latest"

# Workflow model test (with JSON config)
curl -X POST http://localhost:8000/troubleshoot/workflow-model \
  -H "Content-Type: application/json" \
  -d '{"manager_model": "ollama:llama3.3:latest", "ollama_url": "http://host.docker.internal:11434"}'

# Enhanced LLM evaluation
curl -X POST http://localhost:8000/troubleshoot/enhanced-llm-evaluation
```

## Test Categories and Use Cases

### System Health Validation
**Use Case**: Regular system health monitoring and pre-workflow validation
**Tests**: `test_api.py`, `test_docker_ollama.py`, `debug_llm_manager.py`
**Frequency**: Before major workflows, daily health checks

### Model Performance Analysis
**Use Case**: Model selection, performance optimization, troubleshooting model issues
**Tests**: `test_model_debug.py`, `enhanced_llm_evaluator.py`, `evaluate_llms.py`
**Frequency**: When adding new models, performance issues, optimization cycles

### Workflow Validation
**Use Case**: Workflow configuration validation, execution troubleshooting
**Tests**: `test_crew_workflow.py`, `test_workflow_model.py`, `test_ollama_workflow.py`
**Frequency**: Before workflow execution, after configuration changes

### Data Generation Quality
**Use Case**: Synthetic data quality validation, generation pipeline testing
**Tests**: `test_improved_alpaca.py`, `test_json_parser_fix.py`
**Frequency**: After data generation improvements, quality issues

### Integration Testing
**Use Case**: End-to-end system validation, regression testing
**Tests**: `test_workflow_integration.py`, `test_fix_verification.py`
**Frequency**: After system updates, major changes, release validation

### Specialized Diagnostics
**Use Case**: Specific component testing, targeted troubleshooting
**Tests**: `test_chromadb_fix.py`, `test_litellm_fix.py`, `test_crewai_fix.py`
**Frequency**: Component-specific issues, integration problems

## Advanced Features

### Manager Agent Integration
Many tests now integrate with the Manager Agent system:

- **Intelligent Model Selection**: Tests validate Manager Agent's model selection algorithms
- **Performance Tracking**: Tests contribute to and validate performance tracking systems
- **Strategy Validation**: Tests verify different selection strategies (performance-based, balanced, etc.)
- **Optimization Recommendations**: Tests provide data for system optimization

### LLM Shootout Integration
Several tests integrate with the LLM Shootout Arena:

- **Competitive Evaluation**: Tests contribute to model competition data
- **Performance Benchmarking**: Standardized evaluation criteria across tests
- **Real-time Competition**: Tests can trigger or participate in model competitions
- **Historical Performance**: Tests access and contribute to performance history

### Token Analysis Integration
Tests now include comprehensive token analysis:

- **Document Compatibility**: Validate documents fit within model context windows
- **Performance Impact**: Analyze token count impact on processing time
- **Optimization Recommendations**: Suggest optimal configurations based on token analysis

## Best Practices

### Test Execution Strategy
1. **Start with Core Tests**: Begin with `test_api.py` and `test_docker_ollama.py`
2. **Progressive Validation**: Move from basic connectivity to complex workflow testing
3. **Targeted Debugging**: Use specific tests for known issues or components
4. **Comprehensive Validation**: Use `test_fix_verification.py` for overall system health

### Performance Optimization
1. **Regular Benchmarking**: Use evaluation tests to establish performance baselines
2. **Model Selection**: Use Manager Agent recommendations based on test results
3. **Configuration Tuning**: Apply test results to optimize system configurations
4. **Continuous Monitoring**: Regular test execution for proactive issue detection

### Issue Resolution Workflow
1. **Systematic Approach**: Follow structured troubleshooting workflows
2. **Documentation**: Document test results and resolutions
3. **Knowledge Sharing**: Contribute findings to the troubleshooting wiki
4. **Continuous Improvement**: Apply lessons learned to prevent future issues

## Future Enhancements

### Planned Test Improvements
1. **Automated Test Scheduling**: Regular automated test execution
2. **Predictive Testing**: AI-powered test selection based on system state
3. **Performance Regression Detection**: Automated detection of performance degradation
4. **Integration Test Expansion**: More comprehensive end-to-end testing

### New Test Categories
1. **Security Testing**: Comprehensive security validation tests
2. **Load Testing**: System performance under various load conditions
3. **Compatibility Testing**: Cross-platform and version compatibility validation
4. **User Experience Testing**: Automated UI and UX validation

### Enhanced Integration
1. **Real-time Monitoring**: Continuous background testing and monitoring
2. **Intelligent Alerting**: Smart alerts based on test results and patterns
3. **Automated Remediation**: Self-healing capabilities based on test outcomes
4. **Community Testing**: Crowdsourced testing and validation

The troubleshooting test scripts provide comprehensive coverage of the CrewAI Workflow Manager system, enabling effective diagnosis, validation, and optimization of all system components.
