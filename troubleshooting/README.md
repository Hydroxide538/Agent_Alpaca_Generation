# CrewAI Workflow Troubleshooting System

This directory contains the comprehensive troubleshooting system for the CrewAI Workflow Manager, including integrated test scripts, diagnostic tools, troubleshooting history, and a web-based wiki interface. The system provides both automated diagnostics and manual troubleshooting capabilities.

## Directory Structure

```
troubleshooting/
├── README.md                 # This file - System overview and usage guide
├── scripts/                  # All test scripts organized by category
│   ├── README.md            # Test scripts documentation
│   ├── test_api.py          # API health testing
│   ├── test_chromadb_fix.py # ChromaDB functionality testing
│   ├── test_crew_workflow.py # CrewAI workflow execution testing
│   ├── test_crewai_fix.py   # CrewAI specific fixes testing
│   ├── test_docker_ollama.py # Docker Ollama connection testing
│   ├── test_fix_verification.py # Comprehensive fix verification
│   ├── test_improved_alpaca.py # Improved Alpaca data generation testing
│   ├── test_json_parser_fix.py # JSON parsing robustness testing
│   ├── test_litellm_fix.py  # LiteLLM integration testing
│   ├── test_model_debug.py  # Detailed model debugging
│   ├── test_ollama_workflow.py # Ollama workflow configuration testing
│   ├── test_workflow_fix.py # Workflow manager fix verification
│   ├── test_workflow_integration.py # End-to-end workflow testing
│   ├── test_workflow_model.py # Workflow model setup testing
│   ├── debug_llm_manager.py # LLM manager comprehensive debugging
│   ├── dynamic_llm_evaluator.py # Dynamic LLM evaluation system
│   ├── enhanced_llm_evaluator.py # Enhanced LLM evaluation with thinking models
│   └── evaluate_llms.py     # LLM performance evaluation
├── wiki/                     # Web-based wiki interface for troubleshooting
│   ├── index.html           # Main wiki page with search and navigation
│   ├── wiki.js              # Wiki functionality and interactive features
│   ├── JSON_Parser_and_Scoring_Fixes_Summary.md # JSON parsing improvements
│   └── LLM_Evaluation_Troubleshooting_Summary.md # LLM evaluation fixes
└── templates/               # Issue templates and forms
    └── issue_template.md    # Standard issue report template
```

## Related Directories

```
knowledge/
└── wiki/                     # Comprehensive troubleshooting documentation
    ├── README.md            # Documentation overview and integration guide
    ├── ALPACA_IMPROVEMENTS_SUMMARY.md # Alpaca generation improvements
    ├── ALPACA_QUALITY_IMPROVEMENTS.md # Quality enhancements for data generation
    ├── CREWAI_LITELLM_FIX_SUMMARY.md # CrewAI and LiteLLM integration fixes
    ├── CRITICAL_ERRORS_FIX_SUMMARY.md # Critical system error resolutions
    ├── DEPENDENCY_RESOLUTION_SUMMARY.md # Dependency conflict resolutions
    ├── FINAL_FIX_SUMMARY.md # Comprehensive fix summary
    ├── LITELLM_INDEX_ERROR_FIX_SUMMARY.md # LiteLLM index error fixes
    ├── LLM_FAILURE_FIX_SUMMARY.md # LLM failure troubleshooting
    ├── OLLAMA_FIX_SUMMARY.md # Ollama integration fixes
    ├── PYDANTIC_DEPRECATION_FIX_SUMMARY.md # Pydantic compatibility fixes
    └── WORKFLOW_TROUBLESHOOTING_SUMMARY.md # General workflow troubleshooting
```

## System Integration

### 1. Web-Based Troubleshooting Interface
The troubleshooting system is fully integrated into the main CrewAI Workflow Manager interface:

#### Access Methods:
- **Main Interface**: Click the "Troubleshooting" button in the workflow control panel
- **Direct URL**: Navigate to the troubleshooting modal in the main application
- **API Endpoints**: Access diagnostic functions via REST API

#### Features:
- **Real-time Test Execution**: Run diagnostic tests with live output streaming
- **Interactive Results**: View structured test results with pass/fail indicators
- **Export Functionality**: Download complete diagnostic reports in JSON format
- **Visual Status Tracking**: Monitor test progress with color-coded status indicators

### 2. Backend Integration
The troubleshooting system is managed by the `TroubleshootingManager` class in `backend/troubleshooting.py`:

#### Core Capabilities:
- **Async Test Execution**: All tests run asynchronously with WebSocket progress updates
- **Comprehensive Logging**: Detailed logging with multiple severity levels
- **Error Handling**: Robust error handling with fallback mechanisms
- **Performance Metrics**: Timing and performance data collection

#### Integrated Components:
- **LLMManager Integration**: Direct access to LLM testing and management
- **WorkflowManager Integration**: Workflow execution testing and validation
- **WebSocket Manager**: Real-time progress updates and logging
- **Model Performance Tracking**: Integration with model scoring systems

## Available Diagnostic Tests

### Core System Tests

#### 1. API Health Check (`test_api.py`)
- **Purpose**: Tests basic API connectivity and health endpoints
- **Coverage**: Backend health, Ollama connectivity, model availability
- **Integration**: Fully integrated with real-time progress updates
- **Usage**: Validates system readiness before workflow execution

#### 2. Docker Ollama Test (`test_docker_ollama.py`)
- **Purpose**: Validates Docker Ollama container connectivity and functionality
- **Coverage**: Container status, model availability, embedding functionality
- **Integration**: Comprehensive Docker environment validation
- **Usage**: Essential for Docker-based Ollama deployments

### Model and LLM Tests

#### 3. Model Debug Test (`test_model_debug.py`)
- **Purpose**: Deep analysis of specific models including functionality testing
- **Coverage**: Model type detection, embedding capabilities, response quality
- **Integration**: Configurable model selection with real-time analysis
- **Usage**: Troubleshoot specific model issues and performance

#### 4. LLM Manager Debug (`debug_llm_manager.py`)
- **Purpose**: Comprehensive LLM manager debugging and diagnostics
- **Coverage**: Model initialization, performance tracking, error handling
- **Integration**: Full LLMManager system validation
- **Usage**: Diagnose LLM management system issues

#### 5. Enhanced LLM Evaluation (`enhanced_llm_evaluator.py`)
- **Purpose**: Advanced model evaluation with thinking model support
- **Coverage**: Multi-model comparison, performance benchmarking, quality assessment
- **Integration**: Document-based evaluation with structured scoring
- **Usage**: Compare model performance across different tasks

### Workflow Tests

#### 6. CrewAI Workflow Test (`test_crew_workflow.py`)
- **Purpose**: Tests complete CrewAI workflow execution
- **Coverage**: Agent initialization, task execution, hierarchical process management
- **Integration**: Full workflow validation with Manager Agent system
- **Usage**: Validate end-to-end CrewAI functionality

#### 7. Ollama Workflow Test (`test_ollama_workflow.py`)
- **Purpose**: Tests Ollama workflow configuration with dynamic model selection
- **Coverage**: Model discovery, configuration validation, workflow setup
- **Integration**: Dynamic model selection and Manager Agent integration
- **Usage**: Validate Ollama-specific workflow configurations

#### 8. Workflow Model Test (`test_workflow_model.py`)
- **Purpose**: Validates complete workflow configuration and model compatibility
- **Coverage**: Model setup, configuration validation, compatibility checks
- **Integration**: Comprehensive workflow configuration testing
- **Usage**: Pre-workflow validation of model and configuration setup

### System Integration Tests

#### 9. Workflow Integration Test (`test_workflow_integration.py`)
- **Purpose**: End-to-end workflow testing with all components
- **Coverage**: Complete system integration, data flow validation
- **Integration**: Full system validation from document upload to result generation
- **Usage**: Comprehensive system health validation

#### 10. Fix Verification Test (`test_fix_verification.py`)
- **Purpose**: Comprehensive test to verify all applied fixes are working
- **Coverage**: All major system fixes and improvements
- **Integration**: Historical fix validation and regression testing
- **Usage**: Ensure system stability after updates or changes

### Specialized Tests

#### 11. Improved Alpaca Test (`test_improved_alpaca.py`)
- **Purpose**: Test improved Alpaca format data generation with enhanced features
- **Coverage**: Structured fact extraction, concept extraction, Q&A generation
- **Integration**: Advanced data generation pipeline validation
- **Usage**: Validate synthetic data generation quality and functionality

#### 12. JSON Parser Fix Test (`test_json_parser_fix.py`)
- **Purpose**: Test JSON parsing robustness and error handling
- **Coverage**: Robust JSON extraction, error recovery, format validation
- **Integration**: Critical for LLM response parsing reliability
- **Usage**: Ensure reliable parsing of LLM-generated JSON responses

#### 13. ChromaDB Fix Test (`test_chromadb_fix.py`)
- **Purpose**: Test ChromaDB connection and vector database functionality
- **Coverage**: Vector database operations, embedding storage, retrieval
- **Integration**: RAG system validation and vector database health
- **Usage**: Validate vector database functionality for RAG operations

## Advanced Features

### 1. LLM Shootout Integration
The troubleshooting system integrates with the LLM Shootout Arena:

#### Competitive Evaluation:
- **Multi-Model Testing**: Compare multiple models simultaneously
- **Performance Benchmarking**: Standardized evaluation criteria
- **Real-time Competition**: Live progress tracking and scoring
- **Historical Performance**: Integration with model performance database

#### Evaluation Tasks:
- **Fact Extraction**: Structured fact extraction from documents
- **Concept Extraction**: Key concept identification and analysis
- **Analytical Q&A**: Complex question-answer generation
- **Quality Assessment**: Intelligent scoring and evaluation

### 2. Manager Agent System Integration
The troubleshooting system works with the Manager Agent for intelligent diagnostics:

#### Intelligent Model Selection:
- **Performance-Based Selection**: Use historical performance data
- **Dynamic Configuration**: Real-time model availability assessment
- **Strategy Testing**: Validate different selection strategies
- **Optimization Recommendations**: Suggest optimal configurations

#### Performance Tracking:
- **Historical Data**: Track model performance over time
- **Trend Analysis**: Identify performance patterns and issues
- **Predictive Diagnostics**: Anticipate potential problems
- **Optimization Guidance**: Recommend system improvements

### 3. Token Analysis Integration
Comprehensive token analysis and context window management:

#### Document Analysis:
- **Token Counting**: Accurate token counting for all document types
- **Context Window Analysis**: Model-specific context window utilization
- **Performance Impact**: Token count impact on processing time
- **Optimization Recommendations**: Suggest document chunking strategies

#### Model Compatibility:
- **Context Window Validation**: Ensure documents fit within model limits
- **Model Recommendations**: Suggest appropriate models for document sizes
- **Performance Optimization**: Optimize token usage for better performance

## Usage Guide

### Running Diagnostic Tests

#### From the Web Interface:
1. **Access Troubleshooting**: Click the "Troubleshooting" button in the main interface
2. **Select Tests**: Choose individual tests or "Run All Tests" for comprehensive diagnostics
3. **Monitor Progress**: Watch real-time logs and status updates
4. **Review Results**: Switch to "Test Results" view for structured analysis
5. **Export Data**: Use "Export Results" to download diagnostic reports

#### From Command Line:
```bash
# Individual test scripts (from project root)
python troubleshooting/scripts/test_api.py
python troubleshooting/scripts/test_crew_workflow.py
python troubleshooting/scripts/test_docker_ollama.py
python troubleshooting/scripts/test_model_debug.py
python troubleshooting/scripts/debug_llm_manager.py

# Enhanced evaluation
python troubleshooting/scripts/enhanced_llm_evaluator.py

# Comprehensive validation
python troubleshooting/scripts/test_fix_verification.py
```

#### Via API Endpoints:
```bash
# API health check
curl -X POST http://localhost:8000/troubleshoot/api-health

# Docker Ollama test
curl -X POST http://localhost:8000/troubleshoot/docker-ollama

# Model debug test
curl -X POST "http://localhost:8000/troubleshoot/model-debug?model_name=llama3.3:latest"

# Enhanced LLM evaluation
curl -X POST http://localhost:8000/troubleshoot/enhanced-llm-evaluation
```

### Using the Troubleshooting Wiki

#### Accessing Documentation:
1. **Web Interface**: Click "Troubleshooting Wiki" in the troubleshooting modal
2. **Search Functionality**: Use the search bar to find relevant solutions
3. **Category Filtering**: Filter by issue type (LLM, Docker, Model, Workflow)
4. **Historical Reference**: Browse chronological issue resolution history

#### Contributing to Documentation:
1. **Issue Reporting**: Use standardized issue templates
2. **Solution Documentation**: Document fixes and workarounds
3. **Knowledge Sharing**: Contribute to the community knowledge base
4. **Best Practices**: Share optimization strategies and configurations

## Common Troubleshooting Workflows

### 1. System Health Check
```
1. Run "API Health Check" to validate basic connectivity
2. Run "Docker Ollama Test" if using Docker deployment
3. Run "LLM Manager Debug" for comprehensive LLM system validation
4. Review results and address any failures
```

### 2. Model Performance Issues
```
1. Run "Model Debug Test" for specific problematic models
2. Use "Enhanced LLM Evaluation" for comprehensive model comparison
3. Check "LLM Shootout" results for performance benchmarks
4. Apply Manager Agent recommendations for optimal model selection
```

### 3. Workflow Execution Problems
```
1. Run "Workflow Model Test" to validate configuration
2. Use "CrewAI Workflow Test" for execution validation
3. Run "Ollama Workflow Test" for Ollama-specific issues
4. Check "Workflow Integration Test" for end-to-end validation
```

### 4. Data Generation Issues
```
1. Run "Improved Alpaca Test" to validate data generation pipeline
2. Use "JSON Parser Fix Test" to ensure robust response parsing
3. Check token analysis for document compatibility
4. Validate RAG system with "ChromaDB Fix Test"
```

### 5. Performance Optimization
```
1. Use "Enhanced LLM Evaluation" to identify best-performing models
2. Run "LLM Shootout" competitions for comparative analysis
3. Apply Manager Agent optimization recommendations
4. Monitor token usage and context window utilization
```

## Integration with Main System

### WebSocket Integration
- **Real-time Updates**: Live progress streaming during test execution
- **Structured Logging**: Color-coded log levels with timestamps
- **Progress Tracking**: Visual progress indicators and status updates
- **Error Reporting**: Detailed error information with stack traces

### Configuration Sharing
- **Unified Configuration**: Uses main system configuration for consistency
- **Dynamic Updates**: Reflects real-time configuration changes
- **Model Discovery**: Shares model availability with main system
- **Performance Data**: Integrates with system-wide performance tracking

### Result Storage
- **Integrated Storage**: Test results stored with main system results
- **Export Functionality**: Consistent export formats across the system
- **Historical Tracking**: Long-term storage of diagnostic data
- **Performance Metrics**: Integration with system performance monitoring

## Best Practices

### 1. Regular System Health Monitoring
- **Scheduled Diagnostics**: Run comprehensive tests regularly
- **Performance Baselines**: Establish and monitor performance baselines
- **Proactive Issue Detection**: Use predictive diagnostics to prevent problems
- **Documentation Updates**: Keep troubleshooting documentation current

### 2. Issue Resolution Workflow
- **Systematic Approach**: Follow structured troubleshooting workflows
- **Documentation**: Document all issues and resolutions
- **Knowledge Sharing**: Contribute solutions to the community knowledge base
- **Continuous Improvement**: Apply lessons learned to prevent future issues

### 3. Performance Optimization
- **Model Selection**: Use Manager Agent recommendations for optimal performance
- **Resource Monitoring**: Track system resource utilization
- **Configuration Tuning**: Optimize configurations based on diagnostic results
- **Capacity Planning**: Use performance data for system scaling decisions

### 4. Preventive Maintenance
- **Regular Updates**: Keep system components updated
- **Configuration Validation**: Regularly validate system configurations
- **Performance Monitoring**: Continuous monitoring of system performance
- **Backup and Recovery**: Maintain robust backup and recovery procedures

## Future Enhancements

### Planned Features:
1. **Automated Issue Detection** - Proactive problem identification using AI
2. **Performance Monitoring Dashboard** - Real-time system health visualization
3. **Predictive Diagnostics** - Machine learning-based issue prediction
4. **Advanced Analytics** - Comprehensive system performance analysis
5. **Remote Diagnostics** - Support for distributed system troubleshooting

### Integration Roadmap:
1. **AI-Powered Troubleshooting** - Use LLMs for intelligent issue resolution
2. **Community Knowledge Base** - Crowdsourced troubleshooting solutions
3. **Automated Fix Application** - Self-healing system capabilities
4. **Advanced Monitoring** - Comprehensive system observability
5. **Performance Optimization** - Automated performance tuning

The troubleshooting system is designed to be comprehensive, user-friendly, and continuously evolving to meet the needs of the CrewAI Workflow Manager community.
