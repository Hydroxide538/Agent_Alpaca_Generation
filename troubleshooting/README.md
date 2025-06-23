# CrewAI Workflow Troubleshooting System

This directory contains the enhanced troubleshooting system for the CrewAI Workflow Manager, including integrated test scripts, troubleshooting history, and a comprehensive wiki interface.

## Directory Structure

```
troubleshooting/
├── README.md                 # This file
├── scripts/                  # All test scripts organized by category
│   ├── README.md            # Test scripts documentation
│   ├── test_api.py          # API health testing
│   ├── test_chromadb_fix.py # ChromaDB functionality testing
│   ├── test_crew_workflow.py # CrewAI workflow execution testing
│   ├── test_crewai_fix.py   # CrewAI specific fixes testing
│   ├── test_docker_ollama.py # Docker Ollama connection testing
│   ├── test_fix_verification.py # Comprehensive fix verification
│   ├── test_improved_alpaca.py # Improved Alpaca data generation testing
│   ├── test_litellm_fix.py  # LiteLLM integration testing
│   ├── test_model_debug.py  # Detailed model debugging
│   ├── test_ollama_workflow.py # Ollama workflow configuration testing
│   ├── test_workflow_fix.py # Workflow manager fix verification
│   ├── test_workflow_integration.py # End-to-end workflow testing
│   └── test_workflow_model.py # Workflow model setup testing
├── wiki/                     # Wiki interface for troubleshooting
│   ├── index.html           # Main wiki page
│   └── wiki.js              # Wiki functionality
└── templates/               # Issue templates and forms
    └── issue_template.md    # Standard issue report template
```

## Related Directories

```
knowledge/
└── wiki/                     # Comprehensive troubleshooting documentation
    ├── README.md            # Documentation overview
    ├── ALPACA_IMPROVEMENTS_SUMMARY.md
    ├── CREWAI_LITELLM_FIX_SUMMARY.md
    ├── CRITICAL_ERRORS_FIX_SUMMARY.md
    ├── DEPENDENCY_RESOLUTION_SUMMARY.md
    ├── FINAL_FIX_SUMMARY.md
    ├── LITELLM_INDEX_ERROR_FIX_SUMMARY.md
    ├── LLM_FAILURE_FIX_SUMMARY.md
    ├── OLLAMA_FIX_SUMMARY.md
    ├── PYDANTIC_DEPRECATION_FIX_SUMMARY.md
    └── WORKFLOW_TROUBLESHOOTING_SUMMARY.md
```

## Features

### 1. Integrated Test Scripts
The troubleshooting system now includes all test scripts directly in the main interface:

#### Available Tests:
- **API Health Check** (`test_api.py`) - Tests basic API connectivity
- **Docker Ollama Test** (`test_docker_ollama.py`) - Tests Docker Ollama connection
- **Model Debug Test** (`test_model_debug.py`) - Detailed model debugging
- **Workflow Model Test** (`test_workflow_model.py`) - Tests workflow model setup
- **LLM Manager Debug** (`debug_llm_manager.py`) - Comprehensive LLM debugging
- **CrewAI Workflow Test** (`test_crew_workflow.py`) - Tests CrewAI execution
- **Ollama Workflow Test** (`test_ollama_workflow.py`) - Tests Ollama workflow configuration

#### New Integrated Tests:
- **CrewAI Workflow Test** - Comprehensive CrewAI workflow execution testing
- **Ollama Workflow Test** - Dynamic model selection and configuration testing

### 2. Troubleshooting Wiki
A comprehensive wiki interface that provides:

- **Issue History Timeline** - Chronological view of all resolved issues
- **Common Issues Database** - Searchable database of frequent problems
- **Test Scripts Integration** - Direct access to run diagnostic tests
- **Quick Fixes Guide** - Step-by-step solutions for common problems
- **Search Functionality** - Find relevant solutions quickly
- **Category Filtering** - Filter by issue type (LLM, Docker, Model, Workflow)

#### Accessing the Wiki:
1. Open the main troubleshooting interface
2. Click the "Troubleshooting Wiki" button
3. Browse issues by category or use the search function

### 3. Troubleshooting History
All troubleshooting summaries are now organized in the `history/` directory:

- **alpaca_improvements.md** - System improvements and enhancements
- **llm_failure_fix.md** - LLM failure resolution
- **ollama_fix.md** - Ollama configuration fixes
- **workflow_troubleshooting.md** - Workflow execution issues

### 4. Issue Reporting System
Standardized issue reporting with:

- **Issue Templates** - Structured forms for consistent reporting
- **Automatic Metadata Collection** - System info and configuration capture
- **Test Result Integration** - Attach diagnostic test results
- **Export Functionality** - Export troubleshooting sessions

## How to Use

### Running Diagnostic Tests

#### From the Main Interface:
1. Open the CrewAI Workflow Manager
2. Click the "Troubleshooting" button
3. Select individual tests or "Run All Tests"
4. View real-time logs and results

#### From Command Line:
```bash
# Individual test scripts (from project root)
python troubleshooting/scripts/test_api.py
python troubleshooting/scripts/test_crew_workflow.py
python troubleshooting/scripts/test_workflow_fix.py
python troubleshooting/scripts/test_docker_ollama.py
python troubleshooting/scripts/test_model_debug.py
python troubleshooting/scripts/test_ollama_workflow.py
python troubleshooting/scripts/test_workflow_model.py

# Additional test scripts
python troubleshooting/scripts/test_chromadb_fix.py
python troubleshooting/scripts/test_crewai_fix.py
python troubleshooting/scripts/test_fix_verification.py
python troubleshooting/scripts/test_improved_alpaca.py
python troubleshooting/scripts/test_litellm_fix.py
python troubleshooting/scripts/test_workflow_integration.py

# Quick workflow fix verification
python troubleshooting/scripts/test_workflow_fix.py
```

### Using the Wiki

#### Searching for Solutions:
1. Open the wiki interface
2. Use the search bar to find relevant issues
3. Filter by category (LLM, Docker, Model, Workflow)
4. Click on issues to view detailed solutions

#### Reporting New Issues:
1. Click "Report Issue" in the wiki
2. Fill out the issue template
3. Attach relevant test results
4. Submit for tracking

### Exporting Troubleshooting Data

#### From the Interface:
1. Run diagnostic tests
2. Click "Export Results"
3. Save the JSON file with all test data and logs

#### Manual Export:
- Test results are saved in the `results/` directory
- Logs are available in the main interface
- Wiki data can be exported from the wiki interface

## Common Troubleshooting Workflows

### 1. LLM Failed Error
```
1. Run "CrewAI Workflow Test"
2. Check if models are properly initialized
3. Verify Ollama connection with "Docker Ollama Test"
4. Run "LLM Manager Debug" for detailed analysis
5. Check wiki for "LLM Failed Error" solutions
```

### 2. Model Not Found
```
1. Run "Model Debug Test" with the specific model
2. Check available models with "API Health Check"
3. Verify model names match exactly
4. Use "Ollama Workflow Test" for configuration validation
```

### 3. Docker Connection Issues
```
1. Run "Docker Ollama Test"
2. Verify Ollama URL (host.docker.internal:11434)
3. Check Docker network configuration
4. Test with localhost:11434 as fallback
```

### 4. Workflow Execution Problems
```
1. Run "Workflow Model Test" to verify configuration
2. Use "CrewAI Workflow Test" for execution testing
3. Check "LLM Manager Debug" for model issues
4. Review workflow troubleshooting history
```

## Test Script Details

### API Health Check
- Tests basic server connectivity
- Validates Ollama endpoints
- Checks model availability
- **Integration**: Fully integrated in troubleshooting interface

### CrewAI Workflow Test
- Tests complete CrewAI workflow execution
- Validates agent initialization
- Checks LLM assignments
- Tests quick workflow execution
- **Integration**: New integrated test

### Ollama Workflow Test
- Dynamic model selection testing
- Configuration validation
- OpenAI model rejection testing
- CrewAI initialization without API key
- **Integration**: New integrated test

### Docker Ollama Test
- Docker network connectivity
- Model availability testing
- Embedding model functionality
- Connection timeout handling
- **Integration**: Fully integrated

### Model Debug Test
- Detailed model analysis
- Model type detection (embedding vs generation)
- Functionality testing
- Performance metrics
- **Integration**: Fully integrated with model selection

### LLM Manager Debug
- Comprehensive LLM testing
- Model initialization validation
- Error handling verification
- Response time measurement
- **Integration**: Fully integrated

### Workflow Model Test
- Configuration validation
- Model compatibility testing
- Setup verification
- **Integration**: Fully integrated

## Best Practices

### 1. Regular Testing
- Run diagnostic tests before major workflows
- Use "Run All Tests" for comprehensive system checks
- Monitor test results for performance trends

### 2. Issue Documentation
- Use the issue template for consistent reporting
- Include test results with issue reports
- Document workarounds and solutions

### 3. Wiki Maintenance
- Search existing issues before reporting new ones
- Update solutions when better approaches are found
- Contribute to the knowledge base

### 4. Preventive Measures
- Regular model availability checks
- Docker connection validation
- Configuration backup and validation

## Integration with Main System

The troubleshooting system is fully integrated with the main CrewAI Workflow Manager:

- **WebSocket Integration** - Real-time test logging
- **Configuration Sharing** - Uses main system configuration
- **Result Storage** - Integrated with main results system
- **Model Management** - Shares model detection and validation

## Future Enhancements

### Planned Features:
1. **Automated Issue Detection** - Proactive problem identification
2. **Performance Monitoring** - Continuous system health tracking
3. **Solution Recommendations** - AI-powered troubleshooting suggestions
4. **Integration Testing** - End-to-end workflow validation
5. **Remote Diagnostics** - Support for distributed deployments

### Contributing:
- Add new test scripts to the main directory
- Update the troubleshooting manager to integrate new tests
- Document solutions in the wiki
- Improve issue templates and reporting

## Support

For additional support:
1. Check the troubleshooting wiki for existing solutions
2. Run comprehensive diagnostic tests
3. Export test results for analysis
4. Report new issues using the standard template
5. Consult the troubleshooting history for similar problems

The troubleshooting system is designed to be self-service and comprehensive, providing users with the tools needed to diagnose and resolve issues independently while maintaining a knowledge base for the community.
