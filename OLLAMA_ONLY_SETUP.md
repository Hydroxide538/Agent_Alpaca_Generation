# Ollama-Only Setup Guide

This guide explains how to run the CrewAI Workflow Manager using only Ollama models, without requiring an OpenAI API key.

## Problem Fixed

The original error `'OPENAI_API_KEY'` occurred because:
1. CrewAI tools (PDFSearchTool, CSVSearchTool, TXTSearchTool) expected OpenAI embeddings by default
2. The system tried to validate OpenAI API keys even when using Ollama models
3. Some libraries required the OPENAI_API_KEY environment variable to exist

## Solution Implemented

### 1. Updated LLM Creation Logic
- Modified `src/local_crewai_workflow_for_synthetic_data_with_rag_and_llm_options/crew.py`
- Added fallback logic to use Ollama models when OpenAI API keys are missing
- Graceful handling of missing or placeholder API keys

### 2. Updated Safe LLM Wrapper
- Modified `backend/safe_llm_wrapper.py`
- Added fallback to Ollama when OpenAI API key is not provided
- Prevents crashes when API keys are missing

### 3. Fixed Tool Dependencies
- Simplified document processing tools to avoid OpenAI dependencies
- Only use basic FileReadTool when embedding models are not properly configured

### 4. Environment Variable Handling
- Added placeholder OPENAI_API_KEY to prevent library failures
- Updated startup script to set placeholder automatically

## Files Modified

1. **src/local_crewai_workflow_for_synthetic_data_with_rag_and_llm_options/crew.py**
   - Added fallback logic for missing API keys
   - Simplified tool initialization

2. **backend/safe_llm_wrapper.py**
   - Added OpenAI fallback to Ollama
   - Graceful handling of missing API keys

3. **start_server.py**
   - Added automatic placeholder API key setting

4. **Created new files:**
   - `.env` - Environment configuration template
   - `.env.example` - Example configuration
   - `test_ollama_only.py` - Test script for Ollama-only setup
   - `run_test.py` - Helper script to run tests
   - `OLLAMA_ONLY_SETUP.md` - This guide

## How to Use

### Option 1: Start the Server (Recommended)
```bash
python start_server.py
```

The server will:
- Automatically set placeholder environment variables
- Start the web interface at http://localhost:8000
- Work with Ollama models by default

### Option 2: Test the Configuration
```bash
python run_test.py
```

This will run tests to verify that:
- Ollama-only configuration works
- OpenAI models fall back to Ollama when no API key is provided

### Option 3: Manual Environment Setup
If you need to set the environment manually:

```bash
export OPENAI_API_KEY="placeholder_key_not_used"
python start_server.py
```

## Configuration

### Default Ollama Models
The system defaults to these Ollama models:
- **Manager Model**: `ollama:llama3.3:latest`
- **Data Generation**: `ollama:llama3.3:latest`
- **Embedding**: `ollama:bge-m3:latest`

### Using OpenAI Models (Optional)
If you want to use OpenAI models:
1. Get an API key from https://platform.openai.com/api-keys
2. Enter it in the web interface under "OpenAI API Key"
3. Select OpenAI models in the model dropdowns

The system will automatically switch to OpenAI when a valid key is provided.

## Troubleshooting

### If you still get OPENAI_API_KEY errors:
1. Make sure you're using the updated `start_server.py`
2. Try running: `export OPENAI_API_KEY="placeholder_key_not_used"`
3. Restart your terminal/environment

### If Ollama models don't work:
1. Make sure Ollama is running: `ollama serve`
2. Check if models are installed: `ollama list`
3. Pull required models:
   ```bash
   ollama pull llama3.3:latest
   ollama pull bge-m3:latest
   ```

### If the web interface doesn't load:
1. Check if port 8000 is available
2. Try a different port: `python start_server.py --port 8001`
3. Check firewall settings

## Benefits of This Setup

1. **No API Costs**: Use free local Ollama models
2. **Privacy**: All processing happens locally
3. **Offline Capable**: Works without internet connection
4. **Flexible**: Can still use OpenAI models when API key is provided
5. **Robust**: Graceful fallbacks prevent crashes

## Next Steps

1. Start the server: `python start_server.py`
2. Open http://localhost:8000 in your browser
3. Upload some documents (PDF, CSV, or TXT)
4. Configure your workflow settings
5. Run your first workflow!

The system is now ready to work with Ollama-only configuration while maintaining the option to use OpenAI models when needed.
