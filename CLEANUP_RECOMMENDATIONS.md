# File Cleanup Recommendations

This document outlines files that can be safely removed from the project as they are no longer used or have been superseded by enhanced versions.

## Files Safe to Remove

### 1. Test Files (No longer referenced)
- `test_ollama_only.py` - Standalone test file, not imported anywhere
- `run_test.py` - Helper script for the above test file

### 2. Superseded Files
- `backend/improved_alpaca_generator.py` - **ALREADY REMOVED** - Replaced by `backend/enhanced_alpaca_generator.py`

### 3. Potentially Unused Files (Verify before removal)

#### Backend Files to Review:
- `backend/json_parser_fix.py` - Check if still used by enhanced_alpaca_generator
- `backend/safe_llm_wrapper.py` - Check if still used by CrewAI integration

#### Root Level Files:
- `OLLAMA_ONLY_SETUP.md` - Could be kept as it provides valuable setup documentation
- Files in `uploads/` directory - These are user-uploaded files, should be kept or cleaned based on user preference

## Files to Keep

### Essential Backend Files:
- `backend/app.py` - Main FastAPI application
- `backend/enhanced_alpaca_generator.py` - Enhanced Alpaca generation with Stanford Guide
- `backend/enhanced_document_manager.py` - Advanced document management
- `backend/llm_manager.py` - Core LLM management
- `backend/llm_shootout_manager.py` - LLM competition system
- `backend/manager_agent.py` - Intelligent LLM selection
- `backend/manager_scoring_system.py` - Model scoring system
- `backend/workflow_manager.py` - Workflow orchestration
- `backend/websocket_manager.py` - Real-time communication
- `backend/troubleshooting.py` - Diagnostic system
- `backend/rag_system.py` - RAG implementation
- `backend/token_counter.py` - Token analysis
- `backend/message_validator.py` - Message validation
- `backend/models.py` - Pydantic models

### Essential Frontend Files:
- `frontend/index.html` - Main interface
- `frontend/llm_shootout.html` - Shootout arena
- `frontend/script.js` - JavaScript functionality
- `frontend/styles.css` - Styling

### Essential Configuration:
- `pyproject.toml` - Project configuration
- `backend/requirements.txt` - Python dependencies
- `config/llms.yaml` - LLM registry
- `.env.example` - Environment template
- `.gitignore` - Git ignore rules

### Documentation (All should be kept):
- `README.md` - Main documentation (updated)
- `SETUP_GUIDE.md` - Installation guide
- `OLLAMA_ONLY_SETUP.md` - Ollama-specific setup
- `knowledge/` directory - Knowledge base
- `troubleshooting/` directory - Troubleshooting system

### CrewAI Source:
- `src/` directory - CrewAI workflow implementation

## Verification Steps Before Removal

1. **Check imports**: Search for any remaining imports of files to be removed
2. **Test functionality**: Ensure all core features work after removal
3. **Review documentation**: Update any documentation that references removed files

## Commands to Remove Safe Files

```bash
# Remove test files (if confirmed unused)
rm test_ollama_only.py
rm run_test.py

# Clean up any temporary or cache files
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

## Directory Structure After Cleanup

```
├── frontend/                 # Frontend files (keep all)
├── backend/                  # Backend API (keep essential files)
├── src/                      # CrewAI source code (keep all)
├── troubleshooting/          # Troubleshooting system (keep all)
├── knowledge/                # Knowledge management (keep all)
├── config/                   # Configuration files (keep all)
├── uploads/                  # User uploads (clean as needed)
├── collections/              # Document collections (keep)
├── results/                  # Workflow results (clean as needed)
├── vector_db/                # Vector database (clean as needed)
├── logs/                     # Application logs (clean as needed)
├── README.md                 # Updated main documentation
├── SETUP_GUIDE.md           # Installation guide
├── OLLAMA_ONLY_SETUP.md     # Ollama setup guide
├── pyproject.toml           # Project configuration
├── .env.example             # Environment template
├── .gitignore               # Git ignore rules
└── start_server.py          # Server startup script
```

## Notes

- The project is now significantly cleaner with the enhanced implementations
- All core functionality has been preserved and improved
- Documentation has been updated to reflect current state
- Enhanced document management and Alpaca generation provide better user experience
- The Stanford Guide implementation ensures high-quality dataset generation
