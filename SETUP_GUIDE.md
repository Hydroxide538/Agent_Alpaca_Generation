# Setup Guide - CrewAI Workflow Manager

This guide will help you set up and run the CrewAI Workflow Manager on your system.

## Prerequisites Check

Before starting, ensure you have:

1. **Python 3.10 or higher** installed
   - Download from [python.org](https://python.org)
   - Verify installation: `python --version` or `python3 --version`

2. **Git** (if cloning from repository)
   - Download from [git-scm.com](https://git-scm.com)

3. **Optional: Ollama** (for local models)
   - Download from [ollama.ai](https://ollama.ai)

## Installation Steps

### Step 1: Get the Code

If you have the code locally, navigate to the project directory:
```bash
cd local_crewai_workflow_for_synthetic_data_with_rag_and_llm_options
```

### Step 2: Install Dependencies

#### Option A: Automatic Installation (Recommended)
```bash
python start_server.py
```
This will automatically install all dependencies and start the server.

#### Option B: Manual Installation
```bash
pip install -r backend/requirements.txt
```

### Step 3: Configure Environment (Optional)

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your settings:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   OLLAMA_BASE_URL=http://localhost:11434
   PORT=8000
   ```

### Step 4: Start the Application

#### Windows Users:
Double-click `start_server.bat` or run:
```cmd
start_server.bat
```

#### Linux/Mac Users:
```bash
python start_server.py
```

#### With Custom Options:
```bash
python start_server.py --port 8001 --reload
```

### Step 5: Access the Application

Open your web browser and go to:
```
http://localhost:8000
```

## Ollama Setup (Optional)

If you want to use local models:

1. **Install Ollama**:
   - Visit [ollama.ai](https://ollama.ai)
   - Download and install for your OS

2. **Start Ollama**:
   ```bash
   ollama serve
   ```

3. **Pull Models**:
   ```bash
   ollama pull llama3.2
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

## Troubleshooting

### Common Issues

1. **"Python was not found"**:
   - Install Python from [python.org](https://python.org)
   - Make sure Python is added to your PATH
   - Try `python3` instead of `python`

2. **Port already in use**:
   ```bash
   python start_server.py --port 8001
   ```

3. **Permission denied**:
   - On Linux/Mac: `chmod +x start_server.py`
   - Run as administrator on Windows if needed

4. **Module not found errors**:
   ```bash
   pip install --upgrade pip
   pip install -r backend/requirements.txt
   ```

5. **Ollama connection failed**:
   - Make sure Ollama is running: `ollama serve`
   - Check if models are installed: `ollama list`

### Getting Help

1. Check the main README.md for detailed documentation
2. Look at the browser console for frontend errors
3. Check the terminal output for backend errors
4. Ensure all prerequisites are installed

## Development Setup

For development with auto-reload:
```bash
python start_server.py --reload
```

## Next Steps

1. Open the web interface at `http://localhost:8000`
2. Configure your models (OpenAI or Ollama)
3. Upload some documents (PDF, CSV, or TXT)
4. Run a test workflow
5. Explore the results and features

## Support

If you encounter issues:
- Check this guide first
- Review the main README.md
- Check the troubleshooting section
- Look for error messages in the terminal

Happy workflow management! 🚀
