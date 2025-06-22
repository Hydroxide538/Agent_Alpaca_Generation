# CrewAI Workflow Manager for Synthetic Data & RAG

A comprehensive web-based interface for managing CrewAI workflows focused on synthetic data generation and RAG (Retrieval-Augmented Generation) implementation. This project provides a modern frontend and robust backend to leverage CrewAI's multi-agent capabilities with support for both OpenAI and Ollama models.

## 🚀 Features

- **Web-based Interface**: Modern, responsive frontend for easy workflow management
- **Multi-LLM Support**: Compatible with OpenAI GPT models and local Ollama models
- **Docker Integration**: Full support for Ollama running in Docker containers
- **Dynamic Configuration**: Real-time model loading and URL configuration
- **Document Processing**: Upload and process PDF, CSV, and TXT files
- **Real-time Updates**: WebSocket-based live progress tracking
- **GPU Optimization**: Built-in support for dual RTX 4090 GPU setups
- **Flexible Workflows**: Run full workflows or individual components (data generation, RAG implementation)
- **Model Testing**: Built-in model connectivity and performance testing
- **Auto-Model Selection**: Intelligent model recommendations based on available models
- **Results Management**: Download and view workflow results
- **Comprehensive Troubleshooting**: Built-in diagnostic tools for system health monitoring
- **Real-time Diagnostics**: Live troubleshooting with detailed error analysis
- **Export Diagnostics**: Download complete diagnostic reports for support

## 🏗️ Architecture

### Frontend
- **HTML/CSS/JavaScript**: Modern responsive interface
- **Bootstrap 5**: UI framework for consistent styling
- **WebSocket Client**: Real-time communication with backend
- **Local Storage**: Configuration persistence
- **Dynamic Model Loading**: Real-time model discovery and categorization

### Backend
- **FastAPI**: High-performance async web framework
- **WebSocket Support**: Real-time updates and logging
- **Multi-LLM Integration**: OpenAI and Ollama support with dynamic URL configuration
- **CrewAI Integration**: Seamless workflow execution
- **File Management**: Document upload and result storage
- **Docker Support**: Native support for containerized Ollama instances

### CrewAI Agents
- **Document Processor**: Handles file uploads and document preparation
- **Model Selector**: Manages LLM selection and configuration
- **Data Generator**: Creates synthetic data in Alpaca format
- **RAG Implementer**: Implements embedding and reranking capabilities
- **Performance Optimizer**: GPU optimization for high-performance processing

## 📋 Prerequisites

- **Python**: 3.10 or higher (< 3.13)
- **Operating System**: Windows, macOS, or Linux
- **Hardware**: 
  - Minimum: 8GB RAM, modern CPU
  - Recommended: 16GB+ RAM, dual RTX 4090 GPUs
- **Docker**: Required for Ollama containerized deployment
- **Optional**: Local Ollama installation

## 🛠️ Installation

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd local_crewai_workflow_for_synthetic_data_with_rag_and_llm_options
   ```

2. **Run the startup script**:
   ```bash
   python start_server.py
   ```

   This will:
   - Check Python version compatibility
   - Install all required dependencies
   - Create necessary directories
   - Check Ollama status (if applicable)
   - Start the web server

3. **Access the application**:
   Open your browser and navigate to `http://localhost:8000`

### Manual Installation

1. **Install dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Create environment file** (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start the server**:
   ```bash
   uvicorn backend.app:app --host 0.0.0.0 --port 8000
   ```

## 🐳 Docker Setup for Ollama

### Option 1: Docker Compose (Recommended)

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  ollama_data:
```

Start Ollama:
```bash
docker-compose up -d
```

### Option 2: Docker Run

```bash
docker run -d \
  --name ollama \
  --gpus all \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  -e OLLAMA_HOST=0.0.0.0 \
  ollama/ollama:latest
```

### Pull Models in Docker

```bash
# Access the container
docker exec -it ollama bash

# Pull models
ollama pull llama3.2
ollama pull mistral
ollama pull nomic-embed-text
ollama pull phi3
ollama pull gemma:7b

# Exit container
exit
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# OpenAI Configuration (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Ollama Configuration
OLLAMA_BASE_URL=http://host.docker.internal:11434

# Server Configuration
HOST=0.0.0.0
PORT=8000

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1
NVIDIA_VISIBLE_DEVICES=0,1
```

### Ollama URL Configuration

The application now supports dynamic Ollama URL configuration:

- **Local Installation**: `http://localhost:11434`
- **Docker (Windows/Mac)**: `http://host.docker.internal:11434`
- **Docker (Linux)**: `http://172.17.0.1:11434` or container IP
- **Remote Server**: `http://your-server-ip:11434`

**Important**: When you change the Ollama URL in the web interface, the application automatically refreshes the available models from the new URL.

### Local Ollama Setup (Alternative)

If you prefer local installation:

1. **Install Ollama**: Download from [https://ollama.ai](https://ollama.ai)

2. **Pull models**:
   ```bash
   ollama pull llama3.2
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

3. **Start Ollama server**:
   ```bash
   ollama serve
   ```

## 🎯 Usage

### 1. Configure Models

- **Ollama URL**: Set the correct URL for your Ollama instance
  - Docker: `http://host.docker.internal:11434`
  - Local: `http://localhost:11434`
- **Data Generation Model**: Select from OpenAI GPT models or Ollama models
- **Embedding Model**: Choose embedding models for RAG implementation
- **Reranking Model**: Optional model for result reranking
- **API Keys**: Provide OpenAI API key if using OpenAI models

### 2. Model Management

- **Auto-Discovery**: Models are automatically loaded when you change the Ollama URL
- **Model Categorization**: Models are intelligently categorized for different use cases
- **Auto-Selection**: Use the "Auto-select Recommended" button for optimal model selection
- **Model Testing**: Test connectivity and performance before running workflows

### 3. Upload Documents

- Supported formats: PDF, CSV, TXT
- Multiple file upload supported
- Drag and drop interface
- File validation and preview

### 4. Run Workflows

#### Full Workflow
Executes all steps: document processing → model selection → data generation → RAG implementation → optimization

#### Partial Workflows
- **Data Generation Only**: Focus on synthetic data creation
- **RAG Implementation Only**: Implement retrieval and reranking
- **Model Testing**: Test model connectivity and performance

### 5. Monitor Progress

- Real-time progress tracking
- Live log streaming
- Step-by-step status updates
- Error reporting and handling

### 6. View Results

- Download generated data and results
- View results in-browser
- Export in various formats
- Result history and management

### 7. System Troubleshooting

The application includes a comprehensive troubleshooting interface accessible via the **"Troubleshooting"** button in the main workflow control panel.

#### Diagnostic Tests Available:

- **API Health Check**: Tests backend connectivity, health endpoints, and Ollama server accessibility
- **Docker Ollama Test**: Specifically validates Docker Ollama container connectivity and model availability
- **Model Debug**: Deep analysis of specific models including functionality testing and performance metrics
- **Workflow Model Test**: Validates complete workflow configuration and model compatibility
- **LLM Manager Debug**: Comprehensive debugging of the LLM management system

#### Troubleshooting Features:

- **Real-time Logs**: Live streaming of test output with color-coded severity levels
- **Test Results View**: Structured results with pass/fail statistics and detailed error information
- **Individual or Batch Testing**: Run specific tests or execute all diagnostics at once
- **Export Diagnostics**: Download complete diagnostic reports in JSON format for support
- **Visual Status Indicators**: Clear visual feedback on test progress and results
- **Configurable Parameters**: Customize test parameters like model names and URLs

#### Using the Troubleshooting Interface:

1. Click the **"Troubleshooting"** button in the workflow control panel
2. Select individual tests or click **"Run All Tests"** for comprehensive diagnostics
3. Monitor real-time progress in the **"Live Logs"** view
4. Switch to **"Test Results"** view for structured analysis
5. Export results using the **"Export Results"** button for support or documentation

This feature incorporates all existing test scripts (`test_api.py`, `test_docker_ollama.py`, `test_model_debug.py`, `test_workflow_model.py`, `debug_llm_manager.py`) into an intuitive web interface, making system diagnostics accessible to all users.

## 🔍 API Endpoints

### Core Endpoints
- `GET /` - Serve frontend interface
- `GET /health` - Health check
- `POST /upload-documents` - Upload documents
- `POST /start-workflow` - Start workflow execution
- `POST /stop-workflow/{workflow_id}` - Stop running workflow
- `POST /test-models` - Test model connectivity

### WebSocket
- `WS /ws` - Real-time updates and logging

### Results
- `GET /download-result/{result_id}` - Download results
- `GET /view-result/{result_id}` - View results
- `GET /list-results` - List all results

### Ollama Integration
- `GET /ollama-models?ollama_url={url}` - List available Ollama models from specified URL
- `POST /pull-ollama-model` - Pull new Ollama model

### Troubleshooting Endpoints
- `POST /troubleshoot/api-health` - Run comprehensive API health checks
- `POST /troubleshoot/docker-ollama` - Test Docker Ollama connectivity and models
- `POST /troubleshoot/model-debug?model_name={name}` - Debug specific model functionality
- `POST /troubleshoot/workflow-model` - Test workflow model configuration
- `POST /troubleshoot/llm-debug` - Run LLM manager diagnostics

## 🏃‍♂️ Development

### Development Mode

Start the server with auto-reload:
```bash
python start_server.py --reload
```

### Project Structure

```
├── frontend/                 # Frontend files
│   ├── index.html            # Main HTML interface
│   ├── styles.css            # CSS styling
│   └── script.js             # JavaScript functionality
├── backend/                  # Backend API
│   ├── app.py                # FastAPI application
│   ├── models.py             # Pydantic models
│   ├── llm_manager.py        # LLM integration
│   ├── workflow_manager.py   # Workflow execution
│   ├── websocket_manager.py  # WebSocket handling
│   └── troubleshooting.py    # Diagnostic test manager
├── src/                      # CrewAI source code
│   └── local_crewai_workflow_for_synthetic_data_with_rag_and_llm_options/
│       ├── crew.py           # CrewAI crew definition
│       ├── main.py           # CLI interface
│       └── config/           # Agent and task configurations
├── uploads/                  # Uploaded documents
├── results/                  # Workflow results
└── logs/                     # Application logs
```

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   python start_server.py --port 8001
   ```

2. **Ollama connection failed**:
   - **Docker**: Ensure container is running: `docker ps | grep ollama`
   - **Local**: Ensure Ollama is running: `ollama serve`
   - **URL**: Check Ollama URL in configuration matches your setup
   - **Network**: For Docker, use `http://host.docker.internal:11434` on Windows/Mac
   - **Models**: Verify models are available: `docker exec ollama ollama list`

3. **Docker Ollama Issues**:
   ```bash
   # Check container status
   docker ps -a | grep ollama
   
   # View container logs
   docker logs ollama
   
   # Restart container
   docker restart ollama
   
   # Check if port is accessible
   curl http://localhost:11434/api/tags
   ```

4. **OpenAI API errors**:
   - Verify API key is correct
   - Check API quota and billing
   - Ensure model names are valid

5. **GPU not detected in Docker**:
   - Install NVIDIA Container Toolkit
   - Verify GPU access: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`

6. **Models not loading**:
   - Check Ollama URL configuration
   - Verify network connectivity to Ollama instance
   - Use "Refresh Models" button to reload

### Logs and Debugging

- Check browser console for frontend errors
- Server logs are displayed in terminal
- Enable debug mode: `LOG_LEVEL=DEBUG` in `.env`
- Docker logs: `docker logs ollama`

### Network Configuration

For different deployment scenarios:

- **Same machine**: `http://localhost:11434`
- **Docker Desktop (Windows/Mac)**: `http://host.docker.internal:11434`
- **Docker on Linux**: `http://172.17.0.1:11434` or container IP
- **Remote server**: `http://server-ip:11434`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [CrewAI](https://crewai.com) - Multi-agent framework
- [FastAPI](https://fastapi.tiangolo.com) - Web framework
- [Ollama](https://ollama.ai) - Local LLM support
- [Bootstrap](https://getbootstrap.com) - UI framework

## 📞 Support

For support, questions, or feedback:
- Create an issue in the GitHub repository
- Check the [CrewAI documentation](https://docs.crewai.com)
- Join the [CrewAI Discord](https://discord.com/invite/X4JWnZnxPb)

---

**Ready to create powerful AI workflows with Docker-powered local models!** 🚀🐳
