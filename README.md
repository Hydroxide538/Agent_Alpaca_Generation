# Synthetic Alpaca Training Data from Docs

A comprehensive web-based interface for managing CrewAI workflows focused on synthetic data generation and RAG (Retrieval-Augmented Generation) implementation. This project provides a modern frontend and robust backend to leverage CrewAI's multi-agent capabilities with support for multiple LLM providers and an advanced LLM competition system.

## 🚀 Features

### Core Workflow Management
- **Web-based Interface**: Modern, responsive frontend for easy workflow management
- **Multi-LLM Support**: Compatible with OpenAI GPT models and local Ollama models
- **Docker Integration**: Full support for Ollama running in Docker containers
- **Dynamic Configuration**: Real-time model loading and URL configuration
- **Document Processing**: Upload and process PDF, CSV, and TXT files with token counting
- **Real-time Updates**: WebSocket-based live progress tracking
- **GPU Optimization**: Built-in support for dual RTX 4090 GPU setups

### Advanced LLM Management
- **Manager Agent System**: Intelligent LLM selection based on performance data
- **LLM Shootout Arena**: Competitive evaluation system for model performance
- **Performance Tracking**: Automatic scoring and performance history
- **Dynamic Model Selection**: Real-time model discovery and categorization
- **Token Analysis**: Comprehensive token counting and context window analysis
- **Model Testing**: Built-in connectivity and performance testing

### Workflow Types
- **Full Workflow**: Complete pipeline from document processing to RAG implementation
- **Data Generation Only**: Focus on synthetic data creation in Alpaca format
- **RAG Implementation Only**: Implement retrieval and reranking capabilities
- **Model Testing**: Comprehensive model connectivity and performance testing

### Troubleshooting & Diagnostics
- **Integrated Troubleshooting**: Built-in diagnostic tools for system health monitoring
- **Real-time Diagnostics**: Live troubleshooting with detailed error analysis
- **Export Diagnostics**: Download complete diagnostic reports for support
- **Troubleshooting Wiki**: Comprehensive knowledge base with issue history
- **Enhanced LLM Evaluation**: Advanced model evaluation with thinking model support

## 🏗️ Architecture

### Frontend
- **HTML/CSS/JavaScript**: Modern responsive interface with Bootstrap 5
- **WebSocket Client**: Real-time communication with backend
- **Local Storage**: Configuration persistence
- **Dynamic Model Loading**: Real-time model discovery and categorization
- **LLM Shootout Interface**: Dedicated competition arena for model evaluation

### Backend Components
- **FastAPI**: High-performance async web framework
- **WebSocket Support**: Real-time updates and logging
- **Multi-LLM Integration**: OpenAI and Ollama support with dynamic URL configuration
- **CrewAI Integration**: Seamless workflow execution with hierarchical process management

#### Core Managers
- **LLMManager**: Handles all LLM operations, model testing, and performance tracking
- **WorkflowManager**: Orchestrates complete workflow execution
- **LLMShootoutManager**: Manages competitive model evaluation
- **ManagerAgent**: Intelligent LLM selection based on performance data
- **TroubleshootingManager**: Comprehensive diagnostic and testing system
- **WebSocketManager**: Real-time communication and progress updates

#### Advanced Features
- **ImprovedAlpacaGenerator**: Enhanced synthetic data generation with structured extraction
- **RAGSystem**: Complete retrieval-augmented generation implementation
- **ManagerScoringSystem**: Intelligent scoring system for model evaluation
- **TokenCounter**: Comprehensive token analysis and context window management
- **SafeLLMWrapper**: CrewAI-compatible LLM wrapper with error handling

### CrewAI Agents
- **Manager Agent**: Orchestrates workflow and selects optimal LLMs dynamically
- **Document Processor**: Handles file uploads and document preparation
- **Fact Extractor**: Extracts structured facts from document content
- **Concept Extractor**: Extracts structured concepts from document content
- **QA Generator**: Creates high-quality Q&A pairs in Alpaca format
- **Quality Evaluator**: Evaluates the quality of generated Q&A pairs
- **LLM Tester**: Runs LLM performance tests and records results

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
ollama pull llama3.3
ollama pull mistral
ollama pull bge-m3
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

The application supports dynamic Ollama URL configuration:

- **Local Installation**: `http://localhost:11434`
- **Docker (Windows/Mac)**: `http://host.docker.internal:11434`
- **Docker (Linux)**: `http://172.17.0.1:11434` or container IP
- **Remote Server**: `http://your-server-ip:11434`

**Important**: When you change the Ollama URL in the web interface, the application automatically refreshes the available models from the new URL.

## 🎯 Usage

### 1. Configure Models

#### Manager Agent Configuration
- **Manager Model**: Select the LLM that will orchestrate the workflow and make intelligent model selections
- **Selection Strategy**: Choose how the Manager Agent selects LLMs for tasks:
  - **Performance-Based**: Selects models based on historical performance data
  - **Balanced**: Balances performance and speed
  - **Speed-Optimized**: Prioritizes faster models
  - **Quality-Focused**: Prioritizes highest quality models

#### Model Configuration
- **Ollama URL**: Set the correct URL for your Ollama instance
- **Embedding Model**: Choose embedding models for RAG implementation
- **Reranking Model**: Optional model for result reranking
- **API Keys**: Provide OpenAI API key if using OpenAI models

### 2. Model Management

- **Auto-Discovery**: Models are automatically loaded when you change the Ollama URL
- **Model Categorization**: Models are intelligently categorized for different use cases
- **Performance Tracking**: System tracks model performance across different tasks
- **Model Testing**: Test connectivity and performance before running workflows

### 3. Upload Documents

- Supported formats: PDF, CSV, TXT
- Multiple file upload supported
- Drag and drop interface
- File validation and preview
- **Token Analysis**: Automatic token counting and context window analysis

### 4. Run Workflows

#### Full Workflow
Executes all steps: document processing → model selection → data generation → RAG implementation → optimization

#### Partial Workflows
- **Data Generation Only**: Focus on synthetic data creation in Alpaca format
- **RAG Implementation Only**: Implement retrieval and reranking
- **Model Testing**: Test model connectivity and performance

#### LLM Shootout Arena
- **Competitive Evaluation**: Run multiple models against the same tasks
- **Real-time Competition**: Watch models compete in real-time
- **Performance Comparison**: Compare models across different evaluation criteria
- **Automatic Scoring**: Intelligent scoring system evaluates model outputs

### 5. Monitor Progress

- Real-time progress tracking
- Live log streaming
- Step-by-step status updates
- Error reporting and handling
- WebSocket-based real-time updates

### 6. View Results

- Download generated data and results
- View results in-browser
- Export in various formats
- Result history and management
- Token statistics and analysis

### 7. System Troubleshooting

The application includes a comprehensive troubleshooting interface accessible via the **"Troubleshooting"** button.

#### Diagnostic Tests Available:

- **API Health Check**: Tests backend connectivity, health endpoints, and Ollama server accessibility
- **Docker Ollama Test**: Validates Docker Ollama container connectivity and model availability
- **Model Debug**: Deep analysis of specific models including functionality testing and performance metrics
- **Workflow Model Test**: Validates complete workflow configuration and model compatibility
- **LLM Manager Debug**: Comprehensive debugging of the LLM management system
- **CrewAI Workflow Test**: Tests complete CrewAI workflow execution
- **Ollama Workflow Test**: Tests Ollama workflow configuration with dynamic model selection
- **Enhanced LLM Evaluation**: Advanced model evaluation with thinking model support

#### Troubleshooting Features:

- **Real-time Logs**: Live streaming of test output with color-coded severity levels
- **Test Results View**: Structured results with pass/fail statistics and detailed error information
- **Individual or Batch Testing**: Run specific tests or execute all diagnostics at once
- **Export Diagnostics**: Download complete diagnostic reports in JSON format for support
- **Visual Status Indicators**: Clear visual feedback on test progress and results
- **Configurable Parameters**: Customize test parameters like model names and URLs
- **Troubleshooting Wiki**: Comprehensive knowledge base with issue history and solutions

## 🔍 API Endpoints

### Core Endpoints
- `GET /` - Serve frontend interface
- `GET /health` - Health check
- `POST /upload-documents` - Upload documents with token analysis
- `POST /start-workflow` - Start workflow execution
- `POST /stop-workflow/{workflow_id}` - Stop running workflow
- `POST /test-models` - Test model connectivity

### WebSocket
- `WS /ws` - Real-time updates and logging
- `WS /ws/llm-shootout` - LLM Shootout real-time updates

### Results & Documents
- `GET /download-result/{result_id}` - Download results
- `GET /view-result/{result_id}` - View results
- `GET /list-results` - List all results
- `DELETE /clear-results` - Clear all results
- `DELETE /clear-documents` - Clear all documents and reset system

### Token Analysis
- `GET /document-tokens` - Get token statistics for all uploaded documents
- `GET /document-tokens/{document_id}` - Get detailed token statistics for specific document

### Ollama Integration
- `GET /ollama-models?ollama_url={url}` - List available Ollama models from specified URL
- `POST /pull-ollama-model` - Pull new Ollama model

### LLM Shootout
- `GET /llm-shootout` - Serve LLM Shootout Arena interface
- `GET /api/documents` - Get available documents for shootout
- `GET /api/ollama/models` - Get available Ollama models for shootout
- `POST /api/llm-shootout/start` - Start LLM shootout competition
- `POST /api/llm-shootout/stop` - Stop current shootout
- `GET /api/llm-shootout/status` - Get competition status

### Troubleshooting Endpoints
- `POST /troubleshoot/api-health` - Run comprehensive API health checks
- `POST /troubleshoot/docker-ollama` - Test Docker Ollama connectivity and models
- `POST /troubleshoot/model-debug?model_name={name}` - Debug specific model functionality
- `POST /troubleshoot/workflow-model` - Test workflow model configuration
- `POST /troubleshoot/llm-debug` - Run LLM manager diagnostics
- `POST /troubleshoot/crew-workflow` - Run CrewAI workflow execution tests
- `POST /troubleshoot/ollama-workflow` - Run Ollama workflow configuration tests
- `POST /troubleshoot/enhanced-llm-evaluation` - Run enhanced LLM evaluation

### System Information
- `GET /system-info` - Get system information including GPU status

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
│   ├── llm_shootout.html     # LLM Shootout Arena interface
│   ├── styles.css            # CSS styling
│   └── script.js             # JavaScript functionality
├── backend/                  # Backend API
│   ├── app.py                # FastAPI application
│   ├── models.py             # Pydantic models
│   ├── llm_manager.py        # LLM integration and management
│   ├── llm_shootout_manager.py # LLM competition system
│   ├── manager_agent.py      # Intelligent LLM selection
│   ├── manager_scoring_system.py # Model scoring system
│   ├── workflow_manager.py   # Workflow execution
│   ├── websocket_manager.py  # WebSocket handling
│   ├── troubleshooting.py    # Diagnostic test manager
│   ├── improved_alpaca_generator.py # Enhanced data generation
│   ├── rag_system.py         # RAG implementation
│   ├── safe_llm_wrapper.py   # CrewAI-compatible LLM wrapper
│   ├── token_counter.py      # Token analysis and counting
│   ├── json_parser_fix.py    # Robust JSON parsing
│   └── message_validator.py  # Message validation for LLMs
├── src/                      # CrewAI source code
│   └── local_crewai_workflow_for_synthetic_data_with_rag_and_llm_options/
│       ├── crew.py           # CrewAI crew definition with Manager Agent
│       ├── main.py           # CLI interface
│       └── config/           # Agent and task configurations
├── troubleshooting/          # Troubleshooting system
│   ├── README.md             # Troubleshooting documentation
│   ├── scripts/              # Test scripts
│   ├── wiki/                 # Troubleshooting wiki interface
│   └── templates/            # Issue templates
├── knowledge/                # Knowledge management
│   └── wiki/                 # Comprehensive documentation
├── uploads/                  # Uploaded documents
├── results/                  # Workflow results
├── vector_db/                # Vector database storage
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

7. **LLM Shootout Issues**:
   - Ensure at least 2 text generation models are available
   - Check document upload status
   - Verify model connectivity before starting competition

### Logs and Debugging

- Check browser console for frontend errors
- Server logs are displayed in terminal
- Enable debug mode: `LOG_LEVEL=DEBUG` in `.env`
- Docker logs: `docker logs ollama`
- Use integrated troubleshooting interface for comprehensive diagnostics

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
- Use the integrated troubleshooting system for diagnostics
- Check the troubleshooting wiki for common solutions
- Check the [CrewAI documentation](https://docs.crewai.com)
- Join the [CrewAI Discord](https://discord.com/invite/X4JWnZnxPb)

---

**Ready to create powerful AI workflows with intelligent LLM management and competitive model evaluation!** 🚀🏆
