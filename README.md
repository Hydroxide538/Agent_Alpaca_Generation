# CrewAI Workflow Manager for Synthetic Data & RAG

A comprehensive web-based interface for managing CrewAI workflows focused on synthetic data generation and RAG (Retrieval-Augmented Generation) implementation. This project provides a modern frontend and robust backend to leverage CrewAI's multi-agent capabilities with support for both OpenAI and Ollama models.

## 🚀 Features

- **Web-based Interface**: Modern, responsive frontend for easy workflow management
- **Multi-LLM Support**: Compatible with OpenAI GPT models and local Ollama models
- **Document Processing**: Upload and process PDF, CSV, and TXT files
- **Real-time Updates**: WebSocket-based live progress tracking
- **GPU Optimization**: Built-in support for dual RTX 4090 GPU setups
- **Flexible Workflows**: Run full workflows or individual components (data generation, RAG implementation)
- **Model Testing**: Built-in model connectivity and performance testing
- **Results Management**: Download and view workflow results

## 🏗️ Architecture

### Frontend
- **HTML/CSS/JavaScript**: Modern responsive interface
- **Bootstrap 5**: UI framework for consistent styling
- **WebSocket Client**: Real-time communication with backend
- **Local Storage**: Configuration persistence

### Backend
- **FastAPI**: High-performance async web framework
- **WebSocket Support**: Real-time updates and logging
- **Multi-LLM Integration**: OpenAI and Ollama support
- **CrewAI Integration**: Seamless workflow execution
- **File Management**: Document upload and result storage

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
- **Optional**: Ollama installed for local model support

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

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
# OpenAI Configuration (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Server Configuration
HOST=0.0.0.0
PORT=8000

# GPU Configuration
CUDA_VISIBLE_DEVICES=0,1
NVIDIA_VISIBLE_DEVICES=0,1
```

### Ollama Setup (Optional)

If you want to use local models via Ollama:

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

- **Data Generation Model**: Select from OpenAI GPT models or Ollama models
- **Embedding Model**: Choose embedding models for RAG implementation
- **Reranking Model**: Optional model for result reranking
- **API Keys**: Provide OpenAI API key if using OpenAI models
- **Ollama URL**: Configure Ollama server URL (default: http://localhost:11434)

### 2. Upload Documents

- Supported formats: PDF, CSV, TXT
- Multiple file upload supported
- Drag and drop interface
- File validation and preview

### 3. Run Workflows

#### Full Workflow
Executes all steps: document processing → model selection → data generation → RAG implementation → optimization

#### Partial Workflows
- **Data Generation Only**: Focus on synthetic data creation
- **RAG Implementation Only**: Implement retrieval and reranking
- **Model Testing**: Test model connectivity and performance

### 4. Monitor Progress

- Real-time progress tracking
- Live log streaming
- Step-by-step status updates
- Error reporting and handling

### 5. View Results

- Download generated data and results
- View results in-browser
- Export in various formats
- Result history and management

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
- `GET /ollama-models` - List available Ollama models
- `POST /pull-ollama-model` - Pull new Ollama model

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
│   └── websocket_manager.py  # WebSocket handling
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
   - Ensure Ollama is running: `ollama serve`
   - Check Ollama URL in configuration
   - Verify models are pulled: `ollama list`

3. **OpenAI API errors**:
   - Verify API key is correct
   - Check API quota and billing
   - Ensure model names are valid

4. **GPU not detected**:
   - Install NVIDIA drivers
   - Install CUDA toolkit
   - Verify GPU visibility: `nvidia-smi`

### Logs and Debugging

- Check browser console for frontend errors
- Server logs are displayed in terminal
- Enable debug mode: `LOG_LEVEL=DEBUG` in `.env`

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

**Ready to create powerful AI workflows with ease!** 🚀
