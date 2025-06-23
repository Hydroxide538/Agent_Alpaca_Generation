from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
import os
import uuid
import shutil
import logging
from datetime import datetime
import sys
import traceback

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from local_crewai_workflow_for_synthetic_data_with_rag_and_llm_options.crew import LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew
from backend.models import WorkflowConfig, WorkflowStatus, TestResult
from backend.llm_manager import LLMManager
from backend.workflow_manager import WorkflowManager
from backend.websocket_manager import WebSocketManager
from backend.troubleshooting import TroubleshootingManager
from backend.token_counter import TokenCounter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CrewAI Workflow API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
# Get the correct path to frontend directory relative to the project root
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_path = os.path.join(project_root, "frontend")
troubleshooting_path = os.path.join(project_root, "troubleshooting")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")
app.mount("/troubleshooting", StaticFiles(directory=troubleshooting_path), name="troubleshooting")

# Global managers
llm_manager = LLMManager()
workflow_manager = WorkflowManager()
websocket_manager = WebSocketManager()
troubleshooting_manager = TroubleshootingManager()

# Storage directories
UPLOAD_DIR = "uploads"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file"""
    frontend_html_path = os.path.join(project_root, "frontend", "index.html")
    return FileResponse(frontend_html_path)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents for processing"""
    try:
        uploaded_files = []
        token_counter = TokenCounter()
        
        for file in files:
            # Validate file type
            if not file.filename.lower().endswith(('.pdf', '.csv', '.txt')):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file.filename}"
                )
            
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"{file_id}{file_extension}"
            file_path = os.path.join(UPLOAD_DIR, unique_filename)
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Count tokens in the uploaded file
            token_stats = token_counter.count_tokens_in_file(file_path)
            
            uploaded_files.append({
                "id": file_id,
                "original_name": file.filename,
                "filename": unique_filename,
                "path": file_path,
                "size": os.path.getsize(file_path),
                "type": file_extension[1:].upper(),
                "token_count": token_stats.get("token_count", 0),
                "character_count": token_stats.get("character_count", 0),
                "word_count": token_stats.get("word_count", 0),
                "encoding": token_stats.get("encoding", "unknown")
            })
        
        # Calculate total token count
        total_tokens = sum(doc.get("token_count", 0) for doc in uploaded_files)
        
        logger.info(f"Successfully uploaded {len(uploaded_files)} files with {total_tokens} total tokens")
        return {
            "documents": uploaded_files, 
            "count": len(uploaded_files),
            "total_tokens": total_tokens,
            "token_summary": {
                "total_tokens": total_tokens,
                "total_characters": sum(doc.get("character_count", 0) for doc in uploaded_files),
                "total_words": sum(doc.get("word_count", 0) for doc in uploaded_files),
                "encoding": token_counter.encoding_name
            }
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/test-models")
async def test_models(config: WorkflowConfig):
    """Test the selected models for connectivity and functionality"""
    try:
        results = await llm_manager.test_models(config)
        return results
    except Exception as e:
        logger.error(f"Model testing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model testing failed: {str(e)}")

@app.post("/start-workflow")
async def start_workflow(config: WorkflowConfig):
    """Start the CrewAI workflow"""
    try:
        workflow_id = str(uuid.uuid4())
        
        # Start workflow in background
        asyncio.create_task(
            workflow_manager.run_workflow(workflow_id, config, websocket_manager)
        )
        
        return {"workflow_id": workflow_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Failed to start workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")

@app.post("/stop-workflow/{workflow_id}")
async def stop_workflow(workflow_id: str):
    """Stop a running workflow"""
    try:
        success = await workflow_manager.stop_workflow(workflow_id)
        if success:
            return {"status": "stopped"}
        else:
            raise HTTPException(status_code=404, detail="Workflow not found")
    except Exception as e:
        logger.error(f"Failed to stop workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop workflow: {str(e)}")

@app.get("/workflow-status/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get the status of a workflow"""
    try:
        status = workflow_manager.get_workflow_status(workflow_id)
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Workflow not found")
    except Exception as e:
        logger.error(f"Failed to get workflow status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow status: {str(e)}")

@app.get("/download-result/{result_id}")
async def download_result(result_id: str):
    """Download a workflow result"""
    try:
        result_path = os.path.join(RESULTS_DIR, f"{result_id}.json")
        if os.path.exists(result_path):
            return FileResponse(
                result_path,
                media_type='application/json',
                filename=f"result_{result_id}.json"
            )
        else:
            raise HTTPException(status_code=404, detail="Result not found")
    except Exception as e:
        logger.error(f"Failed to download result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download result: {str(e)}")

@app.get("/view-result/{result_id}")
async def view_result(result_id: str):
    """View a workflow result"""
    try:
        result_path = os.path.join(RESULTS_DIR, f"{result_id}.json")
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                result_data = json.load(f)
            return result_data
        else:
            raise HTTPException(status_code=404, detail="Result not found")
    except Exception as e:
        logger.error(f"Failed to view result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to view result: {str(e)}")

@app.get("/list-results")
async def list_results():
    """List all available results"""
    try:
        results = []
        for filename in os.listdir(RESULTS_DIR):
            if filename.endswith('.json'):
                result_id = filename[:-5]  # Remove .json extension
                result_path = os.path.join(RESULTS_DIR, filename)
                
                try:
                    with open(result_path, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    
                    # Ensure result_data is a dictionary
                    if not isinstance(result_data, dict):
                        logger.warning(f"Result file {filename} contains non-dict data: {type(result_data)}")
                        # Skip this file or create a default entry
                        result_data = {
                            "title": f"Result {result_id}",
                            "description": "Invalid result format",
                            "created_at": datetime.now().isoformat(),
                            "type": "unknown"
                        }
                    
                    # Get file size
                    file_size = os.path.getsize(result_path)
                    
                    results.append({
                        "id": result_id,
                        "title": result_data.get("title", f"Result {result_id}"),
                        "description": result_data.get("description", "Workflow result"),
                        "created_at": result_data.get("created_at", datetime.now().isoformat()),
                        "type": result_data.get("type", "unknown"),
                        "file_size": file_size
                    })
                    
                except json.JSONDecodeError as je:
                    logger.error(f"Invalid JSON in result file {filename}: {str(je)}")
                    # Add a placeholder entry for corrupted files
                    results.append({
                        "id": result_id,
                        "title": f"Corrupted Result {result_id}",
                        "description": "File contains invalid JSON",
                        "created_at": datetime.now().isoformat(),
                        "type": "corrupted",
                        "file_size": os.path.getsize(result_path)
                    })
                except Exception as fe:
                    logger.error(f"Error reading result file {filename}: {str(fe)}")
                    continue
        
        # Sort by creation date (newest first)
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Failed to list results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list results: {str(e)}")

@app.delete("/clear-results")
async def clear_results():
    """Clear all results"""
    try:
        deleted_count = 0
        for filename in os.listdir(RESULTS_DIR):
            if filename.endswith('.json'):
                file_path = os.path.join(RESULTS_DIR, filename)
                os.remove(file_path)
                deleted_count += 1
        
        logger.info(f"Cleared {deleted_count} result files")
        return {"status": "success", "deleted_count": deleted_count, "message": f"Cleared {deleted_count} result files"}
    except Exception as e:
        logger.error(f"Failed to clear results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear results: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        websocket_manager.disconnect(websocket)

@app.get("/ollama-models")
async def get_ollama_models(ollama_url: str = "http://localhost:11434"):
    """Get available Ollama models"""
    try:
        models = await llm_manager.get_ollama_models(ollama_url)
        return {"models": models}
    except Exception as e:
        logger.error(f"Failed to get Ollama models: {str(e)}")
        return {"models": [], "error": str(e)}

@app.post("/pull-ollama-model")
async def pull_ollama_model(model_name: str):
    """Pull an Ollama model"""
    try:
        success = await llm_manager.pull_ollama_model(model_name)
        if success:
            return {"status": "success", "message": f"Model {model_name} pulled successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to pull model {model_name}")
    except Exception as e:
        logger.error(f"Failed to pull Ollama model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to pull model: {str(e)}")

@app.get("/system-info")
async def get_system_info():
    """Get system information including GPU status"""
    try:
        import psutil
        import GPUtil
        
        # CPU info
        cpu_info = {
            "cores": psutil.cpu_count(),
            "usage": psutil.cpu_percent(interval=1)
        }
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_info = {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent
        }
        
        # GPU info
        gpu_info = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "memory_total": gpu.memoryTotal,
                    "memory_used": gpu.memoryUsed,
                    "memory_free": gpu.memoryFree,
                    "temperature": gpu.temperature,
                    "load": gpu.load * 100
                })
        except Exception as e:
            logger.warning(f"Could not get GPU info: {str(e)}")
        
        return {
            "cpu": cpu_info,
            "memory": memory_info,
            "gpu": gpu_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get system info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")

# Troubleshooting endpoints
@app.post("/troubleshoot/api-health")
async def troubleshoot_api_health():
    """Run API health check tests"""
    try:
        results = await troubleshooting_manager.run_api_health_test(websocket_manager)
        return results
    except Exception as e:
        logger.error(f"API health test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"API health test failed: {str(e)}")

@app.post("/troubleshoot/docker-ollama")
async def troubleshoot_docker_ollama():
    """Run Docker Ollama connection tests"""
    try:
        results = await troubleshooting_manager.run_docker_ollama_test(websocket_manager)
        return results
    except Exception as e:
        logger.error(f"Docker Ollama test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Docker Ollama test failed: {str(e)}")

@app.post("/troubleshoot/model-debug")
async def troubleshoot_model_debug(model_name: str = "bge-m3:latest"):
    """Run detailed model debugging"""
    try:
        results = await troubleshooting_manager.run_model_debug_test(model_name, websocket_manager)
        return results
    except Exception as e:
        logger.error(f"Model debug test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model debug test failed: {str(e)}")

@app.post("/troubleshoot/workflow-model")
async def troubleshoot_workflow_model(config: WorkflowConfig):
    """Run workflow model tests"""
    try:
        results = await troubleshooting_manager.run_workflow_model_test(config, websocket_manager)
        return results
    except Exception as e:
        logger.error(f"Workflow model test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow model test failed: {str(e)}")

@app.post("/troubleshoot/llm-debug")
async def troubleshoot_llm_debug(config: WorkflowConfig):
    """Run comprehensive LLM manager debugging"""
    try:
        results = await troubleshooting_manager.run_llm_manager_debug(config, websocket_manager)
        return results
    except Exception as e:
        logger.error(f"LLM debug test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM debug test failed: {str(e)}")

@app.post("/troubleshoot/crew-workflow")
async def troubleshoot_crew_workflow(config: WorkflowConfig):
    """Run CrewAI workflow execution tests"""
    try:
        results = await troubleshooting_manager.run_crew_workflow_test(config, websocket_manager)
        return results
    except Exception as e:
        logger.error(f"CrewAI workflow test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CrewAI workflow test failed: {str(e)}")

@app.post("/troubleshoot/ollama-workflow")
async def troubleshoot_ollama_workflow(config: WorkflowConfig):
    """Run Ollama workflow configuration tests"""
    try:
        results = await troubleshooting_manager.run_ollama_workflow_test(config, websocket_manager)
        return results
    except Exception as e:
        logger.error(f"Ollama workflow test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ollama workflow test failed: {str(e)}")

@app.get("/document-tokens")
async def get_document_tokens():
    """Get token statistics for all uploaded documents"""
    try:
        token_counter = TokenCounter()
        
        # Get all uploaded documents
        document_paths = []
        if os.path.exists(UPLOAD_DIR):
            for filename in os.listdir(UPLOAD_DIR):
                if filename.lower().endswith(('.pdf', '.csv', '.txt')):
                    document_paths.append(os.path.join(UPLOAD_DIR, filename))
        
        if not document_paths:
            return {
                "documents": [],
                "summary": {
                    "total_tokens": 0,
                    "total_characters": 0,
                    "total_words": 0,
                    "total_files": 0,
                    "successful_files": 0,
                    "failed_files": 0,
                    "encoding": token_counter.encoding_name
                }
            }
        
        # Count tokens for all documents
        results = token_counter.count_tokens_in_documents(document_paths)
        
        # Add context window analysis for common models
        context_windows = token_counter.get_model_context_windows()
        total_tokens = results["summary"]["total_tokens"]
        
        context_analysis = {}
        for model, window_size in context_windows.items():
            analysis = token_counter.estimate_context_window_usage(total_tokens, window_size)
            context_analysis[model] = analysis
        
        results["context_analysis"] = context_analysis
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to get document tokens: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document tokens: {str(e)}")

@app.get("/document-tokens/{document_id}")
async def get_document_token_details(document_id: str):
    """Get detailed token statistics for a specific document"""
    try:
        token_counter = TokenCounter()
        
        # Find the document file
        document_path = None
        if os.path.exists(UPLOAD_DIR):
            for filename in os.listdir(UPLOAD_DIR):
                if filename.startswith(document_id):
                    document_path = os.path.join(UPLOAD_DIR, filename)
                    break
        
        if not document_path or not os.path.exists(document_path):
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get detailed token statistics
        token_stats = token_counter.count_tokens_in_file(document_path)
        
        # Add context window analysis
        context_windows = token_counter.get_model_context_windows()
        token_count = token_stats.get("token_count", 0)
        
        context_analysis = {}
        for model, window_size in context_windows.items():
            analysis = token_counter.estimate_context_window_usage(token_count, window_size)
            context_analysis[model] = analysis
        
        return {
            "document_id": document_id,
            "document_path": document_path,
            "document_name": os.path.basename(document_path),
            "token_stats": token_stats,
            "context_analysis": context_analysis,
            "formatted_token_count": token_counter.format_token_count(token_count)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document token details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document token details: {str(e)}")

@app.delete("/clear-documents")
async def clear_documents():
    """Clear all uploaded documents and reset system"""
    try:
        deleted_count = 0
        
        # Clear uploaded documents
        if os.path.exists(UPLOAD_DIR):
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    deleted_count += 1
        
        # Clear vector database
        vector_db_dir = os.path.join(os.path.dirname(__file__), "vector_db")
        if os.path.exists(vector_db_dir):
            shutil.rmtree(vector_db_dir)
            os.makedirs(vector_db_dir, exist_ok=True)
        
        # Clear backend uploads
        backend_uploads = os.path.join(os.path.dirname(__file__), "uploads")
        if os.path.exists(backend_uploads):
            shutil.rmtree(backend_uploads)
            os.makedirs(backend_uploads, exist_ok=True)
        
        # Clear root uploads
        root_uploads = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
        if os.path.exists(root_uploads):
            shutil.rmtree(root_uploads)
            os.makedirs(root_uploads, exist_ok=True)
        
        logger.info(f"Cleared {deleted_count} documents and reset system")
        return {
            "status": "success", 
            "deleted_count": deleted_count, 
            "message": f"Cleared {deleted_count} documents and reset system"
        }
    except Exception as e:
        logger.error(f"Failed to clear documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")

@app.post("/troubleshoot/enhanced-llm-evaluation")
async def run_enhanced_llm_evaluation():
    """Run enhanced LLM evaluation with thinking model support"""
    try:
        # Check if there are uploaded documents
        document_paths = []
        if os.path.exists(UPLOAD_DIR):
            for filename in os.listdir(UPLOAD_DIR):
                if filename.lower().endswith(('.pdf', '.csv', '.txt')):
                    document_paths.append(os.path.join(UPLOAD_DIR, filename))
        
        if not document_paths:
            raise HTTPException(
                status_code=400, 
                detail="No documents found. Please upload documents first."
            )
        
        # Use the first document for evaluation
        document_path = document_paths[0]
        
        # Send initial status
        await websocket_manager.broadcast({
            "type": "log",
            "level": "info",
            "message": f"Starting Enhanced LLM Evaluation using document: {os.path.basename(document_path)}"
        })
        
        # Run the enhanced evaluation in background
        asyncio.create_task(
            troubleshooting_manager.run_enhanced_llm_evaluation(document_path, websocket_manager)
        )
        
        return {
            "status": "started",
            "message": "Enhanced LLM evaluation started",
            "document": os.path.basename(document_path)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced LLM evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced LLM evaluation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
