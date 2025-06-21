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
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Global managers
llm_manager = LLMManager()
workflow_manager = WorkflowManager()
websocket_manager = WebSocketManager()

# Storage directories
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file"""
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents for processing"""
    try:
        uploaded_files = []
        
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
            
            uploaded_files.append({
                "id": file_id,
                "original_name": file.filename,
                "filename": unique_filename,
                "path": file_path,
                "size": os.path.getsize(file_path),
                "type": file_extension[1:].upper()
            })
        
        logger.info(f"Successfully uploaded {len(uploaded_files)} files")
        return {"documents": uploaded_files, "count": len(uploaded_files)}
        
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
                
                with open(result_path, 'r') as f:
                    result_data = json.load(f)
                
                results.append({
                    "id": result_id,
                    "title": result_data.get("title", f"Result {result_id}"),
                    "description": result_data.get("description", "Workflow result"),
                    "created_at": result_data.get("created_at"),
                    "type": result_data.get("type", "unknown")
                })
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Failed to list results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list results: {str(e)}")

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
async def get_ollama_models():
    """Get available Ollama models"""
    try:
        models = await llm_manager.get_ollama_models()
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
