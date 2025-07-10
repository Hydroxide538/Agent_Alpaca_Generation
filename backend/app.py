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
from backend.llm_shootout_manager import LLMShootoutManager
from backend.enhanced_document_manager import EnhancedDocumentManager
from backend.graph_rag_system import GraphRAGSystem
from backend.cleanup_manager import CleanupManager

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

# Storage directories
UPLOAD_DIR = "uploads"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Global managers
llm_manager = LLMManager()
workflow_manager = WorkflowManager()
websocket_manager = WebSocketManager()
troubleshooting_manager = TroubleshootingManager()
llm_shootout_manager = LLMShootoutManager()
enhanced_document_manager = EnhancedDocumentManager(UPLOAD_DIR, "collections")

# Initialize GraphRAG system
graph_rag_system = GraphRAGSystem(llm_manager)
cleanup_manager = CleanupManager(
    upload_dir=UPLOAD_DIR,
    backend_upload_dir=os.path.join(os.path.dirname(__file__), "uploads"),
    vector_db_dir="vector_db",
    backend_vector_db_dir=os.path.join(os.path.dirname(__file__), "vector_db"),
    results_dir="results",
    backend_results_dir=RESULTS_DIR,
    neo4j_manager=graph_rag_system.neo4j_manager,
    graph_rag_system=graph_rag_system
)

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file"""
    frontend_html_path = os.path.join(project_root, "frontend", "index.html")
    return FileResponse(frontend_html_path)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Enhanced Document Management Endpoints

class CollectionRequest(BaseModel):
    name: str
    description: Optional[str] = ""

class DirectoryUploadRequest(BaseModel):
    directory_path: str
    collection_name: Optional[str] = None
    recursive: bool = True
    file_filter: Optional[str] = None

@app.post("/api/documents/upload")
async def upload_documents_enhanced(files: List[UploadFile] = File(...), collection_name: Optional[str] = None):
    """Enhanced document upload with collection support"""
    try:
        result = await enhanced_document_manager.upload_files(files, collection_name)
        return result
    except Exception as e:
        logger.error(f"Enhanced document upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/documents/upload-directory")
async def upload_directory(request: DirectoryUploadRequest):
    """Upload all files from a directory"""
    try:
        result = await enhanced_document_manager.upload_directory(
            request.directory_path,
            request.collection_name,
            request.recursive,
            request.file_filter
        )
        return result
    except Exception as e:
        logger.error(f"Directory upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Directory upload failed: {str(e)}")

@app.get("/api/documents")
async def get_all_documents_enhanced():
    """Get all uploaded documents with enhanced metadata"""
    try:
        documents = await enhanced_document_manager.get_all_documents()
        return {"documents": documents, "count": len(documents)}
    except Exception as e:
        logger.error(f"Failed to get documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")

@app.get("/api/documents/{document_id}")
async def get_document_by_id(document_id: str):
    """Get a specific document by ID"""
    try:
        document = await enhanced_document_manager.get_document_by_id(document_id)
        if document:
            return document
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a specific document"""
    try:
        success = await enhanced_document_manager.delete_document(document_id)
        if success:
            return {"status": "success", "message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.post("/api/collections")
async def create_collection(request: CollectionRequest):
    """Create a new document collection"""
    try:
        collection = await enhanced_document_manager.create_collection(request.name, request.description)
        await enhanced_document_manager._save_collection(collection)
        return collection.to_dict()
    except Exception as e:
        logger.error(f"Failed to create collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create collection: {str(e)}")

@app.get("/api/collections")
async def get_all_collections():
    """Get all document collections"""
    try:
        collections = await enhanced_document_manager.get_collections()
        return {"collections": collections, "count": len(collections)}
    except Exception as e:
        logger.error(f"Failed to get collections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get collections: {str(e)}")

@app.get("/api/collections/{collection_id}")
async def get_collection_by_id(collection_id: str):
    """Get a specific collection by ID"""
    try:
        collection = await enhanced_document_manager.get_collection(collection_id)
        if collection:
            return collection
        else:
            raise HTTPException(status_code=404, detail="Collection not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get collection: {str(e)}")

@app.delete("/api/collections/{collection_id}")
async def delete_collection(collection_id: str, delete_files: bool = False):
    """Delete a collection and optionally its files"""
    try:
        success = await enhanced_document_manager.delete_collection(collection_id, delete_files)
        if success:
            return {"status": "success", "message": "Collection deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Collection not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")

@app.post("/api/collections/{collection_id}/documents")
async def add_documents_to_collection(collection_id: str, document_ids: List[str]):
    """Add existing documents to a collection"""
    try:
        success = await enhanced_document_manager.add_documents_to_collection(collection_id, document_ids)
        if success:
            return {"status": "success", "message": "Documents added to collection successfully"}
        else:
            raise HTTPException(status_code=404, detail="Collection not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add documents to collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add documents to collection: {str(e)}")

@app.delete("/api/documents/clear-all")
async def clear_all_documents_enhanced():
    """Clear all documents and collections using enhanced manager"""
    try:
        result = await enhanced_document_manager.clear_all_documents()
        return result
    except Exception as e:
        logger.error(f"Failed to clear all documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear all documents: {str(e)}")

# Legacy document upload endpoint (for backward compatibility)
@app.post("/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents for processing (legacy endpoint)"""
    try:
        # Use enhanced document manager for consistency
        result = await enhanced_document_manager.upload_files(files)
        
        # Format response for backward compatibility
        return {
            "documents": result["documents"], 
            "count": result["count"],
            "total_tokens": result["total_tokens"],
            "token_summary": result["token_summary"]
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

# Enhanced Status API Endpoints

@app.get("/api/workflow/{workflow_id}/detailed-status")
async def get_detailed_workflow_status(workflow_id: str):
    """Get detailed workflow status with enhanced metrics"""
    try:
        detailed_status = workflow_manager.status_tracker.get_workflow_status(workflow_id)
        if detailed_status:
            return detailed_status.dict()
        else:
            raise HTTPException(status_code=404, detail="Workflow not found")
    except Exception as e:
        logger.error(f"Failed to get detailed workflow status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get detailed workflow status: {str(e)}")

@app.get("/api/workflow/{workflow_id}/activity-logs")
async def get_workflow_activity_logs(workflow_id: str, limit: int = 100):
    """Get activity logs for a specific workflow"""
    try:
        logs = workflow_manager.status_tracker.get_recent_activity_logs(limit, workflow_id)
        return {"logs": [log.dict() for log in logs], "count": len(logs)}
    except Exception as e:
        logger.error(f"Failed to get workflow activity logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow activity logs: {str(e)}")

@app.get("/api/activity-logs")
async def get_all_activity_logs(limit: int = 100):
    """Get recent activity logs for all workflows"""
    try:
        logs = workflow_manager.status_tracker.get_recent_activity_logs(limit)
        return {"logs": [log.dict() for log in logs], "count": len(logs)}
    except Exception as e:
        logger.error(f"Failed to get activity logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get activity logs: {str(e)}")

@app.get("/api/model-performance")
async def get_model_performance_history():
    """Get model performance history across all workflows"""
    try:
        performance_history = workflow_manager.status_tracker.model_performance_history
        return {"performance_history": performance_history}
    except Exception as e:
        logger.error(f"Failed to get model performance history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model performance history: {str(e)}")

@app.get("/api/system-resources")
async def get_current_system_resources():
    """Get current system resource usage"""
    try:
        resource_usage = workflow_manager.status_tracker._get_current_resource_usage()
        return resource_usage.dict()
    except Exception as e:
        logger.error(f"Failed to get system resources: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system resources: {str(e)}")

@app.post("/api/workflow/{workflow_id}/update-agent-status")
async def update_agent_status(workflow_id: str, agent_data: dict):
    """Update agent status for a workflow (for external integrations)"""
    try:
        workflow_manager.status_tracker.update_agent_status(workflow_id, agent_data)
        
        # Broadcast the update
        await websocket_manager.broadcast_agent_status(workflow_id, agent_data)
        
        return {"status": "success", "message": "Agent status updated"}
    except Exception as e:
        logger.error(f"Failed to update agent status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update agent status: {str(e)}")

@app.post("/api/workflow/{workflow_id}/update-model-performance")
async def update_model_performance(workflow_id: str, model_name: str, metrics: dict):
    """Update model performance metrics for a workflow"""
    try:
        workflow_manager.status_tracker.update_model_performance(workflow_id, model_name, metrics)
        
        # Broadcast the update
        await websocket_manager.broadcast_model_performance(workflow_id, {
            "model_name": model_name,
            "metrics": metrics
        })
        
        return {"status": "success", "message": "Model performance updated"}
    except Exception as e:
        logger.error(f"Failed to update model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update model performance: {str(e)}")

@app.post("/api/workflow/{workflow_id}/add-warning")
async def add_workflow_warning(workflow_id: str, warning: dict):
    """Add a warning to a workflow"""
    try:
        warning_message = warning.get("message", "Unknown warning")
        workflow_manager.status_tracker.add_warning(workflow_id, warning_message)
        
        # Broadcast the warning
        await websocket_manager.broadcast_activity_log({
            "timestamp": datetime.now().isoformat(),
            "level": "warning",
            "category": "system",
            "source": "api",
            "message": warning_message,
            "workflow_id": workflow_id
        })
        
        return {"status": "success", "message": "Warning added"}
    except Exception as e:
        logger.error(f"Failed to add workflow warning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add workflow warning: {str(e)}")

@app.delete("/api/workflow/{workflow_id}")
async def cleanup_workflow(workflow_id: str):
    """Clean up a completed workflow from tracking"""
    try:
        if workflow_id in workflow_manager.status_tracker.workflow_statuses:
            del workflow_manager.status_tracker.workflow_statuses[workflow_id]
            return {"status": "success", "message": "Workflow cleaned up"}
        else:
            raise HTTPException(status_code=404, detail="Workflow not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cleanup workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup workflow: {str(e)}")

@app.post("/api/cleanup-old-workflows")
async def cleanup_old_workflows(max_age_hours: int = 24):
    """Clean up old completed workflows"""
    try:
        workflow_manager.status_tracker.cleanup_completed_workflows(max_age_hours)
        return {"status": "success", "message": f"Cleaned up workflows older than {max_age_hours} hours"}
    except Exception as e:
        logger.error(f"Failed to cleanup old workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup old workflows: {str(e)}")

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

# LLM Shootout endpoints
class ShootoutRequest(BaseModel):
    document_id: str
    models: List[str]

@app.get("/llm-shootout")
async def serve_shootout_arena():
    """Serve the LLM Shootout Arena HTML file"""
    shootout_html_path = os.path.join(project_root, "frontend", "llm_shootout.html")
    return FileResponse(shootout_html_path)

@app.get("/api/llm-shootout/documents")
async def get_available_documents_for_shootout():
    """Get list of available documents for shootout"""
    try:
        documents = await llm_shootout_manager.get_available_documents()
        return documents
    except Exception as e:
        logger.error(f"Failed to get documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")

@app.get("/api/ollama/models")
async def get_ollama_models_for_shootout():
    """Get available Ollama models for shootout"""
    try:
        models = await llm_shootout_manager.discover_ollama_models()
        return models
    except Exception as e:
        logger.error(f"Failed to get Ollama models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get Ollama models: {str(e)}")

@app.post("/api/llm-shootout/start")
async def start_llm_shootout(request: ShootoutRequest):
    """Start an LLM shootout competition"""
    try:
        result = await llm_shootout_manager.start_shootout(request.document_id, request.models)
        return result
    except Exception as e:
        logger.error(f"Failed to start shootout: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start shootout: {str(e)}")

@app.post("/api/llm-shootout/stop")
async def stop_llm_shootout():
    """Stop the current LLM shootout competition"""
    try:
        result = await llm_shootout_manager.stop_shootout()
        return result
    except Exception as e:
        logger.error(f"Failed to stop shootout: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop shootout: {str(e)}")

@app.get("/api/llm-shootout/status")
async def get_shootout_status():
    """Get current shootout competition status"""
    try:
        status = await llm_shootout_manager.get_competition_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get shootout status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get shootout status: {str(e)}")

@app.websocket("/ws/llm-shootout")
async def shootout_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for LLM Shootout real-time updates"""
    await websocket.accept()
    
    # Add progress callback for shootout updates
    async def shootout_callback(message):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending shootout update: {e}")
    
    llm_shootout_manager.add_progress_callback(shootout_callback)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        llm_shootout_manager.remove_progress_callback(shootout_callback)
    except Exception as e:
        logger.error(f"Shootout WebSocket error: {str(e)}")
        llm_shootout_manager.remove_progress_callback(shootout_callback)

# GraphRAG endpoints
class GraphRAGQueryRequest(BaseModel):
    query: str
    max_results: int = 5
    use_graph_expansion: bool = True
    graph_depth: int = 2

@app.post("/api/graphrag/connect")
async def connect_graphrag():
    """Connect to GraphRAG system (Neo4j)"""
    try:
        success = await graph_rag_system.connect()
        if success:
            return {"status": "connected", "message": "GraphRAG system connected successfully"}
        else:
            return {"status": "failed", "message": "Failed to connect to Neo4j database"}
    except Exception as e:
        logger.error(f"GraphRAG connection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"GraphRAG connection failed: {str(e)}")

@app.post("/api/graphrag/process-document/{document_id}")
async def process_document_graphrag(document_id: str):
    """Process a document through GraphRAG pipeline"""
    try:
        # Find the document
        document_path = None
        if os.path.exists(UPLOAD_DIR):
            for filename in os.listdir(UPLOAD_DIR):
                if filename.startswith(document_id):
                    document_path = os.path.join(UPLOAD_DIR, filename)
                    break
        
        if not document_path or not os.path.exists(document_path):
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Read document content
        with open(document_path, 'r', encoding='utf-8') as f:
            document_text = f.read()
        
        # Process through GraphRAG
        result = await graph_rag_system.process_document(
            document_text, 
            document_id,
            {"filename": os.path.basename(document_path)}
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GraphRAG document processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.post("/api/graphrag/query")
async def query_graphrag(request: GraphRAGQueryRequest):
    """Query the GraphRAG system"""
    try:
        result = await graph_rag_system.query(
            request.query,
            request.max_results,
            request.use_graph_expansion,
            request.graph_depth
        )
        
        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "sources": result.sources,
            "reasoning_path": result.reasoning_path,
            "quality_score": result.quality_score,
            "processing_time": result.processing_time,
            "graph_entities": len(result.graph_context.get('entities', {})),
            "graph_relationships": len(result.graph_context.get('relationships', [])),
            "vector_documents": len(result.vector_context)
        }
        
    except Exception as e:
        logger.error(f"GraphRAG query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"GraphRAG query failed: {str(e)}")

@app.get("/api/graphrag/statistics")
async def get_graphrag_statistics():
    """Get GraphRAG system statistics"""
    try:
        stats = await graph_rag_system.get_graph_statistics()
        return stats
    except Exception as e:
        logger.error(f"Failed to get GraphRAG statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get GraphRAG statistics: {str(e)}")

@app.get("/api/graphrag/visualization")
async def get_graph_visualization(limit: int = 500):
    """Get graph visualization data"""
    try:
        data = await graph_rag_system.get_visualization_data(limit)
        return data
    except Exception as e:
        logger.error(f"Failed to get graph visualization data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get graph visualization data: {str(e)}")

@app.get("/api/graphrag/health")
async def graphrag_health_check():
    """GraphRAG system health check"""
    try:
        health = await graph_rag_system.health_check()
        return health
    except Exception as e:
        logger.error(f"GraphRAG health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"GraphRAG health check failed: {str(e)}")

# Enhanced Cleanup endpoints
@app.get("/api/cleanup/status")
async def get_cleanup_status():
    """Get current cleanup status"""
    try:
        status = await cleanup_manager.get_cleanup_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get cleanup status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cleanup status: {str(e)}")

@app.get("/api/cleanup/estimate/{operation}")
async def estimate_cleanup_impact(operation: str):
    """Estimate the impact of a cleanup operation"""
    try:
        valid_operations = ['clear_queued_documents', 'fresh_start_cleanup', 'clear_graph_only', 'optimize_graph']
        if operation not in valid_operations:
            raise HTTPException(status_code=400, detail=f"Invalid operation. Must be one of: {valid_operations}")
        
        impact = await cleanup_manager.estimate_cleanup_impact(operation)
        return impact
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to estimate cleanup impact: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to estimate cleanup impact: {str(e)}")

@app.post("/api/cleanup/clear-queued-documents")
async def clear_queued_documents():
    """Clear only uploaded documents waiting for processing"""
    try:
        result = await cleanup_manager.clear_queued_documents()
        return result
    except Exception as e:
        logger.error(f"Failed to clear queued documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear queued documents: {str(e)}")

@app.post("/api/cleanup/fresh-start")
async def fresh_start_cleanup():
    """Complete system reset for new dataset generation"""
    try:
        result = await cleanup_manager.fresh_start_cleanup()
        return result
    except Exception as e:
        logger.error(f"Fresh start cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fresh start cleanup failed: {str(e)}")

@app.post("/api/cleanup/clear-graph-only")
async def clear_graph_only():
    """Clear only the graph database, preserving documents and results"""
    try:
        result = await cleanup_manager.clear_graph_only()
        return result
    except Exception as e:
        logger.error(f"Graph-only cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Graph-only cleanup failed: {str(e)}")

@app.post("/api/cleanup/optimize-graph")
async def optimize_graph():
    """Optimize the graph database without full rebuild"""
    try:
        result = await cleanup_manager.optimize_graph()
        return result
    except Exception as e:
        logger.error(f"Graph optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Graph optimization failed: {str(e)}")

# Startup event to initialize GraphRAG
@app.on_event("startup")
async def startup_event():
    """Initialize GraphRAG system on startup"""
    try:
        logger.info("Initializing GraphRAG system...")
        success = await graph_rag_system.connect()
        if success:
            logger.info("GraphRAG system initialized successfully")
        else:
            logger.warning("GraphRAG system initialization failed - will use traditional RAG only")
    except Exception as e:
        logger.error(f"Failed to initialize GraphRAG system: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        await graph_rag_system.disconnect()
        logger.info("GraphRAG system disconnected")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
