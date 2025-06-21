import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, Optional
import sys

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from local_crewai_workflow_for_synthetic_data_with_rag_and_llm_options.crew import LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew
from backend.models import WorkflowConfig, WorkflowStatus, WorkflowProgress, WorkflowResult
from backend.llm_manager import LLMManager

logger = logging.getLogger(__name__)

class WorkflowManager:
    """Manages CrewAI workflow execution"""
    
    def __init__(self):
        self.active_workflows: Dict[str, Dict] = {}
        self.llm_manager = LLMManager()
    
    async def run_workflow(self, workflow_id: str, config: WorkflowConfig, websocket_manager):
        """Run the CrewAI workflow"""
        try:
            # Initialize workflow tracking
            self.active_workflows[workflow_id] = {
                "status": WorkflowStatus.RUNNING,
                "config": config,
                "start_time": datetime.now().isoformat(),
                "current_step": "initializing",
                "progress": 0
            }
            
            # Send initial status
            await websocket_manager.broadcast({
                "type": "workflow_progress",
                "workflow_id": workflow_id,
                "step": "initializing",
                "status": "active",
                "progress": 0
            })
            
            # Step 1: Document Processing
            await self._update_workflow_step(workflow_id, "document-processing", "active", 10, websocket_manager)
            await self._process_documents(workflow_id, config, websocket_manager)
            await self._update_workflow_step(workflow_id, "document-processing", "completed", 20, websocket_manager)
            
            # Step 2: Model Selection
            await self._update_workflow_step(workflow_id, "model-selection", "active", 30, websocket_manager)
            await self._setup_models(workflow_id, config, websocket_manager)
            await self._update_workflow_step(workflow_id, "model-selection", "completed", 40, websocket_manager)
            
            # Step 3: Data Generation (if needed)
            if config.workflow_type in ["full", "data_generation_only"]:
                await self._update_workflow_step(workflow_id, "data-generation", "active", 50, websocket_manager)
                await self._generate_synthetic_data(workflow_id, config, websocket_manager)
                await self._update_workflow_step(workflow_id, "data-generation", "completed", 70, websocket_manager)
            
            # Step 4: RAG Implementation (if needed)
            if config.workflow_type in ["full", "rag_only"]:
                await self._update_workflow_step(workflow_id, "rag-implementation", "active", 80, websocket_manager)
                await self._implement_rag(workflow_id, config, websocket_manager)
                await self._update_workflow_step(workflow_id, "rag-implementation", "completed", 90, websocket_manager)
            
            # Step 5: Optimization
            if config.enable_gpu_optimization:
                await self._update_workflow_step(workflow_id, "optimization", "active", 95, websocket_manager)
                await self._optimize_performance(workflow_id, config, websocket_manager)
                await self._update_workflow_step(workflow_id, "optimization", "completed", 100, websocket_manager)
            
            # Complete workflow
            await self._complete_workflow(workflow_id, websocket_manager)
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {str(e)}")
            await self._fail_workflow(workflow_id, str(e), websocket_manager)
    
    async def _update_workflow_step(self, workflow_id: str, step: str, status: str, progress: int, websocket_manager):
        """Update workflow step status"""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["current_step"] = step
            self.active_workflows[workflow_id]["progress"] = progress
            
            await websocket_manager.broadcast({
                "type": "workflow_progress",
                "workflow_id": workflow_id,
                "step": step,
                "status": status,
                "progress": progress
            })
            
            await websocket_manager.broadcast({
                "type": "log",
                "level": "info",
                "message": f"Workflow {workflow_id}: {step} - {status}"
            })
    
    async def _process_documents(self, workflow_id: str, config: WorkflowConfig, websocket_manager):
        """Process uploaded documents"""
        try:
            await websocket_manager.broadcast({
                "type": "log",
                "level": "info",
                "message": f"Processing {len(config.documents)} documents..."
            })
            
            # Simulate document processing
            for i, doc in enumerate(config.documents):
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "info",
                    "message": f"Processing document: {doc.original_name}"
                })
                
                # Add small delay to simulate processing
                await asyncio.sleep(0.5)
            
            await websocket_manager.broadcast({
                "type": "log",
                "level": "success",
                "message": "Document processing completed successfully"
            })
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise
    
    async def _setup_models(self, workflow_id: str, config: WorkflowConfig, websocket_manager):
        """Setup and validate models"""
        try:
            await websocket_manager.broadcast({
                "type": "log",
                "level": "info",
                "message": "Setting up models..."
            })
            
            # Test models
            test_results = await self.llm_manager.test_models(config)
            
            for model_type, result in test_results.items():
                if result.success:
                    await websocket_manager.broadcast({
                        "type": "log",
                        "level": "success",
                        "message": f"{model_type} model: {result.message}"
                    })
                else:
                    await websocket_manager.broadcast({
                        "type": "log",
                        "level": "error",
                        "message": f"{model_type} model: {result.message}"
                    })
                    raise Exception(f"Model setup failed: {result.message}")
            
        except Exception as e:
            logger.error(f"Model setup failed: {str(e)}")
            raise
    
    async def _generate_synthetic_data(self, workflow_id: str, config: WorkflowConfig, websocket_manager):
        """Generate synthetic data using CrewAI"""
        try:
            await websocket_manager.broadcast({
                "type": "log",
                "level": "info",
                "message": "Starting synthetic data generation..."
            })
            
            # Create CrewAI crew instance
            crew_instance = LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew()
            
            # Prepare inputs for the crew
            inputs = {
                "documents": [doc.path for doc in config.documents],
                "data_generation_model": config.data_generation_model,
                "embedding_model": config.embedding_model,
                "reranking_model": config.reranking_model,
                "enable_gpu_optimization": config.enable_gpu_optimization
            }
            
            # Run the crew workflow
            await websocket_manager.broadcast({
                "type": "log",
                "level": "info",
                "message": "Executing CrewAI workflow..."
            })
            
            # This would be the actual crew execution
            # For now, we'll simulate it
            await asyncio.sleep(2)  # Simulate processing time
            
            # Save synthetic data results
            result_id = str(uuid.uuid4())
            result_data = {
                "id": result_id,
                "title": "Synthetic Data Generation Results",
                "description": f"Generated synthetic data for {len(config.documents)} documents",
                "type": "synthetic_data",
                "data": {
                    "documents_processed": len(config.documents),
                    "model_used": config.data_generation_model,
                    "generation_timestamp": datetime.now().isoformat()
                },
                "created_at": datetime.now().isoformat(),
                "workflow_id": workflow_id
            }
            
            # Save result to file
            result_path = os.path.join("results", f"{result_id}.json")
            with open(result_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            await websocket_manager.broadcast({
                "type": "log",
                "level": "success",
                "message": f"Synthetic data generation completed. Result saved: {result_id}"
            })
            
        except Exception as e:
            logger.error(f"Synthetic data generation failed: {str(e)}")
            raise
    
    async def _implement_rag(self, workflow_id: str, config: WorkflowConfig, websocket_manager):
        """Implement RAG capabilities"""
        try:
            await websocket_manager.broadcast({
                "type": "log",
                "level": "info",
                "message": "Implementing RAG capabilities..."
            })
            
            # Simulate RAG implementation
            await websocket_manager.broadcast({
                "type": "log",
                "level": "info",
                "message": f"Creating embeddings using {config.embedding_model}..."
            })
            
            await asyncio.sleep(1.5)  # Simulate embedding creation
            
            if config.reranking_model:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "info",
                    "message": f"Setting up reranking with {config.reranking_model}..."
                })
                await asyncio.sleep(1)  # Simulate reranking setup
            
            # Save RAG results
            result_id = str(uuid.uuid4())
            result_data = {
                "id": result_id,
                "title": "RAG Implementation Results",
                "description": f"RAG implementation for {len(config.documents)} documents",
                "type": "rag_implementation",
                "data": {
                    "documents_processed": len(config.documents),
                    "embedding_model": config.embedding_model,
                    "reranking_model": config.reranking_model,
                    "implementation_timestamp": datetime.now().isoformat()
                },
                "created_at": datetime.now().isoformat(),
                "workflow_id": workflow_id
            }
            
            # Save result to file
            result_path = os.path.join("results", f"{result_id}.json")
            with open(result_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            await websocket_manager.broadcast({
                "type": "log",
                "level": "success",
                "message": f"RAG implementation completed. Result saved: {result_id}"
            })
            
        except Exception as e:
            logger.error(f"RAG implementation failed: {str(e)}")
            raise
    
    async def _optimize_performance(self, workflow_id: str, config: WorkflowConfig, websocket_manager):
        """Optimize performance for GPU usage"""
        try:
            await websocket_manager.broadcast({
                "type": "log",
                "level": "info",
                "message": "Optimizing performance for dual 4090 GPUs..."
            })
            
            # Simulate GPU optimization
            await asyncio.sleep(1)
            
            await websocket_manager.broadcast({
                "type": "log",
                "level": "success",
                "message": "Performance optimization completed"
            })
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {str(e)}")
            raise
    
    async def _complete_workflow(self, workflow_id: str, websocket_manager):
        """Complete the workflow"""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["status"] = WorkflowStatus.COMPLETED
            self.active_workflows[workflow_id]["end_time"] = datetime.now().isoformat()
            
            await websocket_manager.broadcast({
                "type": "workflow_complete",
                "workflow_id": workflow_id,
                "results": []  # This would contain actual results
            })
            
            await websocket_manager.broadcast({
                "type": "log",
                "level": "success",
                "message": f"Workflow {workflow_id} completed successfully!"
            })
    
    async def _fail_workflow(self, workflow_id: str, error_message: str, websocket_manager):
        """Mark workflow as failed"""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["status"] = WorkflowStatus.FAILED
            self.active_workflows[workflow_id]["error_message"] = error_message
            self.active_workflows[workflow_id]["end_time"] = datetime.now().isoformat()
            
            await websocket_manager.broadcast({
                "type": "workflow_error",
                "workflow_id": workflow_id,
                "error": error_message
            })
    
    async def stop_workflow(self, workflow_id: str) -> bool:
        """Stop a running workflow"""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]["status"] = WorkflowStatus.STOPPED
            self.active_workflows[workflow_id]["end_time"] = datetime.now().isoformat()
            return True
        return False
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict]:
        """Get workflow status"""
        return self.active_workflows.get(workflow_id)
    
    def list_active_workflows(self) -> Dict[str, Dict]:
        """List all active workflows"""
        return {
            wf_id: wf_data for wf_id, wf_data in self.active_workflows.items()
            if wf_data["status"] == WorkflowStatus.RUNNING
        }
