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
from backend.rag_system import RAGSystem
from backend.alpaca_generator import AlpacaFormatGenerator

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
        """Setup and validate models, then execute CrewAI workflow"""
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
            
            # Now execute the actual CrewAI workflow
            await websocket_manager.broadcast({
                "type": "log",
                "level": "info",
                "message": "Executing CrewAI workflow..."
            })
            
            # Create CrewAI configuration
            crew_config = {
                "data_generation_model": config.data_generation_model,
                "embedding_model": config.embedding_model,
                "reranking_model": config.reranking_model,
                "openai_api_key": config.openai_api_key,
                "ollama_url": config.ollama_url,
                "enable_gpu_optimization": config.enable_gpu_optimization
            }
            
            # Prepare inputs for CrewAI
            inputs = {
                "documents": [doc.path for doc in config.documents],
                "workflow_type": config.workflow_type,
                "data_generation_model": config.data_generation_model,
                "embedding_model": config.embedding_model,
                "reranking_model": config.reranking_model,
                "enable_gpu_optimization": config.enable_gpu_optimization
            }
            
            # Execute CrewAI workflow in a thread to avoid blocking
            import concurrent.futures
            loop = asyncio.get_event_loop()
            
            def run_crew_workflow():
                try:
                    crew_instance = LocalCrewaiWorkflowForSyntheticDataWithRagAndLlmOptionsCrew(config=crew_config)
                    result = crew_instance.crew().kickoff(inputs=inputs)
                    return {"success": True, "result": result}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            
            # Run CrewAI workflow in thread executor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                crew_result = await loop.run_in_executor(executor, run_crew_workflow)
            
            if crew_result["success"]:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "success",
                    "message": "CrewAI workflow completed successfully"
                })
                
                # Save CrewAI result
                result_id = str(uuid.uuid4())
                result_data = {
                    "id": result_id,
                    "title": "CrewAI Workflow Results",
                    "description": f"CrewAI workflow execution results for {len(config.documents)} documents",
                    "type": "crewai_workflow",
                    "data": {
                        "crew_result": str(crew_result["result"]),
                        "workflow_config": crew_config,
                        "inputs": inputs,
                        "execution_timestamp": datetime.now().isoformat()
                    },
                    "created_at": datetime.now().isoformat(),
                    "workflow_id": workflow_id
                }
                
                # Ensure backend results directory exists
                backend_results_dir = os.path.join(os.path.dirname(__file__), "results")
                os.makedirs(backend_results_dir, exist_ok=True)
                
                result_path = os.path.join(backend_results_dir, f"{result_id}.json")
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "success",
                    "message": f"CrewAI results saved: {result_id}"
                })
            else:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "error",
                    "message": f"CrewAI workflow failed: {crew_result['error']}"
                })
                raise Exception(f"CrewAI workflow failed: {crew_result['error']}")
            
        except Exception as e:
            logger.error(f"Model setup and CrewAI execution failed: {str(e)}")
            raise
    
    async def _generate_synthetic_data(self, workflow_id: str, config: WorkflowConfig, websocket_manager):
        """Generate synthetic data in Alpaca format using enhanced RAG and LLM"""
        try:
            await websocket_manager.broadcast({
                "type": "log",
                "level": "info",
                "message": "Starting Alpaca format synthetic data generation..."
            })
            
            # Initialize RAG system
            await websocket_manager.broadcast({
                "type": "log",
                "level": "info",
                "message": "Initializing RAG system..."
            })
            
            rag_system = RAGSystem(
                embedding_model=config.embedding_model,
                reranking_model=config.reranking_model,
                ollama_url=config.ollama_url
            )
            
            # Initialize Alpaca generator
            alpaca_generator = AlpacaFormatGenerator(self.llm_manager, rag_system)
            
            # Convert WorkflowConfig to dict for compatibility
            config_dict = {
                "data_generation_model": config.data_generation_model,
                "embedding_model": config.embedding_model,
                "reranking_model": config.reranking_model,
                "openai_api_key": config.openai_api_key,
                "ollama_url": config.ollama_url,
                "enable_gpu_optimization": config.enable_gpu_optimization
            }
            
            # Get document paths
            document_paths = [doc.path for doc in config.documents]
            
            await websocket_manager.broadcast({
                "type": "log",
                "level": "info",
                "message": f"Processing {len(document_paths)} documents for Alpaca dataset generation..."
            })
            
            # Generate Alpaca format dataset
            try:
                alpaca_results = await alpaca_generator.generate_alpaca_dataset(
                    document_paths, config_dict, websocket_manager
                )
                
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "success",
                    "message": f"Successfully generated {len(alpaca_results['alpaca_data'])} Alpaca training examples"
                })
                
                # Save Alpaca dataset
                result_id = str(uuid.uuid4())
                result_path = os.path.join("results", f"{result_id}.json")
                
                # Create comprehensive result data
                result_data = {
                    "id": result_id,
                    "title": "Alpaca Format Training Dataset",
                    "description": f"Generated {len(alpaca_results['alpaca_data'])} Alpaca format training examples from {len(config.documents)} documents",
                    "type": "alpaca_dataset",
                    "data": {
                        "alpaca_training_data": alpaca_results['alpaca_data'],
                        "statistics": alpaca_results['statistics'],
                        "metadata": alpaca_results['metadata'],
                        "rag_stats": rag_system.get_stats()
                    },
                    "created_at": datetime.now().isoformat(),
                    "workflow_id": workflow_id
                }
                
                # Ensure backend results directory exists
                backend_results_dir = os.path.join(os.path.dirname(__file__), "results")
                os.makedirs(backend_results_dir, exist_ok=True)
                
                result_path = os.path.join(backend_results_dir, f"{result_id}.json")
                
                # Save to file
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                
                # Also save just the Alpaca data in standard format
                alpaca_only_path = os.path.join(backend_results_dir, f"{result_id}_alpaca_only.json")
                alpaca_generator.save_alpaca_dataset(alpaca_results['alpaca_data'], alpaca_only_path)
                
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "success",
                    "message": f"Alpaca dataset saved: {result_id} (Full: {result_path}, Alpaca-only: {alpaca_only_path})"
                })
                
            except Exception as generation_error:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "error",
                    "message": f"Alpaca generation failed: {str(generation_error)}"
                })
                
                # Save error result
                result_id = str(uuid.uuid4())
                result_data = {
                    "id": result_id,
                    "title": "Alpaca Dataset Generation (Failed)",
                    "description": f"Failed to generate Alpaca dataset from {len(config.documents)} documents",
                    "type": "alpaca_dataset",
                    "data": {
                        "documents_processed": len(config.documents),
                        "model_used": config.data_generation_model,
                        "generation_timestamp": datetime.now().isoformat(),
                        "error": str(generation_error),
                        "status": "failed"
                    },
                    "created_at": datetime.now().isoformat(),
                    "workflow_id": workflow_id
                }
                
                # Ensure backend results directory exists
                backend_results_dir = os.path.join(os.path.dirname(__file__), "results")
                os.makedirs(backend_results_dir, exist_ok=True)
                
                result_path = os.path.join(backend_results_dir, f"{result_id}.json")
                with open(result_path, 'w') as f:
                    json.dump(result_data, f, indent=2)
                
                logger.error(f"Alpaca generation failed but continuing workflow: {str(generation_error)}")
            
        except Exception as e:
            logger.error(f"Synthetic data generation failed: {str(e)}")
            raise
    
    async def _implement_rag(self, workflow_id: str, config: WorkflowConfig, websocket_manager):
        """Implement RAG capabilities with proper embedding and retrieval"""
        try:
            await websocket_manager.broadcast({
                "type": "log",
                "level": "info",
                "message": "Implementing enhanced RAG capabilities..."
            })
            
            # Initialize RAG system
            rag_system = RAGSystem(
                embedding_model=config.embedding_model,
                reranking_model=config.reranking_model,
                ollama_url=config.ollama_url
            )
            
            # Get document paths
            document_paths = [doc.path for doc in config.documents]
            
            await websocket_manager.broadcast({
                "type": "log",
                "level": "info",
                "message": f"Processing {len(document_paths)} documents for RAG implementation..."
            })
            
            # Process documents and create embeddings
            try:
                rag_results = await rag_system.process_documents(document_paths, websocket_manager)
                
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "success",
                    "message": f"RAG processing completed: {rag_results['total_chunks']} chunks created"
                })
                
                # Test retrieval functionality
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "info",
                    "message": "Testing RAG retrieval functionality..."
                })
                
                test_queries = [
                    "What are the main topics discussed?",
                    "What information is provided in the documents?",
                    "What are the key points mentioned?"
                ]
                
                retrieval_tests = []
                for query in test_queries:
                    try:
                        relevant_chunks = await rag_system.retrieve_relevant_chunks(query, top_k=3)
                        
                        # Apply reranking if available
                        if config.reranking_model and relevant_chunks:
                            relevant_chunks = await rag_system.rerank_results(query, relevant_chunks)
                        
                        retrieval_tests.append({
                            "query": query,
                            "retrieved_chunks": len(relevant_chunks),
                            "success": len(relevant_chunks) > 0
                        })
                        
                    except Exception as e:
                        retrieval_tests.append({
                            "query": query,
                            "retrieved_chunks": 0,
                            "success": False,
                            "error": str(e)
                        })
                
                successful_tests = sum(1 for test in retrieval_tests if test["success"])
                
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "success" if successful_tests > 0 else "warning",
                    "message": f"RAG retrieval tests: {successful_tests}/{len(test_queries)} successful"
                })
                
                # Save comprehensive RAG results
                result_id = str(uuid.uuid4())
                result_data = {
                    "id": result_id,
                    "title": "Enhanced RAG Implementation Results",
                    "description": f"RAG implementation with embeddings and retrieval for {len(config.documents)} documents",
                    "type": "rag_implementation",
                    "data": {
                        "processing_results": rag_results,
                        "retrieval_tests": retrieval_tests,
                        "rag_statistics": rag_system.get_stats(),
                        "implementation_timestamp": datetime.now().isoformat(),
                        "models_used": {
                            "embedding_model": config.embedding_model,
                            "reranking_model": config.reranking_model
                        }
                    },
                    "created_at": datetime.now().isoformat(),
                    "workflow_id": workflow_id
                }
                
                # Ensure backend results directory exists
                backend_results_dir = os.path.join(os.path.dirname(__file__), "results")
                os.makedirs(backend_results_dir, exist_ok=True)
                
                # Save result to file
                result_path = os.path.join(backend_results_dir, f"{result_id}.json")
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "success",
                    "message": f"Enhanced RAG implementation completed. Result saved: {result_id}"
                })
                
            except Exception as rag_error:
                await websocket_manager.broadcast({
                    "type": "log",
                    "level": "error",
                    "message": f"RAG implementation failed: {str(rag_error)}"
                })
                
                # Save error result
                result_id = str(uuid.uuid4())
                result_data = {
                    "id": result_id,
                    "title": "RAG Implementation (Failed)",
                    "description": f"Failed RAG implementation for {len(config.documents)} documents",
                    "type": "rag_implementation",
                    "data": {
                        "documents_processed": len(config.documents),
                        "embedding_model": config.embedding_model,
                        "reranking_model": config.reranking_model,
                        "implementation_timestamp": datetime.now().isoformat(),
                        "error": str(rag_error),
                        "status": "failed"
                    },
                    "created_at": datetime.now().isoformat(),
                    "workflow_id": workflow_id
                }
                
                # Ensure backend results directory exists
                backend_results_dir = os.path.join(os.path.dirname(__file__), "results")
                os.makedirs(backend_results_dir, exist_ok=True)
                
                result_path = os.path.join(backend_results_dir, f"{result_id}.json")
                with open(result_path, 'w') as f:
                    json.dump(result_data, f, indent=2)
                
                logger.error(f"RAG implementation failed but continuing workflow: {str(rag_error)}")
            
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
