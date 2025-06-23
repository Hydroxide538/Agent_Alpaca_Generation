from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum

class WorkflowType(str, Enum):
    FULL = "full"
    DATA_GENERATION_ONLY = "data_generation_only"
    RAG_ONLY = "rag_only"

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

class DocumentInfo(BaseModel):
    id: str
    original_name: str
    filename: str
    path: str
    size: int
    type: str
    token_count: Optional[int] = None
    character_count: Optional[int] = None
    word_count: Optional[int] = None
    encoding: Optional[str] = None

class WorkflowConfig(BaseModel):
    manager_model: str
    selection_strategy: str = "performance_based"
    embedding_model: str
    reranking_model: Optional[str] = None
    data_generation_model: Optional[str] = None
    openai_api_key: Optional[str] = None
    ollama_url: str = "http://host.docker.internal:11434"
    enable_gpu_optimization: bool = True
    documents: List[DocumentInfo] = []
    workflow_type: WorkflowType = WorkflowType.FULL

class TestResult(BaseModel):
    success: bool
    message: str
    response_time: Optional[float] = None
    error: Optional[str] = None

class WorkflowProgress(BaseModel):
    workflow_id: str
    status: WorkflowStatus
    current_step: str
    progress_percentage: int
    steps_completed: List[str]
    steps_remaining: List[str]
    error_message: Optional[str] = None
    start_time: str
    end_time: Optional[str] = None

class WorkflowResult(BaseModel):
    id: str
    title: str
    description: str
    type: str
    data: Dict[str, Any]
    created_at: str
    workflow_id: str

class SystemInfo(BaseModel):
    cpu: Dict[str, Any]
    memory: Dict[str, Any]
    gpu: List[Dict[str, Any]]
    timestamp: str

class OllamaModel(BaseModel):
    name: str
    size: int
    digest: str
    modified_at: str

class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: str
