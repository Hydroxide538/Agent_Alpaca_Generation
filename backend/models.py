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

# Enhanced Status Models for Rich Visualization

class AgentStatus(BaseModel):
    agent_id: str
    agent_name: str
    current_task: str
    status: str  # "idle", "working", "completed", "error"
    progress: int  # 0-100
    start_time: Optional[str] = None
    estimated_completion: Optional[str] = None
    performance_metrics: Dict[str, Any] = {}

class ModelPerformanceMetrics(BaseModel):
    model_name: str
    response_time_ms: float
    tokens_per_second: float
    success_rate: float
    error_count: int
    total_requests: int
    average_quality_score: Optional[float] = None
    resource_usage: Dict[str, Any] = {}

class SubStepProgress(BaseModel):
    step_id: str
    step_name: str
    description: str
    status: str  # "pending", "active", "completed", "error"
    progress: int  # 0-100
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    details: Dict[str, Any] = {}

class ProcessingStats(BaseModel):
    documents_processed: int
    total_documents: int
    chunks_created: int
    embeddings_generated: int
    tokens_processed: int
    processing_rate: float  # items per second
    estimated_time_remaining: Optional[float] = None

class SystemResourceUsage(BaseModel):
    cpu_usage: float
    memory_usage: float
    gpu_usage: List[Dict[str, Any]] = []
    disk_usage: float
    network_io: Dict[str, float] = {}
    timestamp: str

class DetailedWorkflowStatus(BaseModel):
    workflow_id: str
    status: WorkflowStatus
    current_step: str
    current_substep: Optional[str] = None
    progress_percentage: int
    substep_progress: int = 0
    active_agents: List[AgentStatus] = []
    model_performance: List[ModelPerformanceMetrics] = []
    processing_stats: ProcessingStats
    resource_usage: SystemResourceUsage
    sub_steps: List[SubStepProgress] = []
    error_message: Optional[str] = None
    warnings: List[str] = []
    start_time: str
    estimated_completion: Optional[str] = None
    quality_metrics: Dict[str, Any] = {}

class ActivityLogEntry(BaseModel):
    timestamp: str
    level: str  # "info", "warning", "error", "debug", "success"
    category: str  # "system", "agent", "model", "processing", "user"
    source: str  # agent name, model name, or system component
    message: str
    details: Dict[str, Any] = {}
    workflow_id: Optional[str] = None

class CrewAIAgentActivity(BaseModel):
    agent_name: str
    role: str
    current_goal: str
    current_task: str
    task_progress: int
    thoughts: Optional[str] = None
    tools_used: List[str] = []
    collaboration_status: str  # "independent", "waiting", "collaborating"
    performance_score: Optional[float] = None
    execution_time: Optional[float] = None
