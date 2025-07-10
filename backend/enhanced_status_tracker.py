import asyncio
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from backend.models import (
    DetailedWorkflowStatus, AgentStatus, ModelPerformanceMetrics,
    SubStepProgress, ProcessingStats, SystemResourceUsage,
    ActivityLogEntry, CrewAIAgentActivity, WorkflowStatus
)

logger = logging.getLogger(__name__)

class EnhancedStatusTracker:
    """Enhanced status tracking for detailed workflow visualization"""
    
    def __init__(self):
        self.workflow_statuses: Dict[str, DetailedWorkflowStatus] = {}
        self.activity_logs: List[ActivityLogEntry] = []
        self.model_performance_history: Dict[str, List[ModelPerformanceMetrics]] = {}
        self.resource_monitoring_active = False
        self.resource_monitor_task = None
        
    def initialize_workflow(self, workflow_id: str, config: dict) -> DetailedWorkflowStatus:
        """Initialize detailed tracking for a new workflow"""
        
        # Initialize processing stats
        processing_stats = ProcessingStats(
            documents_processed=0,
            total_documents=len(config.get('documents', [])),
            chunks_created=0,
            embeddings_generated=0,
            tokens_processed=0,
            processing_rate=0.0,
            estimated_time_remaining=None
        )
        
        # Initialize resource usage
        resource_usage = self._get_current_resource_usage()
        
        # Create detailed workflow status
        detailed_status = DetailedWorkflowStatus(
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            current_step="initializing",
            current_substep=None,
            progress_percentage=0,
            substep_progress=0,
            active_agents=[],
            model_performance=[],
            processing_stats=processing_stats,
            resource_usage=resource_usage,
            sub_steps=[],
            error_message=None,
            warnings=[],
            start_time=datetime.now().isoformat(),
            estimated_completion=None,
            quality_metrics={}
        )
        
        self.workflow_statuses[workflow_id] = detailed_status
        
        # Start resource monitoring if not already active
        if not self.resource_monitoring_active:
            self.start_resource_monitoring()
        
        return detailed_status
    
    def update_workflow_step(self, workflow_id: str, step: str, status: str, progress: int):
        """Update main workflow step"""
        if workflow_id not in self.workflow_statuses:
            return
        
        workflow_status = self.workflow_statuses[workflow_id]
        workflow_status.current_step = step
        workflow_status.progress_percentage = progress
        
        # Log the step change
        self.add_activity_log(
            level="info",
            category="system",
            source="workflow_manager",
            message=f"Workflow step updated: {step} ({status})",
            workflow_id=workflow_id
        )
    
    def update_substep(self, workflow_id: str, step: str, substep_data: dict):
        """Update sub-step progress within a main step"""
        if workflow_id not in self.workflow_statuses:
            return
        
        workflow_status = self.workflow_statuses[workflow_id]
        workflow_status.current_substep = substep_data.get('name')
        workflow_status.substep_progress = substep_data.get('progress', 0)
        
        # Create or update sub-step
        substep = SubStepProgress(
            step_id=substep_data.get('id', f"{step}_{substep_data.get('name', 'unknown')}"),
            step_name=substep_data.get('name', 'Unknown Sub-step'),
            description=substep_data.get('description', ''),
            status=substep_data.get('status', 'active'),
            progress=substep_data.get('progress', 0),
            start_time=substep_data.get('start_time', datetime.now().isoformat()),
            end_time=substep_data.get('end_time'),
            details=substep_data.get('details', {})
        )
        
        # Update or add sub-step
        existing_substep = None
        for i, existing in enumerate(workflow_status.sub_steps):
            if existing.step_id == substep.step_id:
                workflow_status.sub_steps[i] = substep
                existing_substep = substep
                break
        
        if not existing_substep:
            workflow_status.sub_steps.append(substep)
    
    def update_agent_status(self, workflow_id: str, agent_data: dict):
        """Update CrewAI agent status"""
        if workflow_id not in self.workflow_statuses:
            return
        
        workflow_status = self.workflow_statuses[workflow_id]
        
        agent_status = AgentStatus(
            agent_id=agent_data.get('agent_id', agent_data.get('agent_name', 'unknown')),
            agent_name=agent_data.get('agent_name', 'Unknown Agent'),
            current_task=agent_data.get('current_task', 'No task assigned'),
            status=agent_data.get('status', 'idle'),
            progress=agent_data.get('progress', 0),
            start_time=agent_data.get('start_time'),
            estimated_completion=agent_data.get('estimated_completion'),
            performance_metrics=agent_data.get('performance_metrics', {})
        )
        
        # Update or add agent status
        for i, existing_agent in enumerate(workflow_status.active_agents):
            if existing_agent.agent_id == agent_status.agent_id:
                workflow_status.active_agents[i] = agent_status
                return
        
        workflow_status.active_agents.append(agent_status)
    
    def update_model_performance(self, workflow_id: str, model_name: str, metrics: dict):
        """Update model performance metrics"""
        if workflow_id not in self.workflow_statuses:
            return
        
        workflow_status = self.workflow_statuses[workflow_id]
        
        # Calculate performance metrics
        response_time = metrics.get('response_time_ms', 0.0)
        tokens_generated = metrics.get('tokens_generated', 0)
        time_taken = metrics.get('time_taken_seconds', 1.0)
        
        model_metrics = ModelPerformanceMetrics(
            model_name=model_name,
            response_time_ms=response_time,
            tokens_per_second=tokens_generated / max(time_taken, 0.001),
            success_rate=metrics.get('success_rate', 1.0),
            error_count=metrics.get('error_count', 0),
            total_requests=metrics.get('total_requests', 1),
            average_quality_score=metrics.get('quality_score'),
            resource_usage=metrics.get('resource_usage', {})
        )
        
        # Update or add model performance
        for i, existing_model in enumerate(workflow_status.model_performance):
            if existing_model.model_name == model_name:
                workflow_status.model_performance[i] = model_metrics
                break
        else:
            workflow_status.model_performance.append(model_metrics)
        
        # Store in history
        if model_name not in self.model_performance_history:
            self.model_performance_history[model_name] = []
        self.model_performance_history[model_name].append(model_metrics)
        
        # Keep only last 100 entries per model
        if len(self.model_performance_history[model_name]) > 100:
            self.model_performance_history[model_name] = self.model_performance_history[model_name][-100:]
    
    def update_processing_stats(self, workflow_id: str, stats: dict):
        """Update processing statistics"""
        if workflow_id not in self.workflow_statuses:
            return
        
        workflow_status = self.workflow_statuses[workflow_id]
        
        # Calculate processing rate
        current_time = time.time()
        start_time = datetime.fromisoformat(workflow_status.start_time).timestamp()
        elapsed_time = current_time - start_time
        
        documents_processed = stats.get('documents_processed', workflow_status.processing_stats.documents_processed)
        processing_rate = documents_processed / max(elapsed_time, 1.0) if elapsed_time > 0 else 0.0
        
        # Estimate time remaining
        remaining_docs = workflow_status.processing_stats.total_documents - documents_processed
        estimated_time_remaining = remaining_docs / max(processing_rate, 0.001) if processing_rate > 0 else None
        
        workflow_status.processing_stats = ProcessingStats(
            documents_processed=documents_processed,
            total_documents=workflow_status.processing_stats.total_documents,
            chunks_created=stats.get('chunks_created', workflow_status.processing_stats.chunks_created),
            embeddings_generated=stats.get('embeddings_generated', workflow_status.processing_stats.embeddings_generated),
            tokens_processed=stats.get('tokens_processed', workflow_status.processing_stats.tokens_processed),
            processing_rate=processing_rate,
            estimated_time_remaining=estimated_time_remaining
        )
    
    def add_activity_log(self, level: str, category: str, source: str, message: str, 
                        details: dict = None, workflow_id: str = None):
        """Add an activity log entry"""
        log_entry = ActivityLogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            category=category,
            source=source,
            message=message,
            details=details or {},
            workflow_id=workflow_id
        )
        
        self.activity_logs.append(log_entry)
        
        # Keep only last 1000 log entries
        if len(self.activity_logs) > 1000:
            self.activity_logs = self.activity_logs[-1000:]
    
    def add_crew_agent_activity(self, workflow_id: str, agent_activity: CrewAIAgentActivity):
        """Add CrewAI specific agent activity"""
        self.add_activity_log(
            level="info",
            category="agent",
            source=agent_activity.agent_name,
            message=f"Agent {agent_activity.agent_name} ({agent_activity.role}): {agent_activity.current_task}",
            details={
                "goal": agent_activity.current_goal,
                "progress": agent_activity.task_progress,
                "thoughts": agent_activity.thoughts,
                "tools_used": agent_activity.tools_used,
                "collaboration_status": agent_activity.collaboration_status,
                "performance_score": agent_activity.performance_score,
                "execution_time": agent_activity.execution_time
            },
            workflow_id=workflow_id
        )
    
    def update_quality_metrics(self, workflow_id: str, metrics: dict):
        """Update quality assessment metrics"""
        if workflow_id not in self.workflow_statuses:
            return
        
        workflow_status = self.workflow_statuses[workflow_id]
        workflow_status.quality_metrics.update(metrics)
    
    def add_warning(self, workflow_id: str, warning_message: str):
        """Add a warning to the workflow"""
        if workflow_id not in self.workflow_statuses:
            return
        
        workflow_status = self.workflow_statuses[workflow_id]
        workflow_status.warnings.append(warning_message)
        
        self.add_activity_log(
            level="warning",
            category="system",
            source="workflow_manager",
            message=warning_message,
            workflow_id=workflow_id
        )
    
    def set_error(self, workflow_id: str, error_message: str):
        """Set workflow error status"""
        if workflow_id not in self.workflow_statuses:
            return
        
        workflow_status = self.workflow_statuses[workflow_id]
        workflow_status.status = WorkflowStatus.FAILED
        workflow_status.error_message = error_message
        
        self.add_activity_log(
            level="error",
            category="system",
            source="workflow_manager",
            message=f"Workflow failed: {error_message}",
            workflow_id=workflow_id
        )
    
    def complete_workflow(self, workflow_id: str):
        """Mark workflow as completed"""
        if workflow_id not in self.workflow_statuses:
            return
        
        workflow_status = self.workflow_statuses[workflow_id]
        workflow_status.status = WorkflowStatus.COMPLETED
        workflow_status.progress_percentage = 100
        
        # Complete all active sub-steps
        for substep in workflow_status.sub_steps:
            if substep.status == "active":
                substep.status = "completed"
                substep.progress = 100
                substep.end_time = datetime.now().isoformat()
        
        # Mark all agents as completed
        for agent in workflow_status.active_agents:
            if agent.status == "working":
                agent.status = "completed"
                agent.progress = 100
        
        self.add_activity_log(
            level="success",
            category="system",
            source="workflow_manager",
            message="Workflow completed successfully",
            workflow_id=workflow_id
        )
    
    def get_workflow_status(self, workflow_id: str) -> Optional[DetailedWorkflowStatus]:
        """Get detailed workflow status"""
        return self.workflow_statuses.get(workflow_id)
    
    def get_recent_activity_logs(self, limit: int = 100, workflow_id: str = None) -> List[ActivityLogEntry]:
        """Get recent activity logs"""
        logs = self.activity_logs
        
        if workflow_id:
            logs = [log for log in logs if log.workflow_id == workflow_id]
        
        return logs[-limit:] if logs else []
    
    def _get_current_resource_usage(self) -> SystemResourceUsage:
        """Get current system resource usage"""
        try:
            # CPU and Memory
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # GPU usage (if available)
            gpu_usage = []
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_usage.append({
                        "id": gpu.id,
                        "name": gpu.name,
                        "load": gpu.load * 100,
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "temperature": gpu.temperature
                    })
            except ImportError:
                pass  # GPUtil not available
            except Exception as e:
                logger.warning(f"Could not get GPU usage: {e}")
            
            # Disk usage
            disk_usage = psutil.disk_usage('/').percent
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_stats = {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv
            }
            
            return SystemResourceUsage(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                gpu_usage=gpu_usage,
                disk_usage=disk_usage,
                network_io=network_stats,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return SystemResourceUsage(
                cpu_usage=0.0,
                memory_usage=0.0,
                gpu_usage=[],
                disk_usage=0.0,
                network_io={},
                timestamp=datetime.now().isoformat()
            )
    
    def start_resource_monitoring(self):
        """Start background resource monitoring"""
        if self.resource_monitoring_active:
            return
        
        self.resource_monitoring_active = True
        self.resource_monitor_task = asyncio.create_task(self._resource_monitor_loop())
    
    def stop_resource_monitoring(self):
        """Stop background resource monitoring"""
        self.resource_monitoring_active = False
        if self.resource_monitor_task:
            self.resource_monitor_task.cancel()
    
    async def _resource_monitor_loop(self):
        """Background loop for resource monitoring"""
        try:
            while self.resource_monitoring_active:
                # Update resource usage for all active workflows
                for workflow_id, workflow_status in self.workflow_statuses.items():
                    if workflow_status.status == WorkflowStatus.RUNNING:
                        workflow_status.resource_usage = self._get_current_resource_usage()
                
                # Wait 5 seconds before next update
                await asyncio.sleep(5)
                
        except asyncio.CancelledError:
            logger.info("Resource monitoring stopped")
        except Exception as e:
            logger.error(f"Error in resource monitoring loop: {e}")
    
    def cleanup_completed_workflows(self, max_age_hours: int = 24):
        """Clean up old completed workflows"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        workflows_to_remove = []
        for workflow_id, workflow_status in self.workflow_statuses.items():
            if workflow_status.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
                start_time = datetime.fromisoformat(workflow_status.start_time)
                if start_time < cutoff_time:
                    workflows_to_remove.append(workflow_id)
        
        for workflow_id in workflows_to_remove:
            del self.workflow_statuses[workflow_id]
        
        logger.info(f"Cleaned up {len(workflows_to_remove)} old workflows")
