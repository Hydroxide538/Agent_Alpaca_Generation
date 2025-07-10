import asyncio
import json
import logging
from typing import List, Dict, Any
from fastapi import WebSocket
from datetime import datetime

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific WebSocket connection"""
        try:
            message_with_timestamp = {
                **message,
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_text(json.dumps(message_with_timestamp))
        except Exception as e:
            logger.error(f"Error sending personal message: {str(e)}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected WebSocket clients"""
        if not self.active_connections:
            return
        
        message_with_timestamp = {
            **message,
            "timestamp": datetime.now().isoformat()
        }
        
        message_text = json.dumps(message_with_timestamp)
        
        # Send to all connections and remove any that fail
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message_text)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {str(e)}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_log(self, message: str, level: str = "info"):
        """Broadcast a log message"""
        await self.broadcast({
            "type": "log",
            "level": level,
            "message": message
        })
    
    async def broadcast_workflow_progress(self, workflow_id: str, step: str, status: str, progress: int):
        """Broadcast workflow progress update"""
        await self.broadcast({
            "type": "workflow_progress",
            "workflow_id": workflow_id,
            "step": step,
            "status": status,
            "progress": progress
        })
    
    async def broadcast_workflow_complete(self, workflow_id: str, results: List[Dict]):
        """Broadcast workflow completion"""
        await self.broadcast({
            "type": "workflow_complete",
            "workflow_id": workflow_id,
            "results": results
        })
    
    async def broadcast_workflow_error(self, workflow_id: str, error: str):
        """Broadcast workflow error"""
        await self.broadcast({
            "type": "workflow_error",
            "workflow_id": workflow_id,
            "error": error
        })
    
    # Enhanced status broadcasting methods
    
    async def broadcast_detailed_progress(self, workflow_id: str, detailed_status: dict):
        """Broadcast detailed workflow progress with sub-steps and metrics"""
        await self.broadcast({
            "type": "detailed_progress",
            "workflow_id": workflow_id,
            "detailed_status": detailed_status
        })
    
    async def broadcast_agent_status(self, workflow_id: str, agent_status: dict):
        """Broadcast CrewAI agent status update"""
        await self.broadcast({
            "type": "agent_status",
            "workflow_id": workflow_id,
            "agent_status": agent_status
        })
    
    async def broadcast_model_performance(self, workflow_id: str, model_metrics: dict):
        """Broadcast model performance metrics"""
        await self.broadcast({
            "type": "model_performance",
            "workflow_id": workflow_id,
            "model_metrics": model_metrics
        })
    
    async def broadcast_processing_stats(self, workflow_id: str, processing_stats: dict):
        """Broadcast processing statistics"""
        await self.broadcast({
            "type": "processing_stats",
            "workflow_id": workflow_id,
            "processing_stats": processing_stats
        })
    
    async def broadcast_resource_usage(self, workflow_id: str, resource_usage: dict):
        """Broadcast system resource usage"""
        await self.broadcast({
            "type": "resource_usage",
            "workflow_id": workflow_id,
            "resource_usage": resource_usage
        })
    
    async def broadcast_substep_progress(self, workflow_id: str, step: str, substep: dict):
        """Broadcast sub-step progress within a main workflow step"""
        await self.broadcast({
            "type": "substep_progress",
            "workflow_id": workflow_id,
            "step": step,
            "substep": substep
        })
    
    async def broadcast_activity_log(self, activity_entry: dict):
        """Broadcast detailed activity log entry"""
        await self.broadcast({
            "type": "activity_log",
            "activity": activity_entry
        })
    
    async def broadcast_crew_agent_activity(self, workflow_id: str, agent_activity: dict):
        """Broadcast CrewAI agent specific activity"""
        await self.broadcast({
            "type": "crew_agent_activity",
            "workflow_id": workflow_id,
            "agent_activity": agent_activity
        })
    
    async def broadcast_quality_metrics(self, workflow_id: str, quality_metrics: dict):
        """Broadcast quality assessment metrics"""
        await self.broadcast({
            "type": "quality_metrics",
            "workflow_id": workflow_id,
            "quality_metrics": quality_metrics
        })
    
    async def broadcast_manager_agent_decision(self, workflow_id: str, decision_data: dict):
        """Broadcast Manager Agent decision-making process"""
        await self.broadcast({
            "type": "manager_agent_decision",
            "workflow_id": workflow_id,
            "decision_data": decision_data
        })
    
    def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)
