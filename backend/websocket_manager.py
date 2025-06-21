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
    
    def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)
