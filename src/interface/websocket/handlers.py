from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import logging
import asyncio

logger = logging.getLogger(__name__)


class WebSocketManager:
    """WebSocket connection manager"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.project_subscribers: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket):
        """Accept websocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove websocket connection"""
        self.active_connections.discard(websocket)

        # Remove from project subscriptions
        for project, subscribers in self.project_subscribers.items():
            subscribers.discard(websocket)

        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def subscribe_to_project(self, websocket: WebSocket, project: str):
        """Subscribe websocket to project updates"""
        if project not in self.project_subscribers:
            self.project_subscribers[project] = set()

        self.project_subscribers[project].add(websocket)

        await self.send_personal_message(websocket, {
            "type": "subscription_confirmed",
            "project": project
        })

        logger.info(f"WebSocket subscribed to project: {project}")

    async def send_personal_message(self, websocket: WebSocket, message: dict):
        """Send message to specific websocket"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {str(e)}")
            self.disconnect(websocket)

    async def broadcast_to_project(self, project: str, message: dict):
        """Broadcast message to all subscribers of a project"""
        if project not in self.project_subscribers:
            return

        disconnected = set()

        for websocket in self.project_subscribers[project]:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to project {project}: {str(e)}")
                disconnected.add(websocket)

        # Clean up disconnected websockets
        for websocket in disconnected:
            self.disconnect(websocket)

    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all connected websockets"""
        disconnected = set()

        for websocket in self.active_connections.copy():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to all: {str(e)}")
                disconnected.add(websocket)

        # Clean up disconnected websockets
        for websocket in disconnected:
            self.disconnect(websocket)


# Global WebSocket manager
websocket_manager = WebSocketManager()


async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint"""
    await websocket_manager.connect(websocket)

    try:
        while True:
            # Wait for message from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                await handle_websocket_message(websocket, message)
            except json.JSONDecodeError:
                await websocket_manager.send_personal_message(websocket, {
                    "type": "error",
                    "message": "Invalid JSON format"
                })

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        websocket_manager.disconnect(websocket)


async def handle_websocket_message(websocket: WebSocket, message: dict):
    """Handle incoming WebSocket message"""
    message_type = message.get("type")

    if message_type == "subscribe_project":
        project = message.get("project")
        if project:
            await websocket_manager.subscribe_to_project(websocket, project)
        else:
            await websocket_manager.send_personal_message(websocket, {
                "type": "error",
                "message": "Project name required for subscription"
            })

    elif message_type == "ping":
        await websocket_manager.send_personal_message(websocket, {
            "type": "pong",
            "timestamp": message.get("timestamp")
        })

    else:
        await websocket_manager.send_personal_message(websocket, {
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        })


# Event handlers for broadcasting
async def broadcast_commit_received(event_data: dict):
    """Broadcast commit received event"""
    project = event_data.get("project")
    if project:
        await websocket_manager.broadcast_to_project(project, {
            "type": "commit_received",
            "data": event_data
        })


async def broadcast_commit_analyzed(event_data: dict):
    """Broadcast commit analysis completed event"""
    # In a real implementation, you'd get the project from the commit
    await websocket_manager.broadcast_to_all({
        "type": "commit_analyzed",
        "data": event_data
    })
