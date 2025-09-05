# src/application/event_handlers/agent_event_handlers.py
"""
Event handlers for agent processing
"""
import logging

from ...domain.entities.agent import AgentRole
from ...domain.events.event_handler_registry import local_handler
from ..events.agent_events import AgentAnalysisRequestedEvent
from ...domain.services.queue_service import IQueueService
from ...interface.api.dependencies import (
    get_agent_service,
    get_commit_repository,
    get_document_repository,
    get_search_service, get_queue_service
)

logger = logging.getLogger(__name__)


@local_handler(AgentAnalysisRequestedEvent, priority=1)
async def process_agent_analysis(event: AgentAnalysisRequestedEvent):
    """Process commit with agents when requested"""
    try:
        queue_service: IQueueService = await get_queue_service()

        task_id = await queue_service.enqueue(
            "agent_analyze_commit",
            {"commit_id": event.commit_id, "document_id": event.document_id}
        )
        logger.info(f"Queued agent analysis for commit {event.commit_id}, task_id: {task_id}")

    except Exception as ex:
        logger.error(f"Failed to queue agent analysis for {event.commit_id}: {str(ex)}", exc_info=True)
