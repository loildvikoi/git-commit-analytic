import logging
import socket

from src.application.events.commit_events import CommitReceivedEvent, CommitAnalysisCompletedEvent
from src.infrastructure.messaging.events.cache_events import CacheInvalidationEvent
from src.domain.events.event_handler_registry import local_handler, distributed_handler
from src.domain.services.queue_service import IQueueService
from src.infrastructure.persistence.database import get_session
from src.interface.api.dependencies import get_queue_service, get_cache_service, get_event_dispatcher, \
    get_sync_commits_use_case, get_commit_repository

logger = logging.getLogger(__name__)


@local_handler(CommitReceivedEvent, priority=1)
async def analysis_queue_handler(event: CommitReceivedEvent):
    """High priority local handler - queue analysis task"""
    try:
        queue_service: IQueueService = await get_queue_service()

        task_id = await queue_service.enqueue(
            "analyze_commit",
            {"commit_id": event.commit_id}
        )
        logger.info(f"Queued analysis for commit {event.commit_hash}, task_id: {task_id}")

    except Exception as ex:
        logger.error(f"Failed to queue analysis for {event.commit_hash}: {str(ex)}", exc_info=True)


@distributed_handler(CommitReceivedEvent, priority=3)
async def cache_invalidation_handler(event: CommitReceivedEvent):
    """Distributed handler - trigger cache invalidation across all nodes"""
    try:
        # This will execute locally AND distribute to other nodes
        cache_event = CacheInvalidationEvent(
            patterns=[
                f"commits:project:{event.project}:*",
                f"commits:author:{event.author}:*",
                "commits:recent:*"
            ],
            source_node=socket.gethostname(),
            reason=f"New commit {event.commit_hash}"
        )

        # Dispatch cache invalidation event (will be distributed automatically)
        dispatcher = await get_event_dispatcher()
        await dispatcher.dispatch(cache_event)

        logger.info(f"Triggered distributed cache invalidation for {event.commit_hash}")

    except Exception as ex:
        logger.error(f"Failed to trigger cache invalidation for {event.commit_hash}: {str(ex)}")


@local_handler(CommitAnalysisCompletedEvent, priority=1)
async def analysis_completed_handler(event: CommitAnalysisCompletedEvent):
    """Local handler - process completed analysis"""
    try:
        logger.info(f"Analysis completed for commit {event.commit_id}, summary: {event.summary}")

        async with get_session() as session:
            commit_repository = await get_commit_repository(session)

            # Fetch the commit entity
            saved_commit = await commit_repository.find_by_id(event.commit_id)
            if not saved_commit:
                logger.warning(f"Commit {event.commit_id} not found for analysis completion")
                return

            # Sync commit to documents
            sync_commit_documents_use_case = await get_sync_commits_use_case()
            if sync_commit_documents_use_case:
                await sync_commit_documents_use_case.commit2document(saved_commit, skip_existing=True)

    except Exception as ex:
        logger.error(f"Failed to handle analysis completion for {event.commit_id}: {str(ex)}")