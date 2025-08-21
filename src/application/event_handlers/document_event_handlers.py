import logging
from src.application.events.document_events import DocumentIndexedEvent
from src.domain.events.event_handler_registry import local_handler
from src.interface.api.dependencies import get_cache_service

logger = logging.getLogger(__name__)


@local_handler(DocumentIndexedEvent, priority=1)
async def invalidate_search_cache_handler(event: DocumentIndexedEvent):
    """Invalidate search cache when new documents are indexed"""
    try:
        cache_service = await get_cache_service()

        # Invalidate search cache patterns
        patterns = ["search:docs:*"]
        if event.project:
            patterns.append(f"search:project:{event.project}:*")

        for pattern in patterns:
            count = await cache_service.invalidate_pattern(pattern)
            logger.info(f"Invalidated {count} cache entries for pattern: {pattern}")

    except Exception as e:
        logger.error(f"Failed to invalidate cache for DocumentIndexedEvent: {str(e)}")