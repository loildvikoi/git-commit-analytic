import logging
import socket
from ..events.cache_events import CacheInvalidationEvent, CacheWarmupEvent
from src.domain.events.event_handler_registry import event_subscriber
from src.interface.api.dependencies import get_cache_service

logger = logging.getLogger(__name__)


@event_subscriber("events.CacheInvalidationEvent")
async def handle_cache_invalidation_distributed(event_data: dict, channel: str):
    """Handle distributed cache invalidation"""
    try:
        # Skip if from same node
        current_node = socket.gethostname()
        source_node = event_data.get('data', {}).get('source_node', '')

        if current_node in source_node:
            logger.debug(f"Skipping cache invalidation from same node: {source_node}")
            return

        cache_service = await get_cache_service()
        patterns = event_data.get('data', {}).get('patterns', [])
        keys = event_data.get('data', {}).get('keys', [])
        reason = event_data.get('data', {}).get('reason', 'unknown')

        # Invalidate patterns
        for pattern in patterns:
            await cache_service.invalidate_pattern(pattern)
            logger.info(f"Invalidated cache pattern: {pattern} (reason: {reason})")

        # Invalidate specific keys
        for key in keys:
            await cache_service.delete(key)
            logger.info(f"Invalidated cache key: {key} (reason: {reason})")

    except Exception as e:
        logger.error(f"Failed to handle distributed cache invalidation: {e}")


@event_subscriber("events.CacheWarmupEvent")
async def handle_cache_warmup_distributed(event_data: dict, channel: str):
    """Handle distributed cache warmup"""
    try:
        cache_service = await get_cache_service()
        cache_type = event_data.get('data', {}).get('cache_type')
        data = event_data.get('data', {}).get('data', {})
        ttl = event_data.get('data', {}).get('ttl', 3600)

        # Warmup cache based on type
        if cache_type == "commit_search":
            for key, value in data.items():
                await cache_service.set(key, value, ttl=ttl)
                logger.info(f"Warmed up cache key: {key}")

    except Exception as e:
        logger.error(f"Failed to handle distributed cache warmup: {e}")

