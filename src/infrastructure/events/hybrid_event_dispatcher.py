import asyncio
import logging
import socket
from typing import Dict, List, Callable, Any
from ...domain.events.base import DomainEvent
from ...domain.services.event_dispatcher import IEventDispatcher
from .redis_event_bus import RedisEventBus
from .local_event_dispatcher import EventDispatcher

logger = logging.getLogger(__name__)


class HybridEventDispatcher(IEventDispatcher):
    """
    Hybrid dispatcher that handles both local and distributed events

    HOW IT WORKS:
    1. @local_handler → only local processing
    2. @distributed_handler → local processing + distribute to other nodes
    3. @event_subscriber → listen to Redis pub/sub from other nodes
    """

    def __init__(self, redis_event_bus: RedisEventBus = None):
        self.local_dispatcher = EventDispatcher()
        self.redis_bus = redis_event_bus
        self.node_id = self._generate_node_id()

    def _generate_node_id(self) -> str:
        """Generate unique node identifier"""
        import uuid
        hostname = socket.gethostname()
        return f"{hostname}-{str(uuid.uuid4())[:8]}"

    async def dispatch(self, event: DomainEvent):
        """
        MAIN METHOD: Application layer uses ONLY this method

        Automatically handles:
        - Execute all local handlers
        - Distribute event if needed (based on @distributed_handler decoration)
        """
        event_type = event.__class__.__name__
        logger.debug(f"Dispatching event: {event_type}")

        # 1. Always process locally first
        await self.local_dispatcher.dispatch(event)

        # 2. Check if event should be distributed
        if self._should_distribute(event):
            await self._dispatch_distributed(event)

    async def dispatch_local_only(self, event: DomainEvent):
        """Force local-only processing"""
        await self.local_dispatcher.dispatch(event)

    async def publish_distributed(self, event: DomainEvent):
        """Force distributed-only publishing"""
        await self._dispatch_distributed(event)

    def _should_distribute(self, event: DomainEvent) -> bool:
        """
        Determine if event should be distributed

        An event is distributed if:
        1. It has @distributed_handler decorations, OR
        2. It's marked in EventHandlerRegistry as distributed
        """
        from ...domain.events.event_handler_registry import EventHandlerRegistry
        event_type = event.__class__.__name__
        return EventHandlerRegistry.should_distribute(event_type)

    async def _dispatch_distributed(self, event: DomainEvent):
        """Distribute event to other nodes via Redis"""
        if not self.redis_bus:
            logger.warning("Redis bus not available for distributed dispatch")
            return

        try:
            # Add source node info to prevent echo
            if hasattr(event, 'source_node') and not event.source_node:
                event.source_node = self.node_id

            await self.redis_bus.publish(event)
            logger.debug(f"Distributed event {event.__class__.__name__} via Redis")

        except Exception as e:
            logger.error(f"Failed to distribute event via Redis: {e}")

    def register_handler(self, event_type: str, handler: Callable):
        """Register local handler manually"""
        self.local_dispatcher.register_handler(event_type, handler)

