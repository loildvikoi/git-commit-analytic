from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Any
import asyncio
import logging

from src.domain.events.base import DomainEvent
from src.domain.services.event_dispatcher import IEventDispatcher

logger = logging.getLogger(__name__)


class EventDispatcher(IEventDispatcher):
    """Implementation of event dispatcher"""

    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = {}

    def register_handler(self, event_type: str, handler: Callable):
        """Register a handler for a specific event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        logger.debug(f"Handler registered for event type: {event_type}")

    async def dispatch(self, event: DomainEvent):
        """Dispatch an event to all registered handlers."""
        event_type = event.__class__.__name__

        if event_type not in self.handlers:
            logger.warning(f"No handlers registered for event type: {event_type}")

        logger.debug(f"Dispatching event: {event_type}")

        tasks = []
        for handler in self.handlers[event_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(event))
                else:
                    # Run sync handler in thread pool
                    loop = asyncio.get_event_loop()
                    tasks.append(loop.run_in_executor(None, handler, event))
            except Exception as e:
                logger.error(f"Error preparing handler for {event_type}: {str(e)}")

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log any handler exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Handler {i} failed for {event_type}: {str(result)}")
