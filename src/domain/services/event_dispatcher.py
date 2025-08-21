from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Any
from ...domain.events.base import DomainEvent


class IEventDispatcher(ABC):
    """Enhanced event dispatcher interface"""

    @abstractmethod
    async def dispatch(self, event: DomainEvent) -> None:
        """
        Dispatch an event to all registered handlers.

        This is the MAIN method application layer should use.
        It automatically handles:
        - Local event processing
        - Distributed event publishing (if event is marked for distribution)
        - Error handling and isolation
        """
        pass

    @abstractmethod
    def register_handler(self, event_type: str, handler: Callable[[DomainEvent], Any]) -> None:
        """Register a handler manually (alternative to decorators)"""
        pass

    # Optional: Advanced methods for specific use cases
    async def dispatch_local_only(self, event: DomainEvent) -> None:
        """
        Dispatch event only locally (skip distribution).
        USE CASE: When you know event shouldn't be distributed
        """
        pass

    async def publish_distributed(self, event: DomainEvent) -> None:
        """
        Publish event only to distributed layer (skip local).
        USE CASE: Forwarding events from external sources
        """
        pass


class IEventBus(ABC):
    """Interface for event bus implementations"""

    async def connect(self):
        """Connect to the event bus"""
        pass

    async def disconnect(self):
        """Disconnect from the event bus"""
        pass

    async def is_connected(self) -> bool:
        """Check if the event bus is connected"""
        pass

    async def publish(self, event: DomainEvent, channel: str = None):
        """Publish an event to the bus"""
        pass

    async def subscribe(self, pattern: str, handler: Callable):
        """Subscribe to events matching a pattern"""
        pass

    async def start_listening(self):
        """Start listening for events"""
        pass