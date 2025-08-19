from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Any
from ...domain.events.base import DomainEvent


class IEventDispatcher(ABC):
    @abstractmethod
    def dispatch(self, event: DomainEvent) -> None:
        """Dispatch an event to all registered handlers."""
        pass

    @abstractmethod
    def register_handler(self, event_type: type, handler: Callable[[DomainEvent], Any]) -> None:
        """Register a handler for a specific event type."""
        pass
