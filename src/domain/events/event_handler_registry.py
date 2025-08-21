import logging
from typing import Dict, List, Callable, Any, Type, Set
from functools import wraps
from enum import Enum
from .base import DomainEvent
from ...interface.api.dependencies import get_redis_event_bus

logger = logging.getLogger(__name__)


class HandlerType(Enum):
    """Types of event handlers"""
    LOCAL = "local"
    DISTRIBUTED = "distributed"
    SUBSCRIBER = "subscriber"


class EventHandlerRegistry:
    """Enhanced global registry for event handlers"""

    _handlers: Dict[str, List[Callable]] = {}
    _priority_handlers: Dict[str, Dict[int, List[Callable]]] = {}
    _handler_metadata: Dict[str, Dict[str, Any]] = {}  # Store handler metadata
    _redis_subscribers: Dict[str, List[Callable]] = {}  # Redis subscribers
    _distributed_events: Set[str] = set()  # Events that should be distributed

    @classmethod
    def register(cls, event_type: Type[DomainEvent], priority: int = 5,
                 handler_type: HandlerType = HandlerType.LOCAL,
                 distribute: bool = False):
        """Enhanced decorator to register event handlers"""

        def decorator(func: Callable):
            event_name = event_type.__name__

            # Store by priority for ordered execution
            if event_name not in cls._priority_handlers:
                cls._priority_handlers[event_name] = {}
            if priority not in cls._priority_handlers[event_name]:
                cls._priority_handlers[event_name][priority] = []

            cls._priority_handlers[event_name][priority].append(func)

            # Store in simple handlers for compatibility
            if event_name not in cls._handlers:
                cls._handlers[event_name] = []
            cls._handlers[event_name].append(func)

            # Store metadata
            cls._handler_metadata[f"{event_name}.{func.__name__}"] = {
                'event_type': event_name,
                'handler_name': func.__name__,
                'priority': priority,
                'handler_type': handler_type.value,
                'distribute': distribute,
                'module': func.__module__
            }

            # Mark for distribution if needed
            if distribute:
                cls._distributed_events.add(event_name)

            logger.info(f"Registered {handler_type.value} handler {func.__name__} "
                        f"for {event_name} with priority {priority}")

            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            return wrapper

        return decorator

    @classmethod
    def register_subscriber(cls, channel_pattern: str, priority: int = 5):
        """Register Redis subscriber"""

        def decorator(func: Callable):
            if channel_pattern not in cls._redis_subscribers:
                cls._redis_subscribers[channel_pattern] = []
            cls._redis_subscribers[channel_pattern].append((func, priority))

            # Store metadata
            cls._handler_metadata[f"{channel_pattern}.{func.__name__}"] = {
                'channel_pattern': channel_pattern,
                'handler_name': func.__name__,
                'priority': priority,
                'handler_type': HandlerType.SUBSCRIBER.value,
                'module': func.__module__
            }

            logger.info(f"Registered subscriber {func.__name__} for pattern {channel_pattern}")
            return func

        return decorator

    @classmethod
    def get_handlers(cls, event_type: str) -> List[Callable]:
        """Get handlers for event type ordered by priority"""
        if event_type not in cls._priority_handlers:
            return []

        handlers = []
        for priority in sorted(cls._priority_handlers[event_type].keys()):
            handlers.extend(cls._priority_handlers[event_type][priority])

        return handlers

    @classmethod
    def get_subscribers(cls, channel_pattern: str = None) -> Dict[str, List[Callable]]:
        """Get Redis subscribers"""
        if channel_pattern:
            return {channel_pattern: [h[0] for h in cls._redis_subscribers.get(channel_pattern, [])]}

        result = {}
        for pattern, handlers in cls._redis_subscribers.items():
            result[pattern] = [h[0] for h in handlers]
        return result

    @classmethod
    def should_distribute(cls, event_type: str) -> bool:
        """Check if event should be distributed"""
        return event_type in cls._distributed_events

    @classmethod
    def get_handler_stats(cls) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            'total_handlers': sum(len(handlers) for handlers in cls._handlers.values()),
            'total_events': len(cls._handlers),
            'total_subscribers': sum(len(handlers) for handlers in cls._redis_subscribers.values()),
            'distributed_events': len(cls._distributed_events),
            'handlers_by_event': {k: len(v) for k, v in cls._handlers.items()},
            'subscribers_by_pattern': {k: len(v) for k, v in cls._redis_subscribers.items()}
        }


# Convenience decorators
def local_handler(event_type: Type[DomainEvent], priority: int = 5):
    """Local event handler decorator"""
    return EventHandlerRegistry.register(event_type, priority, HandlerType.LOCAL)


def distributed_handler(event_type: Type[DomainEvent], priority: int = 5):
    """Distributed event handler decorator - executes locally AND distributes"""
    return EventHandlerRegistry.register(event_type, priority, HandlerType.DISTRIBUTED, distribute=True)


def event_subscriber(channel_pattern: str, priority: int = 5):
    """Redis event subscriber decorator"""
    return EventHandlerRegistry.register_subscriber(channel_pattern, priority)


async def init_event_handlers():
    """Initialize all event handlers"""
    try:
        # Import business domain handlers
        from src.application.event_handlers import commit_event_handlers
        from src.application.event_handlers import document_event_handlers

        # Import infrastructure handlers
        from src.infrastructure.messaging.event_handlers import cache_event_handlers

        logger.info("Imported all event handler modules")

        # Log registry stats
        stats = EventHandlerRegistry.get_handler_stats()
        logger.info(f"Event registry stats: {stats}")

    except Exception as e:
        logger.error(f"Failed to initialize event handlers: {str(e)}")


async def init_redis_subscribers():
    """Initialize Redis event subscribers"""
    try:
        redis_bus = await get_redis_event_bus()
        if not redis_bus:
            logger.warning("Redis event bus not available, skipping subscriber initialization")
            return

        # Import modules with @event_subscriber decorators
        from ...infrastructure.messaging.event_handlers import cache_event_handlers

        # Register subscribers from EventHandlerRegistry
        subscribers = EventHandlerRegistry.get_subscribers()

        for pattern, handlers in subscribers.items():
            for handler in handlers:
                await redis_bus.subscribe(pattern, handler)
                logger.info(f"Registered Redis subscriber: {handler.__name__} for {pattern}")

        # Start background listener
        import asyncio
        asyncio.create_task(redis_bus.start_listening())

        logger.info("Redis event subscribers initialized")

    except Exception as e:
        logger.error(f"Failed to initialize Redis subscribers: {e}")
