import asyncio
import logging
from typing import Dict, List, Callable, Any
from ...domain.events.base import DomainEvent
from ...domain.services.event_dispatcher import IEventDispatcher

logger = logging.getLogger(__name__)


class EventDispatcher(IEventDispatcher):
    """Event dispatcher with registry integration and error handling"""

    def __init__(self, use_registry: bool = True):
        self.manual_handlers: Dict[str, List[Callable]] = {}
        self.use_registry = use_registry
        self.circuit_breaker = EventCircuitBreaker()

    def register_handler(self, event_type: str, handler: Callable):
        """Register handler manually (alternative to decorators)"""
        if event_type not in self.manual_handlers:
            self.manual_handlers[event_type] = []
        self.manual_handlers[event_type].append(handler)
        logger.info(f"Manually registered handler {handler.__name__} for {event_type}")

    async def dispatch(self, event: DomainEvent):
        """Dispatch event to all registered handlers with error isolation"""
        event_type = event.__class__.__name__
        logger.info(f"Dispatching event: {event_type} with ID {event.event_id}")

        # Collect handlers from multiple sources
        handlers = []

        # From registry (decorator-based)
        if self.use_registry:
            from ...domain.events.event_handler_registry import EventHandlerRegistry
            logger.info(f"Registered handlers for {event_type}: {EventHandlerRegistry.get_handlers(event_type)}")
            handlers.extend(EventHandlerRegistry.get_handlers(event_type))

        # From manual registration
        if event_type in self.manual_handlers:
            handlers.extend(self.manual_handlers[event_type])

        if not handlers:
            logger.debug(f"No handlers found for event type: {event_type}")
            return

        logger.info(f"Dispatching {event_type} to {len(handlers)} handlers")

        # Execute handlers with error isolation and circuit breaker
        results = await self._execute_handlers_safely(handlers, event, event_type)

        # Log results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful

        if failed > 0:
            logger.warning(f"Event {event_type}: {successful} succeeded, {failed} failed")
        else:
            logger.debug(f"Event {event_type}: all {successful} handlers succeeded")

    async def _execute_handlers_safely(self, handlers: List[Callable], event: DomainEvent, event_type: str) -> List[
        Any]:
        """Execute handlers with safety measures"""
        tasks = []

        for i, handler in enumerate(handlers):
            # Check circuit breaker
            if self.circuit_breaker.is_open(handler.__name__):
                logger.warning(f"Circuit breaker open for handler {handler.__name__}, skipping")
                continue

            # Wrap handler execution with timeout and error handling
            task = self._execute_single_handler(handler, event, event_type, i)
            tasks.append(task)

        if not tasks:
            return []

        # Execute all handlers concurrently with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0  # Global timeout for all handlers
            )
            return results
        except asyncio.TimeoutError:
            logger.error(f"Global timeout exceeded for event {event_type}")
            return [TimeoutError("Global handler timeout")]

    async def _execute_single_handler(self, handler: Callable, event: DomainEvent, event_type: str, handler_index: int):
        """Execute single handler with individual timeout and error tracking"""
        handler_name = handler.__name__

        try:
            # Individual handler timeout
            if asyncio.iscoroutinefunction(handler):
                result = await asyncio.wait_for(handler(event), timeout=10.0)
            else:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, handler, event),
                    timeout=10.0
                )

            # Record success
            self.circuit_breaker.record_success(handler_name)
            logger.debug(f"Handler {handler_name} completed successfully for {event_type}")
            return result

        except asyncio.TimeoutError:
            error = f"Handler {handler_name} timed out for {event_type}"
            logger.error(error)
            self.circuit_breaker.record_failure(handler_name)
            return TimeoutError(error)

        except Exception as e:
            error_msg = f"Handler {handler_name} failed for {event_type}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.circuit_breaker.record_failure(handler_name)
            return e


class EventCircuitBreaker:
    """Circuit breaker for event handlers to prevent cascade failures"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_counts: Dict[str, int] = {}
        self.last_failure_time: Dict[str, float] = {}
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

    def is_open(self, handler_name: str) -> bool:
        """Check if circuit breaker is open for a handler"""
        import time

        if handler_name not in self.failure_counts:
            return False

        failure_count = self.failure_counts[handler_name]
        last_failure = self.last_failure_time.get(handler_name, 0)

        # If under threshold, circuit is closed
        if failure_count < self.failure_threshold:
            return False

        # If enough time has passed, try to recover
        if time.time() - last_failure > self.recovery_timeout:
            self.failure_counts[handler_name] = 0
            return False

        return True

    def record_failure(self, handler_name: str):
        """Record a failure for a handler"""
        import time

        self.failure_counts[handler_name] = self.failure_counts.get(handler_name, 0) + 1
        self.last_failure_time[handler_name] = time.time()

        if self.failure_counts[handler_name] >= self.failure_threshold:
            logger.warning(f"Circuit breaker opened for handler {handler_name}")

    def record_success(self, handler_name: str):
        """Record a success for a handler"""
        if handler_name in self.failure_counts:
            self.failure_counts[handler_name] = 0
