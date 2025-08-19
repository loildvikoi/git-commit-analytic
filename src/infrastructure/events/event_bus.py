import asyncio
import json
import logging
from typing import Callable, Dict, List
from redis.asyncio import Redis
from ...domain.events.base import DomainEvent
from ...application.services.event_dispatcher import IEventDispatcher

logger = logging.getLogger(__name__)


class RedisEventBus(IEventDispatcher):
    """Redis-based event bus for distributed event handling"""

    def __init__(self, redis_client: Redis, channel_prefix: str = "events"):
        self.redis = redis_client
        self.channel_prefix = channel_prefix
        self.handlers: Dict[str, List[Callable]] = {}
        self.subscriber_task = None
        self._running = False

    async def start(self):
        """Start event bus subscriber"""
        if not self._running:
            self._running = True
            self.subscriber_task = asyncio.create_task(self._subscriber_loop())
            logger.info(f"Redis event bus started, subscribing to channels with prefix: {self.channel_prefix}")

    async def stop(self):
        """Stop event bus subscriber"""
        self._running = False
        if self.subscriber_task:
            self.subscriber_task.cancel()
            try:
                await self.subscriber_task
            except asyncio.CancelledError:
                pass
        logger.info("Redis event bus stopped")

    def register_handler(self, event_type: str, handler: Callable):
        """Register event handler"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        logger.info(f"Registered Redis handler for event type: {event_type}")

    async def dispatch(self, event: DomainEvent):
        """Dispatch event to Redis"""
        try:
            event_type = event.__class__.__name__
            channel = f"{self.channel_prefix}:{event_type}"

            message = event.to_json()
            await self.redis.publish(channel, message)

            logger.debug(f"Published event {event_type} to Redis channel: {channel}")

        except Exception as e:
            logger.error(f"Error publishing event to Redis: {str(e)}")

    async def _subscriber_loop(self):
        """Main subscriber loop"""
        try:
            pubsub = self.redis.pubsub()

            # Subscribe to all event channels
            pattern = f"{self.channel_prefix}:*"
            await pubsub.psubscribe(pattern)

            logger.info(f"Subscribed to Redis pattern: {pattern}")

            async for message in pubsub.listen():
                if not self._running:
                    break

                if message['type'] == 'pmessage':
                    await self._handle_message(message)

        except asyncio.CancelledError:
            logger.info("Redis subscriber loop cancelled")
        except Exception as e:
            logger.error(f"Error in Redis subscriber loop: {str(e)}")
        finally:
            try:
                await pubsub.unsubscribe()
                await pubsub.close()
            except Exception:
                pass

    async def _handle_message(self, message):
        """Handle incoming Redis message"""
        try:
            channel = message['channel'].decode('utf-8')
            event_type = channel.split(':')[-1]

            if event_type not in self.handlers:
                return

            # Parse event data
            event_data = json.loads(message['data'].decode('utf-8'))

            # Execute handlers
            tasks = []
            for handler in self.handlers[event_type]:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(event_data))
                else:
                    loop = asyncio.get_event_loop()
                    tasks.append(loop.run_in_executor(None, handler, event_data))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Log handler exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Redis handler {i} failed for {event_type}: {str(result)}")

        except Exception as e:
            logger.error(f"Error handling Redis message: {str(e)}")
