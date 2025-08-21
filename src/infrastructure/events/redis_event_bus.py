import asyncio
import json
import logging
from abc import ABC
from typing import Dict, List, Callable, Any, Optional
import redis.asyncio as redis
from ...domain.events.base import DomainEvent
from ...core.config import settings
from ...domain.services.event_dispatcher import IEventBus

logger = logging.getLogger(__name__)


class RedisEventBus(IEventBus):
    """Redis-based event bus for distributed pub/sub pattern"""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.subscribers: Dict[str, List[Callable]] = {}
        self.running = False

    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_keepalive=True,
                socket_keepalive_options={},
            )
            await self.redis_client.ping()
            self.pubsub = self.redis_client.pubsub()
            logger.info("Connected to Redis for event bus")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self):
        """Disconnect from Redis"""
        self.running = False
        if self.pubsub:
            await self.pubsub.close()
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Disconnected from Redis event bus")

    async def is_connected(self) -> bool:
        """Check if Redis client is connected"""
        if self.redis_client:
            try:
                return await self.redis_client.ping() is True
            except redis.ConnectionError:
                return False
        return False

    async def publish(self, event: DomainEvent, channel: str = None):
        """Publish event to Redis channel"""
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")

        channel = channel or f"events.{event.__class__.__name__}"
        event_data = event.to_json()

        try:
            await self.redis_client.publish(channel, event_data)
            logger.debug(f"Published event {event.__class__.__name__} to channel {channel}")
        except Exception as e:
            logger.error(f"Failed to publish event to Redis: {e}")
            raise

    async def subscribe(self, pattern: str, handler: Callable):
        """Subscribe to Redis channel pattern"""
        if pattern not in self.subscribers:
            self.subscribers[pattern] = []
        self.subscribers[pattern].append(handler)

        if self.pubsub:
            await self.pubsub.psubscribe(pattern)
            logger.info(f"Subscribed to pattern {pattern}")

    async def start_listening(self):
        """Start listening for events"""
        if not self.pubsub or not self.subscribers:
            logger.warning("No subscribers registered or pubsub not initialized")
            return

        self.running = True
        logger.info("Starting Redis event listener")

        try:
            async for message in self.pubsub.listen():
                if not self.running:
                    break

                if message['type'] == 'pmessage':
                    await self._handle_message(message)
        except Exception as e:
            logger.error(f"Error in Redis event listener: {e}")
        finally:
            self.running = False

    async def _handle_message(self, message):
        """Handle incoming Redis message"""
        try:
            pattern = message['pattern']
            channel = message['channel']
            data = json.loads(message['data'])

            # Find matching handlers
            handlers = self.subscribers.get(pattern, [])

            if handlers:
                logger.debug(f"Processing {len(handlers)} handlers for channel {channel}")

                # Execute handlers concurrently
                tasks = []
                for handler in handlers:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(data, channel))
                    else:
                        tasks.append(asyncio.create_task(
                            asyncio.get_event_loop().run_in_executor(None, handler, data, channel)
                        ))

                # Wait for all handlers with timeout
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Timeout processing handlers for channel {channel}")

        except Exception as e:
            logger.error(f"Error handling Redis message: {e}")
