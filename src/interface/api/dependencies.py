from typing import AsyncGenerator
from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

# Import domain interfaces
from ...domain.repositories.commit_repository import ICommitRepository
from ...domain.services.ai_analyzer import IAIAnalyzer
from ...domain.services.cache_service import ICacheService
from ...domain.services.message_queue import IMessageQueue
from ...domain.services.event_dispatcher import IEventDispatcher

# Import infrastructure implementations
from ...infrastructure.persistence.database import get_session
from ...infrastructure.persistence.repositories.sqlite_commit_repository import SqliteCommitRepository
from ...infrastructure.ai.services.ollama_service import OllamaService
from ...infrastructure.events.event_bus import RedisEventBus
from ...infrastructure.cache.memory_cache import MemoryCacheService
from ...infrastructure.cache.redis_cache import RedisCacheService
from ...infrastructure.messaging.celery_config import celery_app, CeleryMessageQueue

import os
import logging

logger = logging.getLogger(__name__)

# Global singletons
_redis_client = None
_cache_service = None
_event_dispatcher = None
_message_queue = None
_ai_service = None


async def get_redis_client() -> Redis:
    """Get Redis client singleton"""
    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            _redis_client = Redis.from_url(redis_url, decode_responses=False)
            # Test connection
            await _redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {str(e)}, falling back to memory cache")
            _redis_client = None
    return _redis_client


async def get_cache_service() -> ICacheService:
    """Get cache service singleton"""
    global _cache_service
    if _cache_service is None:
        redis_client = await get_redis_client()
        if redis_client:
            _cache_service = RedisCacheService(redis_client)
        else:
            _cache_service = MemoryCacheService()
        logger.info(f"Cache service initialized: {type(_cache_service).__name__}")
    return _cache_service


async def get_message_queue() -> IMessageQueue:
    """Get message queue singleton"""
    global _message_queue
    if _message_queue is None:
        _message_queue = CeleryMessageQueue(celery_app)
        logger.info("Message queue initialized")
    return _message_queue


async def get_event_dispatcher() -> IEventDispatcher:
    """Get event dispatcher singleton"""
    global _event_dispatcher
    if _event_dispatcher is None:
        redis_client = await get_redis_client()
        if redis_client:
            _event_dispatcher = RedisEventBus(redis_client)
            await _event_dispatcher.start()
        else:
            from ...application.services.event_dispatcher import EventDispatcher
            _event_dispatcher = EventDispatcher()
        logger.info(f"Event dispatcher initialized: {type(_event_dispatcher).__name__}")
    return _event_dispatcher


async def get_ai_service() -> IAIAnalyzer:
    """Get AI service singleton"""
    global _ai_service
    if _ai_service is None:
        _ai_service = OllamaService()
        # Test AI service health
        try:
            health = await _ai_service.health_check()
            if not health:
                logger.warning("AI service health check failed")
        except Exception as e:
            logger.warning(f"AI service health check error: {str(e)}")
        logger.info("AI service initialized")
    return _ai_service


async def get_commit_repository(
        session: AsyncSession = Depends(get_session)
) -> ICommitRepository:
    """Get commit repository with injected session"""
    return SqliteCommitRepository(session)


# Use case dependencies
async def get_process_commit_use_case(
        repo: ICommitRepository = Depends(get_commit_repository),
        dispatcher: IEventDispatcher = Depends(get_event_dispatcher),
        message_queue: IMessageQueue = Depends(get_message_queue),
        cache: ICacheService = Depends(get_cache_service)
):
    """Get process commit use case with all dependencies"""
    from ...application.use_cases.process_commit import ProcessCommitUseCase
    return ProcessCommitUseCase(repo, dispatcher, message_queue, cache)


async def get_search_commits_use_case(
        repo: ICommitRepository = Depends(get_commit_repository),
        cache: ICacheService = Depends(get_cache_service)
):
    """Get search commits use case"""
    from ...application.use_cases.search_commits import SearchCommitsUseCase
    return SearchCommitsUseCase(repo, cache)


async def get_chat_use_case(
        repo: ICommitRepository = Depends(get_commit_repository),
        ai_service: IAIAnalyzer = Depends(get_ai_service),
        cache: ICacheService = Depends(get_cache_service)
):
    """Get chat use case"""
    from ...application.use_cases.chat_with_ai import ChatWithAIUseCase
    return ChatWithAIUseCase(repo, ai_service, cache)


def verify_webhook_signature(x_hub_signature: str = None) -> bool:
    """Verify webhook signature (simplified for demo)"""
    webhook_secret = os.getenv("WEBHOOK_SECRET")
    if not webhook_secret:
        return True  # Skip verification in development

    # In production, implement proper HMAC verification
    # For demo purposes, just check if signature exists
    return x_hub_signature is not None
