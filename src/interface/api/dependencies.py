from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis
import os
import logging

# Import domain interfaces
from ...domain.repositories.commit_repository import ICommitRepository
from ...domain.repositories.document_repository import IDocumentRepository
from ...domain.repositories.vector_repository import IVectorRepository
from ...domain.services.embedding_service import IEmbeddingService
from ...domain.services.search_service import ISearchService
from ...domain.services.rag_service import IRAGService

from ...domain.services.ai_analyzer import IAIAnalyzer
from ...domain.services.cache_service import ICacheService
from ...domain.services.queue_service import IQueueService
from ...domain.services.event_dispatcher import IEventDispatcher

# Import infrastructure implementations
from ...infrastructure.persistence.database import get_session
from ...infrastructure.persistence.repositories.sqlite_commit_repository import SqliteCommitRepository
from ...infrastructure.ai.services.ollama_service import OllamaService
from ...infrastructure.cache.memory_cache import MemoryCacheService
from ...infrastructure.cache.redis_cache import RedisCacheService
from ...infrastructure.queue.celery_queue import celery_app, CeleryQueue
from ...infrastructure.events.redis_event_bus import RedisEventBus
from ...infrastructure.events.hybrid_event_dispatcher import HybridEventDispatcher

logger = logging.getLogger(__name__)

# Global singletons
_redis_client = None
_cache_service = None
_event_dispatcher = None
_queue_service = None
_ai_service = None
_redis_event_bus = None


async def get_redis_client() -> Redis:
    """Get Redis client singleton"""
    global _redis_client
    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            _redis_client = Redis.from_url(redis_url, decode_responses=False)
            await _redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {str(e)}")
            _redis_client = None
    return _redis_client


async def get_redis_event_bus() -> RedisEventBus:
    """Get Redis event bus singleton"""
    global _redis_event_bus
    if _redis_event_bus is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _redis_event_bus = RedisEventBus(redis_url)
        try:
            await _redis_event_bus.connect()
            logger.info("Redis event bus connected")
        except Exception as e:
            logger.error(f"Redis event bus connection failed: {e}")
            _redis_event_bus = None
    return _redis_event_bus


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


async def get_queue_service() -> IQueueService:
    """Get queue service singleton"""
    global _queue_service
    if _queue_service is None:
        _queue_service = CeleryQueue(celery_app)
        logger.info("Queue service initialized")
    return _queue_service


async def get_event_dispatcher() -> IEventDispatcher:
    """Get hybrid event dispatcher singleton"""
    global _event_dispatcher
    if _event_dispatcher is None:
        redis_bus = await get_redis_event_bus()
        _event_dispatcher = HybridEventDispatcher(redis_bus)
        logger.info("Hybrid event dispatcher initialized")
    return _event_dispatcher


async def get_ai_service() -> IAIAnalyzer:
    """Get AI service singleton"""
    global _ai_service
    if _ai_service is None:
        _ai_service = OllamaService()
        try:
            health = await _ai_service.health_check()
            if not health:
                logger.warning("AI service health check failed")
        except Exception as e:
            logger.warning(f"AI service health check error: {str(e)}")
        logger.info("AI service initialized")
    return _ai_service


# Database and repository dependencies (unchanged)
async def get_db_session() -> AsyncSession:
    """FastAPI dependency to get database session"""
    async with get_session() as session:
        yield session


async def get_commit_repository(
        session: AsyncSession = Depends(get_db_session)
) -> ICommitRepository:
    """Get commit repository with injected session"""
    return SqliteCommitRepository(session)


# Use case dependencies
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
        return True
    return x_hub_signature is not None


async def get_document_repository(
    session: AsyncSession = Depends(get_db_session)
) -> IDocumentRepository:
    """Get document repository"""
    # Will be implemented in infrastructure
    from ...infrastructure.persistence.repositories.sqlite_document_repository import SqliteDocumentRepository
    return SqliteDocumentRepository(session)


async def get_vector_repository() -> IVectorRepository:
    """Get vector repository singleton"""
    # Will be implemented in infrastructure
    from ...infrastructure.rag.chroma_vector_repository import ChromaVectorRepository
    return ChromaVectorRepository()


async def get_embedding_service() -> IEmbeddingService:
    """Get embedding service singleton"""
    # Will be implemented in infrastructure
    from ...infrastructure.rag.sentence_transformer_service import SentenceTransformerEmbeddingService
    return SentenceTransformerEmbeddingService()


async def get_search_service(
    document_repo: IDocumentRepository = Depends(get_document_repository),
    vector_repo: IVectorRepository = Depends(get_vector_repository),
    embedding_service: IEmbeddingService = Depends(get_embedding_service)
) -> ISearchService:
    """Get search service"""
    # Will be implemented in infrastructure
    from ...infrastructure.rag.hybrid_search_service import HybridSearchService
    return HybridSearchService(document_repo, vector_repo, embedding_service)


async def get_rag_service(
    search_service: ISearchService = Depends(get_search_service),
    ai_service: IAIAnalyzer = Depends(get_ai_service)
) -> IRAGService:
    """Get RAG service"""
    # Will be implemented in infrastructure
    from ...infrastructure.rag.ollama_rag_service import OllamaRAGService
    return OllamaRAGService(search_service, ai_service)


# Use case dependencies for Phase 2
async def get_sync_documents_use_case(
    commit_repo: ICommitRepository = Depends(get_commit_repository),
    document_repo: IDocumentRepository = Depends(get_document_repository),
    embedding_service: IEmbeddingService = Depends(get_embedding_service),
    vector_repo: IVectorRepository = Depends(get_vector_repository),
):
    """Get sync documents use case"""
    from ...application.use_cases.sync_commit_documents import SyncCommitDocumentsUseCase
    return SyncCommitDocumentsUseCase(
        commit_repo,
        document_repo,
        embedding_service,
        vector_repo,
    )


async def get_index_document_use_case(
    document_repo: IDocumentRepository = Depends(get_document_repository),
    vector_repo: IVectorRepository = Depends(get_vector_repository),
    embedding_service: IEmbeddingService = Depends(get_embedding_service),
    event_dispatcher: IEventDispatcher = Depends(get_event_dispatcher)
):
    """Get index document use case"""
    from ...application.use_cases.index_document import IndexDocumentUseCase
    return IndexDocumentUseCase(
        document_repo,
        vector_repo,
        embedding_service,
        event_dispatcher
    )


async def get_process_commit_use_case(
        repo: ICommitRepository = Depends(get_commit_repository),
        dispatcher: IEventDispatcher = Depends(get_event_dispatcher),
        sync_commits_use_case = Depends(get_sync_documents_use_case),
):
    """Get process commit use case with all dependencies"""
    from ...application.use_cases.process_commit import ProcessCommitUseCase
    return ProcessCommitUseCase(repo, dispatcher, sync_commits_use_case)



async def get_search_documents_use_case(
    search_service: ISearchService = Depends(get_search_service),
    cache_service: ICacheService = Depends(get_cache_service)
):
    """Get search documents use case"""
    from ...application.use_cases.search_documents import SearchDocumentsUseCase
    return SearchDocumentsUseCase(search_service, cache_service)


async def get_rag_chat_use_case(
    rag_service: IRAGService = Depends(get_rag_service),
    search_service: ISearchService = Depends(get_search_service),
    document_repo: IDocumentRepository = Depends(get_document_repository),
    cache_service: ICacheService = Depends(get_cache_service)
):
    """Get RAG chat use case"""
    from ...application.use_cases.rag_chat import RAGChatUseCase
    return RAGChatUseCase(
        rag_service,
        search_service,
        document_repo,
        cache_service
    )


async def get_sync_commits_use_case(
    commit_repo: ICommitRepository = Depends(get_commit_repository),
    document_repo: IDocumentRepository = Depends(get_document_repository),
    embedding_service: IEmbeddingService = Depends(get_embedding_service),
    vector_repo: IVectorRepository = Depends(get_vector_repository)
):
    """Get sync commits use case"""
    from ...application.use_cases.sync_commit_documents import SyncCommitDocumentsUseCase
    return SyncCommitDocumentsUseCase(
        commit_repo,
        document_repo,
        embedding_service,
        vector_repo
    )
