from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData
from contextlib import asynccontextmanager
import logging

from ...core.config import settings

logger = logging.getLogger(__name__)

# Disable SQLAlchemy logging if not in debug mode
if not settings.debug:
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.dialects').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.orm').setLevel(logging.WARNING)

# SQLAlchemy base
Base = declarative_base()
metadata = MetaData()

# Import all models to ensure they're registered with SQLAlchemy
from .models.commit_model import CommitModel, AnalysisModel
from .models.document_model import DocumentModel

# Global engine and session maker
engine = None
SessionLocal = None


async def init_database_connection():
    """Initialize database connection"""
    global engine, SessionLocal

    if engine is None:
        database_url = settings.database_url
        logger.info(f"Connecting to database: {database_url.split('://')[0]}://...")

        engine = create_async_engine(
            database_url,
            echo=settings.db_echo,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow
        )

        SessionLocal = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # Create tables
        # async with engine.begin() as conn:
        #     await conn.run_sync(Base.metadata.create_all)

        logger.info("Database connection initialized successfully")


async def close_database():
    """Close database connection"""
    global engine
    if engine:
        await engine.dispose()
        logger.info("Database connection closed")


@asynccontextmanager
async def get_session() -> AsyncSession:
    """Get database session"""
    if SessionLocal is None:
        await init_database_connection()

    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Health check function
async def check_database_health() -> bool:
    """Check database connection health"""
    try:
        from sqlalchemy import text
        async with get_session() as session:
            result = await session.execute(text("SELECT 1"))
            result.scalar()
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


# Database utility functions
async def get_database_info() -> dict:
    """Get database information"""
    return {
        "provider": settings.db_provider.value,
        "host": settings.db_host if settings.db_provider.value != "sqlite" else None,
        "database": settings.db_name if settings.db_provider.value != "sqlite" else settings.db_path,
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "echo": settings.db_echo
    }
