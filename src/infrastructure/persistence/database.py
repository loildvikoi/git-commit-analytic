from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData
from contextlib import asynccontextmanager
import os
import logging

logger = logging.getLogger(__name__)

# SQLAlchemy base
Base = declarative_base()
metadata = MetaData()

# Global engine and session maker
engine = None
SessionLocal = None


def get_database_url() -> str:
    """Get database URL from environment"""
    db_provider = os.getenv("DB_PROVIDER", "sqlite")

    if db_provider == "sqlite":
        db_path = os.getenv("DB_PATH", "data/git_analytics.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return f"sqlite+aiosqlite:///{db_path}"

    elif db_provider == "postgresql":
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        name = os.getenv("DB_NAME", "git_analytics")
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "")
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"

    else:
        raise ValueError(f"Unsupported database provider: {db_provider}")


async def init_database():
    """Initialize database connection"""
    global engine, SessionLocal

    if engine is None:
        database_url = get_database_url()
        logger.info(f"Connecting to database: {database_url.split('://')[0]}://...")

        engine = create_async_engine(
            database_url,
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
            pool_pre_ping=True,
            pool_recycle=3600
        )

        SessionLocal = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # Create tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database initialized successfully")

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
        await init_database()

    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
