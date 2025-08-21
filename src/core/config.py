from pydantic_settings import BaseSettings
from pydantic import Field, model_validator, field_validator, validator, root_validator
from typing import Optional, List, Dict, Any
from enum import Enum
import os


from dotenv import load_dotenv
import os

# Load base config
load_dotenv('.env')

# Override with environment-specific
env = os.getenv('ENVIRONMENT', 'development')
env_file = f'.env.{env}'
if os.path.exists(env_file):
    load_dotenv(env_file, override=True)


class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseProvider(str, Enum):
    """Database providers"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class Settings(BaseSettings):
    """Application settings with validation and computed properties"""

    # Application
    app_name: str = Field(default="Git AI Analytics", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Environment")
    debug: bool = Field(default=True, description="Debug mode")
    secret_key: str = Field(default="dev-secret-key-change-in-production", description="Secret key for signing")

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    workers: int = Field(default=1, ge=1, le=32, description="Number of workers")
    reload: bool = Field(default=True, description="Auto-reload on code changes")

    # CORS
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    cors_allow_credentials: bool = Field(default=True, description="CORS allow credentials")

    # Database
    db_provider: DatabaseProvider = Field(default=DatabaseProvider.SQLITE, description="Database provider")
    db_path: str = Field(default="data/git_analytics.db", description="SQLite database path")
    db_host: Optional[str] = Field(default=None, description="Database host")
    db_port: Optional[int] = Field(default=None, ge=1, le=65535, description="Database port")
    db_name: Optional[str] = Field(default=None, description="Database name")
    db_user: Optional[str] = Field(default=None, description="Database user")
    db_password: Optional[str] = Field(default=None, description="Database password")
    db_echo: bool = Field(default=False, description="Echo SQL queries")
    db_pool_size: int = Field(default=5, ge=1, le=50, description="Database connection pool size")
    db_max_overflow: int = Field(default=10, ge=0, le=100, description="Database max overflow connections")

    # Redis
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    redis_db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_prefix: str = Field(default="git_analytics", description="Redis key prefix")
    redis_max_connections: int = Field(default=20, ge=1, le=100, description="Redis max connections")

    # Celery
    celery_broker_db: int = Field(default=0, ge=0, le=15, description="Celery broker Redis DB")
    celery_result_db: int = Field(default=1, ge=0, le=15, description="Celery result Redis DB")
    celery_task_serializer: str = Field(default="json", description="Celery task serializer")
    celery_result_serializer: str = Field(default="json", description="Celery result serializer")
    celery_timezone: str = Field(default="UTC", description="Celery timezone")
    celery_worker_prefetch_multiplier: int = Field(default=1, ge=1, le=10, description="Celery worker prefetch")

    # AI/Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    ollama_model: str = Field(default="llama3.2:1b", description="Ollama model name")
    ollama_timeout: int = Field(default=30, ge=1, le=300, description="Ollama request timeout")
    ollama_max_retries: int = Field(default=3, ge=1, le=10, description="Ollama max retries")

    # OpenAI (backup)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model")

    # Security
    webhook_secret: Optional[str] = Field(default=None, description="GitHub webhook secret")
    api_key: Optional[str] = Field(default=None, description="API authentication key")
    jwt_secret_key: Optional[str] = Field(default=None, description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expire_minutes: int = Field(default=30, ge=1, le=10080, description="JWT expiration minutes")

    # Rate Limiting
    rate_limit_requests: int = Field(default=100, ge=1, le=10000, description="Rate limit requests per window")
    rate_limit_window: int = Field(default=60, ge=1, le=3600, description="Rate limit window in seconds")

    # Cache
    cache_ttl_default: int = Field(default=3600, ge=60, le=86400, description="Default cache TTL")
    cache_ttl_search: int = Field(default=300, ge=30, le=3600, description="Search cache TTL")
    cache_ttl_chat: int = Field(default=600, ge=60, le=3600, description="Chat cache TTL")
    memory_cache_size: int = Field(default=1000, ge=100, le=10000, description="Memory cache size")

    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    log_file: Optional[str] = Field(default=None, description="Log file path")
    log_rotation: str = Field(default="1 day", description="Log rotation")
    log_retention: str = Field(default="30 days", description="Log retention")

    # Vector Store & Embeddings
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model name")
    embedding_dimension: int = Field(default=384, ge=128, le=1536, description="Embedding dimensions")
    chroma_persist_dir: str = Field(default="./data/chroma", description="ChromaDB persist directory")
    chroma_collection_name: str = Field(default="git_documents", description="ChromaDB collection name")

    # Text Processing
    max_chunk_size: int = Field(default=500, ge=100, le=2000, description="Maximum chunk size")
    chunk_overlap: int = Field(default=50, ge=0, le=200, description="Chunk overlap size")
    min_chunk_size: int = Field(default=50, ge=10, le=500, description="Minimum chunk size")

    # RAG Configuration
    hybrid_search_semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Semantic search weight")
    rag_max_context_length: int = Field(default=2000, ge=500, le=8000, description="RAG max context length")
    rag_temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="RAG temperature")
    rag_top_k: int = Field(default=5, ge=1, le=20, description="RAG top-k results")
    rag_score_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="RAG score threshold")

    # Background Tasks
    max_concurrent_tasks: int = Field(default=10, ge=1, le=100, description="Max concurrent background tasks")
    task_timeout: int = Field(default=300, ge=30, le=3600, description="Task timeout in seconds")

    # File Processing
    max_file_size: int = Field(default=10 * 1024 * 1024, ge=1024, description="Max file size in bytes")  # 10MB
    allowed_file_types: List[str] = Field(
        default=[".txt", ".md", ".py", ".js", ".json", ".yaml", ".yml"],
        description="Allowed file types"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        use_enum_values = True

    @property
    def database_url(self) -> str:
        """Computed database URL"""
        if self.db_provider == DatabaseProvider.SQLITE:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            return f"sqlite+aiosqlite:///{self.db_path}"

        elif self.db_provider == DatabaseProvider.POSTGRESQL:
            password_part = f":{self.db_password}" if self.db_password else ""
            return f"postgresql+asyncpg://{self.db_user}{password_part}@{self.db_host}:{self.db_port}/{self.db_name}"

        elif self.db_provider == DatabaseProvider.MYSQL:
            password_part = f":{self.db_password}" if self.db_password else ""
            return f"mysql+aiomysql://{self.db_user}{password_part}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def redis_url(self) -> str:
        """Computed Redis URL"""
        password_part = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{password_part}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def celery_broker_url(self) -> str:
        """Computed Celery broker URL"""
        password_part = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{password_part}{self.redis_host}:{self.redis_port}/{self.celery_broker_db}"

    @property
    def celery_result_backend(self) -> str:
        """Computed Celery result backend URL"""
        password_part = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{password_part}{self.redis_host}:{self.redis_port}/{self.celery_result_db}"

    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION

    @property
    def is_testing(self) -> bool:
        """Check if running in testing"""
        return self.environment == Environment.TESTING

    def get_cors_origins(self) -> List[str]:
        """Get CORS origins based on environment"""
        if self.is_production:
            return [origin for origin in self.cors_origins if origin != "*"]
        return self.cors_origins

    def model_dump_safe(self) -> Dict[str, Any]:
        """Dump settings without sensitive information"""
        data = self.dict()

        # Remove sensitive fields
        sensitive_fields = [
            "db_password", "redis_password", "webhook_secret",
            "api_key", "jwt_secret_key", "openai_api_key", "secret_key"
        ]

        for field in sensitive_fields:
            if field in data and data[field]:
                data[field] = "***"

        return data


# Global settings instance
settings = Settings()
