import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer
from sqlalchemy import text
from starlette.middleware.sessions import SessionMiddleware

from src.core.config import settings
from src.core.logging import setup_logging
from src.infrastructure.persistence.database import check_database_health
from src.interface.api.v1 import webhooks, commits, chat
from src.interface.websocket.handlers import websocket_endpoint
from src.interface.api.v2 import documents, rag

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    startup_time = time.time()

    # Startup
    logger.info("=" * 50)
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"🌍 Environment: {settings.environment}")
    logger.info(f"🐛 Debug mode: {settings.debug}")
    logger.info(f"🔧 Config loaded from: {settings.environment} environment")
    logger.info("=" * 50)

    try:
        # Initialize database
        logger.info("📊 Initializing database connection...")
        from src.infrastructure.persistence.database import init_database_connection, close_database
        await init_database_connection()
        logger.info("✅ Database connection established")

        # Initialize event handlers
        logger.info("🔄 Initializing event handlers...")
        from src.domain.events.event_handler_registry import init_event_handlers
        await init_event_handlers()
        logger.info("✅ Event handlers initialized")

        # Initialize Redis event system
        logger.info("🔴 Initializing Redis event system...")
        from src.domain.events.event_handler_registry import init_redis_subscribers
        await init_redis_subscribers()
        logger.info("✅ Redis event system initialized")

        # Initialize vector store (Phase 2)
        logger.info("🔍 Initializing vector store...")
        try:
            from src.interface.api.dependencies import get_vector_repository
            vector_repo = await get_vector_repository()
            logger.info("✅ Vector store initialized")
        except Exception as e:
            logger.warning(f"⚠️ Vector store initialization failed: {e}")

        # Initialize embedding service (Phase 2)
        logger.info("🧠 Initializing embedding service...")
        try:
            from src.interface.api.dependencies import get_embedding_service
            embedding_service = await get_embedding_service()
            logger.info("✅ Embedding service initialized")
        except Exception as e:
            logger.warning(f"⚠️ Embedding service initialization failed: {e}")

        # Startup complete
        startup_duration = time.time() - startup_time
        logger.info("=" * 50)
        logger.info(f"🎉 Application started successfully in {startup_duration:.2f}s")
        logger.info(f"🌐 Server running on http://{settings.host}:{settings.port}")
        logger.info(f"📚 API docs available at http://{settings.host}:{settings.port}/docs")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        raise

    yield

    # Shutdown
    logger.info("🛑 Shutting down application...")

    try:
        # Close database
        await close_database()
        logger.info("✅ Database connection closed")

        # Cleanup Redis connections
        try:
            from src.interface.api.dependencies import get_redis_event_bus
            redis_bus = await get_redis_event_bus()
            await redis_bus.disconnect()
            logger.info("✅ Redis connections closed")
        except Exception as e:
            logger.warning(f"⚠️ Redis cleanup warning: {e}")

    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

    logger.info("👋 Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered Git commit analytics with RAG capabilities",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan
)

# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

# Session middleware (for state management)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
    max_age=settings.jwt_expire_minutes * 60
)

# Trusted host middleware (security)
if settings.is_production:
    trusted_hosts = ["yourdomain.com", "*.yourdomain.com"]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count", "X-Page-Count"],
)


# ============================================================================
# CUSTOM MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log requests in production"""
    if not settings.debug:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        return response
    return await call_next(request)


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    if settings.debug:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "type": type(exc).__name__,
                "path": request.url.path,
                "timestamp": time.time()
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "path": request.url.path,
                "timestamp": time.time()
            }
        )


# ============================================================================
# API ROUTERS
# ============================================================================

# Phase 1 routers
app.include_router(webhooks.router, prefix="/api/v1", tags=["Webhooks"])
app.include_router(commits.router, prefix="/api/v1", tags=["Commits"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])

# Phase 2 routers
app.include_router(documents.router, prefix="/api/v2", tags=["Documents"])
app.include_router(rag.router, prefix="/api/v2", tags=["RAG"])


# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket_endpoint(websocket)


# ============================================================================
# HEALTH & MONITORING ENDPOINTS
# ============================================================================

@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "timestamp": time.time()
    }


@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check():
    """Detailed health check with dependencies"""
    health_status = {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "timestamp": time.time(),
        "checks": {}
    }

    # Check database
    try:
        from src.infrastructure.persistence.database import get_session
        healthy = await check_database_health()
        if healthy:
            health_status["checks"]["database"] = {"status": "healthy"}
        else:
            health_status["checks"]["database"] = {"status": "unhealthy", "error": "Database connection failed"}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["database"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"

    # Check Redis
    try:
        from src.interface.api.dependencies import get_redis_event_bus
        redis_bus = await get_redis_event_bus()
        await redis_bus.is_connected()
        health_status["checks"]["redis"] = {"status": "healthy"}
    except Exception as e:
        health_status["checks"]["redis"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"

    # Check Vector Store
    try:
        from src.interface.api.dependencies import get_vector_repository
        vector_repo = await get_vector_repository()
        health_status["checks"]["vector_store"] = {"status": "healthy"}
    except Exception as e:
        health_status["checks"]["vector_store"] = {"status": "unhealthy", "error": str(e)}

    return health_status


@app.get("/info", tags=["Info"])
async def app_info():
    """Application information"""
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "debug": settings.debug,
        "config": settings.model_dump_safe(),  # Safe dump without secrets
        "features": {
            "phase_1": ["webhooks", "commits", "chat"],
            "phase_2": ["documents", "rag", "vector_search", "embeddings"]
        }
    }


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """Root endpoint with API documentation"""
    environment_color = {
        "development": "#4CAF50",
        "staging": "#FF9800",
        "production": "#F44336",
        "testing": "#2196F3"
    }.get(settings.environment, "#9E9E9E")

    return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{settings.app_name} - {settings.environment.title()}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .env-badge {{ display: inline-block; padding: 5px 15px; background: {environment_color}; color: white; border-radius: 20px; font-size: 12px; font-weight: bold; }}
                .endpoint {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #007bff; }}
                .method {{ font-weight: bold; color: #007bff; padding: 3px 8px; background: #e3f2fd; border-radius: 4px; font-size: 12px; }}
                .phase2 {{ border-left-color: #4CAF50; }}
                .phase2 .method {{ color: #4CAF50; background: #e8f5e8; }}
                .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
                .card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; }}
                .feature {{ margin: 5px 0; }}
                .feature:before {{ content: "✅ "; }}
                .quick-start {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 8px; margin: 20px 0; }}
                .links {{ text-align: center; margin-top: 30px; }}
                .links a {{ margin: 0 10px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; }}
                .links a:hover {{ background: #0056b3; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 {settings.app_name}</h1>
                    <p>AI-powered Git analysis with RAG capabilities</p>
                    <span class="env-badge">{settings.environment.upper()}</span>
                    <span class="env-badge">v{settings.app_version}</span>
                </div>

                <div class="quick-start">
                    <h3>🏃‍♂️ Quick Start - Phase 2:</h3>
                    <ol>
                        <li>Sync existing commits: <code>POST /api/v2/documents/sync-commits</code></li>
                        <li>Search with hybrid approach: <code>POST /api/v2/documents/search</code></li>
                        <li>Chat with RAG: <code>POST /api/v2/rag/chat</code></li>
                    </ol>
                </div>

                <div class="grid">
                    <div class="card">
                        <h3>Phase 1 Endpoints (Original):</h3>
                        <div class="endpoint">
                            <span class="method">POST</span> /api/v1/webhooks/github - GitHub webhook handler
                        </div>
                        <div class="endpoint">
                            <span class="method">GET</span> /api/v1/commits/ - Search commits
                        </div>
                        <div class="endpoint">
                            <span class="method">POST</span> /api/v1/chat/ - Chat with AI (basic)
                        </div>
                    </div>

                    <div class="card">
                        <h3>Phase 2 Endpoints (RAG-Enhanced):</h3>
                        <div class="endpoint phase2">
                            <span class="method">POST</span> /api/v2/documents/index - Index documents
                        </div>
                        <div class="endpoint phase2">
                            <span class="method">POST</span> /api/v2/documents/search - Hybrid search
                        </div>
                        <div class="endpoint phase2">
                            <span class="method">POST</span> /api/v2/documents/sync-commits - Sync commits to RAG
                        </div>
                        <div class="endpoint phase2">
                            <span class="method">POST</span> /api/v2/rag/chat - RAG-powered chat
                        </div>
                        <div class="endpoint phase2">
                            <span class="method">GET</span> /api/v2/rag/health - RAG system health
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h3>✨ Features:</h3>
                    <div class="feature">Vector embeddings with ChromaDB</div>
                    <div class="feature">Hybrid search (semantic + keyword)</div>
                    <div class="feature">Document chunking with overlap</div>
                    <div class="feature">RAG-based Q&A</div>
                    <div class="feature">Intelligent caching</div>
                    <div class="feature">Real-time WebSocket support</div>
                    <div class="feature">Environment-specific configurations</div>
                    <div class="feature">Comprehensive health checks</div>
                </div>

                <div class="links">
                    <a href="/docs">📚 API Documentation</a>
                    <a href="/health/detailed">🏥 Health Check</a>
                    <a href="/info">ℹ️ App Info</a>
                </div>
            </div>
        </body>
        </html>
        """


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Uvicorn configuration
    uvicorn_config = {
        "app": "src.main:app",
        "host": settings.host,
        "port": settings.port,
        "reload": settings.reload and settings.is_development,
        "workers": settings.workers if not settings.reload else 1,
        "log_level": settings.log_level.lower(),
        "access_log": settings.debug,
        "use_colors": True,
    }

    # Production-specific settings
    if settings.is_production:
        uvicorn_config.update({
            "reload": False,
            "access_log": False,
            "server_header": False,
            "date_header": False,
        })

    logger.info(f"🔧 Starting server with config: {uvicorn_config} ")
    uvicorn.run(**uvicorn_config)
