import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.core.config import settings
from src.core.logging import setup_logging
from src.infrastructure.persistence.database import init_database, close_database
from src.interface.api.v1 import webhooks, commits, chat
from src.interface.websocket.handlers import websocket_endpoint

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    # Initialize database
    await init_database()

    yield

    # Shutdown
    logger.info("Shutting down application")
    await close_database()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Git commit analytics with AI-powered insights",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(webhooks.router, prefix="/api/v1")
app.include_router(commits.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    await websocket_endpoint(websocket)


# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment
    }


# Root endpoint with simple UI
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Git AI Analytics</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .method { font-weight: bold; color: #2196F3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 Git AI Analytics API</h1>
                <p>AI-powered Git commit analysis and search system</p>

                <div class="endpoint">
                    <span class="method">POST</span> /api/v1/webhooks/github - GitHub webhook handler
                </div>

                <div class="endpoint">
                    <span class="method">GET</span> /api/v1/commits/ - Search commits
                </div>

                <div class="endpoint">
                    <span class="method">GET</span> /api/v1/commits/{id} - Get specific commit
                </div>

                <div class="endpoint">
                    <span class="method">POST</span> /api/v1/chat/ - Chat with AI about commits
                </div>

                <div class="endpoint">
                    <span class="method">GET</span> /health - Health check
                </div>

                <div class="endpoint">
                    <span class="method">GET</span> /docs - Interactive API documentation
                </div>

                <h2>WebSocket:</h2>
                <div class="endpoint">
                    <span class="method">WS</span> /ws - Real-time updates
                </div>

                <h2>Quick Start:</h2>
                <ol>
                    <li>Setup webhook: POST to /api/v1/webhooks/github</li>
                    <li>Search commits: GET /api/v1/commits/?q=bugfix</li>
                    <li>Chat with AI: POST /api/v1/chat/ with your question</li>
                </ol>
            </div>
        </body>
        </html>
        """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=settings.workers if not settings.reload else 1,
        log_level=settings.log_level.lower()
    )
