import logging
import os
from typing import Dict, Any, Optional, Callable, List
from celery import Celery

from src.domain.services.queue_service import IQueueService, TaskStatus
from src.core.config import settings

logger = logging.getLogger(__name__)


class CeleryQueue(IQueueService):
    """Celery queue implementation"""

    def __init__(self, celery_app):
        self.celery_app = celery_app

    async def enqueue(self, task_name: str, payload: Dict[str, Any], delay_seconds: int = 0, priority: int = 5) -> str:
        """Enqueue task for processing"""
        try:
            # Send payload as first argument to task
            task = self.celery_app.send_task(
                task_name,
                args=[payload],
                countdown=delay_seconds,
                priority=priority
            )
            logger.info(f"✅ Task enqueued: {task_name} with ID: {task.id}")
            return task.id
        except Exception as e:
            logger.error(f"❌ Failed to enqueue task {task_name}: {e}")
            raise

    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get task status"""
        try:
            task = self.celery_app.AsyncResult(task_id)
            status = TaskStatus(task.status)
            logger.debug(f"Task {task_id} status: {status}")
            return status
        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {e}")
            return TaskStatus("FAILURE")

    async def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get task result"""
        try:
            task = self.celery_app.AsyncResult(task_id)
            return task.result if task.successful() else None
        except Exception as e:
            logger.error(f"Failed to get task result for {task_id}: {e}")
            return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel pending task"""
        try:
            task = self.celery_app.AsyncResult(task_id)
            if not task.ready():
                task.revoke(terminate=True)
                logger.info(f"Task {task_id} cancelled")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False

    def register_task(self, task_name: str, handler: Callable):
        """Register task handler"""
        return self.celery_app.task(name=task_name)(handler)


def create_celery_app(
        broker_url: str,
        backend_url: str,
        celery_tasks: List[str] = None,
        app_name: str = 'git_analytics',
        redis_prefix: str = None
) -> Celery:
    """Create and configure Celery app with Redis prefix support"""

    if celery_tasks is None:
        celery_tasks = []

    logger.info(f"🔧 Creating Celery app: {app_name}")
    logger.info(f"📡 Broker: {broker_url}")
    logger.info(f"💾 Backend: {backend_url}")
    logger.info(f"📦 Tasks: {celery_tasks}")

    celery_app = Celery(app_name, broker=broker_url, backend=backend_url)

    # Enhanced configuration
    celery_app.conf.update(
        # Serialization
        task_serializer=settings.celery_task_serializer,
        result_serializer=settings.celery_result_serializer,
        accept_content=['json'],

        # Timezone
        timezone=settings.celery_timezone,
        enable_utc=True,

        # Queue configuration
        task_default_queue='default',
        task_default_exchange='default',
        task_default_routing_key='default',

        # Task execution
        task_acks_late=True,
        worker_prefetch_multiplier=settings.celery_worker_prefetch_multiplier,
        task_reject_on_worker_lost=True,

        # Result backend
        result_expires=settings.cache_ttl_default,  # Use cache TTL
        result_persistent=True,

        # Redis specific
        redis_max_connections=settings.redis_max_connections,
        redis_retry_on_timeout=True,

        # Task routing
        task_routes={
            'src.application.tasks.analysis_tasks.*': {'queue': 'analysis'},
        },

        # Worker configuration
        worker_hijack_root_logger=False,
        worker_log_color=True,

        # Include task modules
        include=celery_tasks,

        # Enable task events for monitoring
        task_send_sent_event=True,
        worker_send_task_events=True,
        task_track_started=True,
    )

    # Set Redis key prefix using Celery configuration (correct way)
    if redis_prefix:
        celery_redis_prefix = f'{redis_prefix}:celery'

        celery_app.conf.update(
            # Redis key prefix for broker
            broker_transport_options={
                'global_keyprefix': f'{celery_redis_prefix}:',
                'fanout_prefix': True,
                'fanout_patterns': True
            },
            # Redis key prefix for result backend
            result_backend_transport_options={
                'global_keyprefix': f'{celery_redis_prefix}:results:',
            }
        )

    logger.info(f"✅ Celery app created successfully")
    return celery_app


# Task modules to include
celery_tasks = [
    'src.application.tasks.analysis_tasks',
]

# Environment setup for multiprocessing
os.environ.setdefault('FORKED_BY_MULTIPROCESSING', '1')

# Create Celery app with configuration from settings
logger.info("🚀 Initializing Celery with settings...")
celery_app = create_celery_app(
    broker_url=settings.celery_broker_url,
    backend_url=settings.celery_result_backend,
    celery_tasks=celery_tasks,
    app_name=f"{settings.app_name.lower().replace(' ', '_')}_{settings.environment}",
    redis_prefix=settings.redis_prefix
)

# Export the main app
__all__ = ['celery_app', 'CeleryQueue', 'create_celery_app']