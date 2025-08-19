from typing import Dict, Any, Optional, Callable

from celery import Celery

from src.domain.services.message_queue import IMessageQueue, TaskStatus


class CeleryMessageQueue(IMessageQueue):
    """Celery implementation of IMessageQueue"""

    def __init__(self, celery_app):
        self.celery_app = celery_app

    async def enqueue(self, task_name: str, payload: Dict[str, Any], delay_seconds: int = 0, priority: int = 5) -> str:
        """Enqueue task for processing"""
        task = self.celery_app.send_task(task_name, args=[payload], countdown=delay_seconds, priority=priority)
        return task.id

    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get task status"""
        task = self.celery_app.AsyncResult(task_id)
        return TaskStatus(task.status)

    async def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get task result"""
        task = self.celery_app.AsyncResult(task_id)
        return task.result if task.successful() else None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel pending task"""
        task = self.celery_app.AsyncResult(task_id)
        if not task.ready():
            return task.revoke(terminate=True)
        return False

    def register_task(self, task_name: str, handler: Callable):
        """Register task handler"""
        self.celery_app.task(name=task_name)(handler)


def create_celery_app(broker_url: str, backend_url: str) -> Celery:
    """Create and configure Celery app"""
    from celery import Celery

    celery_app = Celery('tasks', broker=broker_url, backend=backend_url)
    celery_app.conf.update(
        task_serializer='json',
        result_serializer='json',
        accept_content=['json'],
        timezone='UTC',
        enable_utc=True,
        task_default_queue='default',
        task_default_exchange='default',
        task_default_routing_key='default'
    )
    return celery_app


celery_app = create_celery_app(
    broker_url='redis://localhost:6379/0',
    backend_url='redis://localhost:6379/0'
)
