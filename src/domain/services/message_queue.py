from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from enum import Enum


class TaskStatus(Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"


class IMessageQueue(ABC):
    """Interface for message queue"""

    @abstractmethod
    async def enqueue(
            self,
            task_name: str,
            payload: Dict[str, Any],
            delay_seconds: int = 0,
            priority: int = 5
    ) -> str:
        """Enqueue task for processing"""
        pass

    @abstractmethod
    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get task status"""
        pass

    @abstractmethod
    async def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get task result"""
        pass

    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel pending task"""
        pass

    @abstractmethod
    def register_task(self, task_name: str, handler: Callable):
        """Register task handler"""
        pass
