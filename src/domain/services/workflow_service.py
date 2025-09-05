# src/domain/services/workflow_service.py
from abc import ABC, abstractmethod
from ..entities.workflow import Workflow, WorkflowStep
from ..entities.agent import Agent


class IWorkflowService(ABC):
    """Service interface for workflow orchestration"""

    @abstractmethod
    async def create_workflow(
            self,
            name: str,
            steps: List[WorkflowStep]
    ) -> Workflow:
        """Create a new workflow"""
        pass

    @abstractmethod
    async def execute_workflow(
            self,
            workflow: Workflow,
            input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a complete workflow"""
        pass

    @abstractmethod
    async def execute_step(
            self,
            workflow: Workflow,
            step: WorkflowStep,
            agent: Agent,
            input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step"""
        pass

    @abstractmethod
    async def get_workflow_status(
            self,
            workflow_id: str
    ) -> Dict[str, Any]:
        """Get workflow execution status"""
        pass

