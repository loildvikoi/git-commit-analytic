# src/domain/entities/workflow.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum

from .agent import AgentRole
from .base import Entity, ValueObject


@dataclass
class WorkflowStep(ValueObject):
    """A step in a workflow"""
    name: str
    agent_role: AgentRole
    input_required: List[str]
    output_expected: List[str]
    timeout_seconds: int = 30

    def _validate(self):
        if not self.name:
            raise ValueError("Step name is required")
        if self.timeout_seconds < 1:
            raise ValueError("Timeout must be positive")


class Workflow(Entity):
    """Workflow entity - orchestrates multiple agents"""

    def __init__(
            self,
            name: str,
            description: str,
            steps: List[WorkflowStep],
            parallel_execution: bool = False
    ):
        super().__init__()
        self.name = name
        self.description = description
        self.steps = steps
        self.parallel_execution = parallel_execution

        # Runtime state
        self.current_step_index: int = 0
        self.status: str = "pending"
        self.results: Dict[str, Any] = {}
        self.errors: List[str] = []

    @property
    def current_step(self) -> Optional[WorkflowStep]:
        """Get current workflow step"""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def advance_step(self):
        """Move to next step"""
        self.current_step_index += 1
        if self.current_step_index >= len(self.steps):
            self.status = "completed"
        else:
            self.status = "in_progress"

    def add_result(self, step_name: str, result: Any):
        """Add step result"""
        self.results[step_name] = result
        self.updated_at = datetime.now()

    def add_error(self, error: str):
        """Add error"""
        self.errors.append(error)
        self.status = "failed"