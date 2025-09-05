# src/infrastructure/workflow/simple_workflow_service.py
from typing import Dict, Any, List
import asyncio
import logging
from datetime import datetime

from ...domain.services.workflow_service import IWorkflowService
from ...domain.entities.workflow import Workflow, WorkflowStep
from ...domain.entities.agent import Agent, AgentRole

logger = logging.getLogger(__name__)


class SimpleWorkflowService(IWorkflowService):
    """Simple implementation of workflow service"""

    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.agent_service = None  # Will be injected

    async def create_workflow(
            self,
            name: str,
            steps: List[WorkflowStep]
    ) -> Workflow:
        """Create a new workflow"""
        workflow = Workflow(
            name=name,
            description=f"Workflow with {len(steps)} steps",
            steps=steps,
            parallel_execution=False
        )

        self.workflows[workflow.id] = workflow
        logger.info(f"Created workflow: {name} with {len(steps)} steps")

        return workflow

    async def execute_workflow(
            self,
            workflow: Workflow,
            input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a complete workflow"""
        workflow.status = "in_progress"
        workflow_data = input_data.copy()

        try:
            # Execute each step sequentially
            for step in workflow.steps:
                logger.info(f"Executing workflow step: {step.name}")

                # Get or create agent for this step
                agent = await self._get_agent_for_role(step.agent_role)

                # Prepare step input
                step_input = self._prepare_step_input(step, workflow_data)

                # Execute step
                step_result = await self.execute_step(
                    workflow, step, agent, step_input
                )

                # Store result
                workflow.add_result(step.name, step_result)

                # Add to workflow data for next steps
                for output_key in step.output_expected:
                    if output_key in step_result:
                        workflow_data[output_key] = step_result[output_key]

                # Move to next step
                workflow.advance_step()

            workflow.status = "completed"
            logger.info(f"Workflow {workflow.name} completed successfully")

            return workflow.results

        except Exception as e:
            workflow.add_error(str(e))
            logger.error(f"Workflow {workflow.name} failed: {str(e)}")
            raise

    async def execute_step(
            self,
            workflow: Workflow,
            step: WorkflowStep,
            agent: Agent,
            input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step"""

        # Set timeout
        try:
            result = await asyncio.wait_for(
                self._execute_step_with_agent(agent, step, input_data),
                timeout=step.timeout_seconds
            )

            return result

        except asyncio.TimeoutError:
            logger.error(f"Step {step.name} timed out after {step.timeout_seconds}s")
            return {"error": "Step timed out"}

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow execution status"""
        workflow = self.workflows.get(workflow_id)

        if not workflow:
            return {"status": "not_found"}

        return {
            "workflow_id": workflow.id,
            "name": workflow.name,
            "status": workflow.status,
            "current_step": workflow.current_step.name if workflow.current_step else None,
            "completed_steps": list(workflow.results.keys()),
            "errors": workflow.errors
        }

    async def _get_agent_for_role(self, role: AgentRole) -> Agent:
        """Get or create agent for role"""
        if not self.agent_service:
            # Lazy import to avoid circular dependency
            from ...interface.api.dependencies import get_agent_service
            self.agent_service = await get_agent_service()

        # Create agent for role
        agent = await self.agent_service.create_agent(
            name=f"{role.value}_agent",
            role=role,
            model="llama3.2:1b"
        )

        return agent

    def _prepare_step_input(
            self,
            step: WorkflowStep,
            workflow_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare input for a step"""
        step_input = {}

        for required_key in step.input_required:
            if required_key in workflow_data:
                step_input[required_key] = workflow_data[required_key]
            else:
                logger.warning(f"Missing required input: {required_key} for step {step.name}")

        return step_input

    async def _execute_step_with_agent(
            self,
            agent: Agent,
            step: WorkflowStep,
            input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute step with agent"""

        # Build task description
        task = f"Execute {step.name}: Process the provided data and generate {', '.join(step.output_expected)}"

        # Execute with agent
        result = await self.agent_service.execute_task(
            agent=agent,
            task=task,
            context=input_data
        )

        # Parse agent output to expected format
        output = {}
        for expected_key in step.output_expected:
            # Simple extraction - in production, use better parsing
            output[expected_key] = result.get("output", "")

        return output

