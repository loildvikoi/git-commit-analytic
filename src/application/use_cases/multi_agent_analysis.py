# src/application/use_cases/multi_agent_analysis.py
from typing import Dict, Any

from src.domain.entities.agent import AgentRole
from src.domain.repositories.commit_repository import ICommitRepository
from src.domain.services.agent_service import IAgentService
from src.domain.services.workflow_service import IWorkflowService


class MultiAgentAnalysisUseCase:
    """Use case for multi-agent collaborative analysis"""

    def __init__(
            self,
            agent_service: IAgentService,
            workflow_service: IWorkflowService,
            commit_repository: ICommitRepository
    ):
        self.agent_service = agent_service
        self.workflow_service = workflow_service
        self.commit_repository = commit_repository

    async def execute(
            self,
            project: str,
            analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Execute multi-agent analysis"""

        # 1. Define workflow based on analysis type
        workflow = await self._create_analysis_workflow(analysis_type)

        # 2. Get project commits
        commits = await self.commit_repository.find_by_project(
            project=project,
            limit=100
        )

        # 3. Prepare input data
        input_data = {
            "project": project,
            "commits": [c.to_dict() for c in commits],
            "analysis_type": analysis_type
        }

        # 4. Execute workflow with multiple agents
        results = await self.workflow_service.execute_workflow(
            workflow=workflow,
            input_data=input_data
        )

        # 5. Aggregate results from all agents
        aggregated = self._aggregate_agent_results(results)

        return {
            "project": project,
            "analysis_type": analysis_type,
            "workflow_id": workflow.id,
            "agent_results": results,
            "summary": aggregated["summary"],
            "insights": aggregated["insights"],
            "recommendations": aggregated["recommendations"],
            "metrics": aggregated["metrics"]
        }

    async def _create_analysis_workflow(self, analysis_type: str):
        """Create workflow based on analysis type"""
        from ...domain.entities.workflow import WorkflowStep

        if analysis_type == "comprehensive":
            steps = [
                WorkflowStep(
                    name="code_quality",
                    agent_role=AgentRole.CODE_REVIEWER,
                    input_required=["commits"],
                    output_expected=["quality_report"],
                    timeout_seconds=60
                ),
                WorkflowStep(
                    name="bug_detection",
                    agent_role=AgentRole.BUG_DETECTOR,
                    input_required=["commits", "quality_report"],
                    output_expected=["bug_report"],
                    timeout_seconds=60
                ),
                WorkflowStep(
                    name="performance_analysis",
                    agent_role=AgentRole.PERFORMANCE_ANALYZER,
                    input_required=["commits"],
                    output_expected=["performance_report"],
                    timeout_seconds=60
                ),
                WorkflowStep(
                    name="security_scan",
                    agent_role=AgentRole.SECURITY_SCANNER,
                    input_required=["commits", "bug_report"],
                    output_expected=["security_report"],
                    timeout_seconds=60
                )
            ]
        else:
            steps = [
                WorkflowStep(
                    name="quick_review",
                    agent_role=AgentRole.CODE_REVIEWER,
                    input_required=["commits"],
                    output_expected=["review_report"],
                    timeout_seconds=30
                )
            ]

        return await self.workflow_service.create_workflow(
            name=f"{analysis_type}_analysis",
            steps=steps
        )

    def _aggregate_agent_results(self, results: Dict) -> Dict[str, Any]:
        """Aggregate results from multiple agents"""

        aggregated = {
            "summary": "",
            "insights": [],
            "recommendations": [],
            "metrics": {}
        }

        # Combine all agent outputs
        for step_name, step_result in results.items():
            if "summary" in step_result:
                aggregated["summary"] += f"\n{step_name}: {step_result['summary']}"
            if "insights" in step_result:
                aggregated["insights"].extend(step_result["insights"])
            if "recommendations" in step_result:
                aggregated["recommendations"].extend(step_result["recommendations"])
            if "metrics" in step_result:
                aggregated["metrics"][step_name] = step_result["metrics"]

        return aggregated

