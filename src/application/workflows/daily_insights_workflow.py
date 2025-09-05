# src/application/workflows/daily_insights_workflow.py
"""
Automated workflow that combines Phase 2 RAG with Phase 3 Agents
"""
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from ...domain.entities.workflow import Workflow, WorkflowStep
from ...domain.entities.agent import AgentRole

logger = logging.getLogger(__name__)


class DailyInsightsWorkflow:
    """
    Daily automated workflow that:
    1. Uses Phase 2 to search recent commits
    2. Uses Phase 3 agents to analyze patterns
    3. Generates insights and sends notifications
    """

    def __init__(
            self,
            workflow_service,
            search_service,
            agent_service,
            commit_repository
    ):
        self.workflow_service = workflow_service
        self.search_service = search_service
        self.agent_service = agent_service
        self.commit_repository = commit_repository

    async def execute(self) -> Dict[str, Any]:
        """Execute daily insights workflow"""

        # Step 1: Gather data using Phase 2 search
        recent_data = await self._gather_recent_data()

        # Step 2: Create workflow with multiple agents
        workflow = await self._create_workflow()

        # Step 3: Execute workflow
        results = await self.workflow_service.execute_workflow(
            workflow=workflow,
            input_data=recent_data
        )

        # Step 4: Generate and distribute insights
        insights = await self._generate_insights(results)

        return insights

    async def _gather_recent_data(self) -> Dict[str, Any]:
        """Use Phase 2 to gather recent data"""

        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)

        # Get recent commits
        commits = await self.commit_repository.search(
            query="",
            filters={
                "start_date": start_date,
                "end_date": end_date
            },
            limit=100
        )

        # Use Phase 2 semantic search to find patterns
        from ...domain.services.search_service import SearchQuery

        # Search for bugs
        bug_search = SearchQuery(
            text="bug fix error issue problem",
            use_hybrid=True,
            max_results=10
        )
        bug_related = await self.search_service.search(bug_search)

        # Search for features
        feature_search = SearchQuery(
            text="feature add implement new functionality",
            use_hybrid=True,
            max_results=10
        )
        feature_related = await self.search_service.search(feature_search)

        return {
            "date": datetime.now().isoformat(),
            "total_commits": len(commits),
            "commits": [c.to_dict() for c in commits],
            "bug_related": [r.document.to_dict() for r in bug_related],
            "feature_related": [r.document.to_dict() for r in feature_related]
        }

    async def _create_workflow(self) -> Workflow:
        """Create multi-agent workflow"""

        steps = [
            WorkflowStep(
                name="analyze_activity",
                agent_role=AgentRole.GENERAL_ASSISTANT,
                input_required=["commits", "date"],
                output_expected=["activity_summary"],
                timeout_seconds=30
            ),
            WorkflowStep(
                name="detect_risks",
                agent_role=AgentRole.BUG_DETECTOR,
                input_required=["bug_related", "activity_summary"],
                output_expected=["risk_assessment"],
                timeout_seconds=30
            ),
            WorkflowStep(
                name="review_quality",
                agent_role=AgentRole.CODE_REVIEWER,
                input_required=["commits", "activity_summary"],
                output_expected=["quality_report"],
                timeout_seconds=30
            ),
            WorkflowStep(
                name="generate_recommendations",
                agent_role=AgentRole.GENERAL_ASSISTANT,
                input_required=["risk_assessment", "quality_report"],
                output_expected=["recommendations"],
                timeout_seconds=30
            )
        ]

        return await self.workflow_service.create_workflow(
            name="daily_insights",
            steps=steps
        )

    async def _generate_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final insights from workflow results"""

        insights = {
            "date": datetime.now().isoformat(),
            "type": "daily_insights",
            "summary": "",
            "key_findings": [],
            "risks": [],
            "recommendations": [],
            "metrics": {}
        }

        # Extract from workflow results
        if "analyze_activity" in results:
            insights["summary"] = results["analyze_activity"].get("activity_summary", "")

        if "detect_risks" in results:
            insights["risks"] = self._parse_risks(
                results["detect_risks"].get("risk_assessment", "")
            )

        if "review_quality" in results:
            insights["metrics"]["quality_score"] = self._extract_quality_score(
                results["review_quality"].get("quality_report", "")
            )

        if "generate_recommendations" in results:
            insights["recommendations"] = self._parse_recommendations(
                results["generate_recommendations"].get("recommendations", "")
            )

        # Add Phase 2 enhanced context
        insights["enhanced_context"] = await self._add_historical_context(insights)

        return insights

    async def _add_historical_context(self, insights: Dict) -> Dict[str, Any]:
        """Use Phase 2 RAG to add historical context"""

        # Search for similar past patterns
        from ...domain.services.search_service import SearchQuery

        pattern_query = SearchQuery(
            text=insights.get("summary", ""),
            use_hybrid=True,
            max_results=5
        )

        similar_patterns = await self.search_service.search(pattern_query)

        return {
            "similar_past_patterns": [
                {
                    "date": p.document.created_at.isoformat(),
                    "summary": p.document.summary,
                    "similarity": p.score.combined_score
                }
                for p in similar_patterns
            ],
            "trend": self._analyze_trend(similar_patterns)
        }

    def _parse_risks(self, risk_text: str) -> List[Dict[str, Any]]:
        """Parse risks from agent output"""
        risks = []

        # Simple parsing logic
        if "high" in risk_text.lower():
            risks.append({
                "level": "high",
                "description": "High risk detected in recent commits"
            })

        return risks

    def _extract_quality_score(self, quality_report: str) -> float:
        """Extract quality score from report"""
        # Simple extraction logic
        return 7.5  # Placeholder

    def _parse_recommendations(self, recommendations_text: str) -> List[str]:
        """Parse recommendations from agent output"""
        # Split by newlines and filter
        lines = recommendations_text.split('\n')
        return [line.strip() for line in lines if line.strip()]

    def _analyze_trend(self, similar_patterns) -> str:
        """Analyze trend from similar patterns"""
        if not similar_patterns:
            return "No historical data"

        avg_score = sum(p.score.combined_score for p in similar_patterns) / len(similar_patterns)

        if avg_score > 0.7:
            return "Recurring pattern detected"
        else:
            return "Normal activity"
