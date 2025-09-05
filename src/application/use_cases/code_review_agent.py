# src/application/use_cases/code_review_agent.py
import logging
from typing import Dict, Any, List, Optional
from ...domain.entities.agent import Agent, AgentRole, AgentCapability, AgentDecision
from ...domain.services.agent_service import IAgentService
from ...domain.services.tool_service import IToolService
from ...domain.repositories.commit_repository import ICommitRepository
from ...domain.services.search_service import ISearchService

logger = logging.getLogger(__name__)


class CodeReviewAgentUseCase:
    """Use case for autonomous code review"""

    def __init__(
            self,
            agent_service: IAgentService,
            tool_service: IToolService,
            commit_repository: ICommitRepository,
            search_service: ISearchService
    ):
        self.agent_service = agent_service
        self.tool_service = tool_service
        self.commit_repository = commit_repository
        self.search_service = search_service

    async def execute(
            self,
            commit_id: str,
            review_depth: str = "standard"  # quick, standard, thorough
    ) -> Dict[str, Any]:
        """Execute autonomous code review"""

        # 1. Get commit details
        commit = await self.commit_repository.find_by_id(commit_id)
        if not commit:
            raise ValueError(f"Commit {commit_id} not found")

        # 2. Create or get code reviewer agent
        agent = await self._get_or_create_agent()

        # 3. Prepare review context
        context = await self._prepare_review_context(commit, review_depth)

        # 4. Execute multi-step review
        review_results = {}

        # Step 1: Analyze code changes
        code_analysis = await self.agent_service.execute_task(
            agent=agent,
            task="Analyze code changes for quality issues",
            context=context
        )
        review_results["code_quality"] = code_analysis

        # Step 2: Check for bugs
        bug_detection = await self.agent_service.execute_task(
            agent=agent,
            task="Identify potential bugs and edge cases",
            context={**context, "previous_analysis": code_analysis}
        )
        review_results["bug_detection"] = bug_detection

        # Step 3: Security review (if thorough)
        if review_depth == "thorough":
            security_review = await self.agent_service.execute_task(
                agent=agent,
                task="Review code for security vulnerabilities",
                context={**context, "previous_analyses": [code_analysis, bug_detection]}
            )
            review_results["security"] = security_review

        # 5. Generate final review decision
        decision = await self._make_review_decision(agent, review_results)

        # 6. Format and return results
        return {
            "commit_id": commit_id,
            "reviewer_agent": agent.name,
            "review_depth": review_depth,
            "analyses": review_results,
            "decision": decision,
            "recommendations": self._generate_recommendations(review_results),
            "score": self._calculate_review_score(review_results)
        }

    async def _get_or_create_agent(self) -> Agent:
        """Get or create code reviewer agent"""
        agent = await self.agent_service.create_agent(
            name="CodeReviewer-001",
            role=AgentRole.CODE_REVIEWER,
            model="llama3.2:1b"  # Using Ollama
        )
        return agent

    async def _prepare_review_context(
            self,
            commit,
            review_depth: str
    ) -> Dict[str, Any]:
        """Prepare context for review"""

        context = {
            "commit_hash": commit.commit_hash.value,
            "message": commit.message,
            "author": commit.author_name,
            "files_changed": [f.__dict__ for f in commit.files_changed],
            "timestamp": commit.timestamp.isoformat()
        }

        # Add historical context for thorough review
        if review_depth in ["standard", "thorough"]:
            # Find similar past commits
            similar_commits = await self.search_service.semantic_search(
                query_text=commit.message,
                limit=5
            )
            context["similar_commits"] = [
                {
                    "hash": c.document.metadata.get("commit_hash"),
                    "message": c.document.title,
                    "relevance": c.score.combined_score
                }
                for c in similar_commits
            ]

        return context

    async def _make_review_decision(
            self,
            agent: Agent,
            review_results: Dict[str, Any]
    ) -> AgentDecision:
        """Make final review decision"""

        decision_context = {
            "review_results": review_results,
            "question": "Should this commit be approved, needs changes, or rejected?"
        }

        decision = await self.agent_service.get_agent_decision(
            agent=agent,
            situation="Code review completion",
            options=["APPROVE", "REQUEST_CHANGES", "REJECT"]
        )

        return decision

    def _generate_recommendations(self, review_results: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Parse AI responses for issues
        for category, result in review_results.items():
            if "issues" in str(result).lower():
                recommendations.append(f"Address {category} issues identified")

        return recommendations

    def _calculate_review_score(self, review_results: Dict) -> float:
        """Calculate overall review score"""
        # Simple scoring logic
        base_score = 10.0

        for category, result in review_results.items():
            result_str = str(result).lower()
            if "critical" in result_str:
                base_score -= 3
            elif "major" in result_str:
                base_score -= 2
            elif "minor" in result_str:
                base_score -= 1

        return max(0, min(10, base_score))

