# src/application/use_cases/proactive_insights.py
from typing import Optional, Dict, Any, List

from src.domain.entities.agent import AgentRole, Agent
from src.domain.repositories.commit_repository import ICommitRepository
from src.domain.services.agent_service import IAgentService
from src.domain.services.cache_service import ICacheService
from src.domain.services.search_service import ISearchService


class ProactiveInsightsUseCase:
    """Use case for generating proactive insights"""

    def __init__(
            self,
            agent_service: IAgentService,
            commit_repository: ICommitRepository,
            search_service: ISearchService,
            cache_service: ICacheService
    ):
        self.agent_service = agent_service
        self.commit_repository = commit_repository
        self.search_service = search_service
        self.cache_service = cache_service

    async def execute(
            self,
            project: Optional[str] = None,
            lookback_days: int = 7
    ) -> Dict[str, Any]:
        """Generate proactive insights"""

        # 1. Create insights agent
        agent = await self.agent_service.create_agent(
            name="InsightsAgent",
            role=AgentRole.GENERAL_ASSISTANT,
            model="llama3.2:1b"
        )

        # 2. Gather data
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        commits = await self.commit_repository.find_by_project(
            project=project,
            start_date=start_date,
            end_date=end_date,
            limit=200
        ) if project else await self.commit_repository.search(
            query="",
            filters={"start_date": start_date, "end_date": end_date},
            limit=200
        )

        # 3. Analyze patterns
        patterns = await self._analyze_patterns(agent, commits)

        # 4. Identify risks
        risks = await self._identify_risks(agent, commits, patterns)

        # 5. Generate recommendations
        recommendations = await self._generate_recommendations(
            agent,
            patterns,
            risks
        )

        # 6. Create insights report
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": lookback_days
            },
            "statistics": {
                "total_commits": len(commits),
                "active_developers": len(set(c.author_email for c in commits)),
                "projects_affected": len(set(c.project for c in commits))
            },
            "patterns": patterns,
            "risks": risks,
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat()
        }

    async def _analyze_patterns(self, agent: Agent, commits: List) -> Dict[str, Any]:
        """Analyze commit patterns"""

        context = {
            "commits": [
                {
                    "message": c.message,
                    "author": c.author_name,
                    "timestamp": c.timestamp.isoformat(),
                    "files_count": len(c.files_changed)
                }
                for c in commits[-50:]  # Last 50 for context
            ]
        }

        analysis = await self.agent_service.execute_task(
            agent=agent,
            task="Identify patterns in these commits: frequency, types, team dynamics",
            context=context
        )

        return {
            "commit_frequency": self._calculate_frequency(commits),
            "peak_hours": self._find_peak_hours(commits),
            "ai_analysis": analysis
        }

    async def _identify_risks(
            self,
            agent: Agent,
            commits: List,
            patterns: Dict
    ) -> List[Dict[str, Any]]:
        """Identify potential risks"""

        risks = []

        # High commit frequency might indicate rushed work
        if patterns["commit_frequency"] > 10:  # per day
            risks.append({
                "type": "quality",
                "severity": "medium",
                "description": "High commit frequency may indicate rushed development"
            })

        # Check for lack of tests
        test_commits = [c for c in commits if any(
            'test' in f.filename.lower() for f in c.files_changed
        )]
        if len(test_commits) / len(commits) < 0.2:
            risks.append({
                "type": "testing",
                "severity": "high",
                "description": "Low test coverage in recent commits"
            })

        # Ask AI for additional risk assessment
        ai_risks = await self.agent_service.execute_task(
            agent=agent,
            task="Identify potential risks in this development pattern",
            context={"patterns": patterns, "existing_risks": risks}
        )

        return risks

    async def _generate_recommendations(
            self,
            agent: Agent,
            patterns: Dict,
            risks: List
    ) -> List[str]:
        """Generate actionable recommendations"""

        context = {
            "patterns": patterns,
            "risks": risks
        }

        recommendations = await self.agent_service.execute_task(
            agent=agent,
            task="Generate 5 specific, actionable recommendations to improve development process",
            context=context
        )

        return recommendations

    def _calculate_frequency(self, commits: List) -> float:
        """Calculate average commits per day"""
        if not commits:
            return 0
        dates = set(c.timestamp.date() for c in commits)
        return len(commits) / max(len(dates), 1)

    def _find_peak_hours(self, commits: List) -> List[int]:
        """Find peak commit hours"""
        hours = [c.timestamp.hour for c in commits]
        from collections import Counter
        hour_counts = Counter(hours)
        return [h for h, _ in hour_counts.most_common(3)]