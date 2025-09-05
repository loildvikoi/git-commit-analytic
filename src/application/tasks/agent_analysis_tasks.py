from celery import shared_task
from celery.utils.log import get_task_logger
from typing import Dict, Any, Optional
import traceback
import asyncio

from src.domain.entities.agent import AgentRole
from src.domain.entities.commit import Commit
from src.domain.entities.document import Document
from src.domain.repositories.commit_repository import ICommitRepository
from src.domain.repositories.document_repository import IDocumentRepository
from src.domain.repositories.vector_repository import IVectorRepository
from src.domain.services.agent_service import IAgentService
from src.domain.services.ai_analyzer import IAIAnalyzer
from src.domain.services.embedding_service import IEmbeddingService
from src.domain.services.event_dispatcher import IEventDispatcher
from src.domain.services.search_service import ISearchService
from src.infrastructure.persistence.database import get_session
from src.application.events.commit_events import (
    CommitAnalysisStartedEvent,
    CommitAnalysisCompletedEvent,
    CommitAnalysisFailedEvent
)
from src.interface.api.dependencies import get_commit_repository, get_ai_service, get_event_dispatcher, \
    get_document_repository, get_search_service, get_agent_service, get_vector_repository, get_embedding_service

logger = get_task_logger(__name__)


@shared_task(bind=True, name='agent_analyze_commit')
def agent_analyze_commit_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze commit with AI Agent - accepts payload dict.
    Since this is a Celery task:
    - it should be idempotent (can be retried without issues).
    - it should not perform any heavy computations or long-running tasks.
    - it should handle exceptions gracefully and log them.
    - it should communicate with application layer through events.
    """
    try:
        # Extract commit_id from payload
        commit_id = payload.get("commit_id")
        if not commit_id:
            raise ValueError("commit_id is required in payload")

        logger.info(f"Starting agent analysis for commit: {commit_id}")

        # Extract commit_id from payload
        document_id = payload.get("document_id")
        if not document_id:
            raise ValueError("document_id is required in payload")

        logger.info(f"Starting agent analysis for document: {document_id}")

        # Run async analysis
        result = asyncio.run(_agent_analyze_commit_async(commit_id, document_id))

        logger.info(f"Processed agent analyze commit: {commit_id}")
        return result

    except Exception as e:
        logger.error(f"Failed to agent analyze commit {payload.get('commit_id', 'unknown')}: {str(e)}")
        logger.error(traceback.format_exc())

        # Fire failure event
        commit_id = payload.get("commit_id", "unknown")
        asyncio.run(_fire_analysis_failed_event(commit_id, str(e), self.request.retries))

        # Re-raise for Celery retry mechanism
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


async def _agent_analyze_commit_async(commit_id: str, document_id: str) -> Dict[str, Any]:
    """Async function to analyze commit"""

    # Get dependencies
    async with get_session() as session:
        try:
            commit_repo: ICommitRepository = await get_commit_repository(session)

            # Load commit
            commit = await commit_repo.find_by_id(commit_id)
            if not commit:
                return {
                    'status': 'error',
                    'error': f"Commit {commit_id} not found"
                }

            agent_service: IAgentService = await get_agent_service()
            document_repo: IDocumentRepository = await get_document_repository(session)
            vector_repo: IVectorRepository = await get_vector_repository()
            embedding_service: IEmbeddingService = await get_embedding_service()
            search_service: ISearchService = await get_search_service(document_repo, vector_repo, embedding_service)

            # Get commit and document
            commit: Optional[Commit] = await commit_repo.find_by_id(commit_id)
            document: Optional[Document] = await document_repo.find_by_id(document_id)

            if not commit or not document:
                logger.error(f"Commit or document not found for agent analysis")
                return {
                    'status': 'failed',
                    'error': f"Commit {commit_id} or document {document_id} not found"
                }

            # Create or get code reviewer agent
            reviewer = await agent_service.create_agent(
                name="AutoReviewer",
                role=AgentRole.CODE_REVIEWER,
                model="llama3.2:1b"
            )

            # Prepare context with similar commits (using Phase 2 RAG)
            from ...domain.services.search_service import SearchQuery
            search_query = SearchQuery(
                text=commit.message,
                use_hybrid=True,
                max_results=5
            )
            similar_commits = await search_service.search(search_query)

            context = {
                "commit_hash": commit.commit_hash.value,
                "message": commit.message,
                "files_changed": len(commit.files_changed),
                "similar_commits": [
                    {
                        "message": result.document.title,
                        "score": result.score.combined_score
                    }
                    for result in similar_commits
                ]
            }

            # Execute agent analysis
            logger.info(f"Executing agent analysis for commit {commit.id} with context: {context}")
            analysis_result = await agent_service.execute_task(
                agent=reviewer,
                task="Analyze this commit for code quality, potential bugs, and improvements",
                context=context
            )

            # Update document with agent analysis
            document.summary = f"Agent Analysis: {analysis_result.get('output', '')[:500]}"
            document.tags.append("agent_analyzed")
            await document_repo.update(document)
            logger.info(f"Document {document.id} updated with agent analysis")

            # Check if this is a risky commit
            if "bug" in analysis_result.get('output', '').lower():
                # Create bug detector agent for deeper analysis
                bug_detector = await agent_service.create_agent(
                    name="BugDetector",
                    role=AgentRole.BUG_DETECTOR,
                    model="llama3.2:1b"
                )

                bug_analysis = await agent_service.execute_task(
                    agent=bug_detector,
                    task="Identify specific bugs and edge cases in this commit",
                    context={**context, "initial_analysis": analysis_result}
                )

                # Fire alert event if critical bugs found
                if "critical" in bug_analysis.get('output', '').lower():
                    from ..events.agent_events import AgentAnalysisRequestedEvent, CriticalBugDetectedEvent
                    alert_event = CriticalBugDetectedEvent(
                        commit_id=commit.id,
                        bug_analysis=bug_analysis.get('output', ''),
                        severity="critical"
                    )
                    # This would trigger notifications

            logger.info(f"Agent analysis completed for commit {commit.id}")

            return {
                'status': 'success',
                'commit_id': commit_id,
                'summary': document.summary,
            }
        except Exception as e:
            logger.error(f"Error during agent analysis for commit {commit_id}: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e)
            }


async def _fire_analysis_failed_event(commit_id: str, error_message: str, retry_count: int):
    """Fire analysis failed event"""
    failed_event = CommitAnalysisFailedEvent(
        commit_id=commit_id,
        error_message=error_message,
        retry_count=retry_count
    )
    # Note: In a full implementation, we'd dispatch this event


@shared_task(name='generate_summary')
def generate_summary_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate project summary for recent commits - accepts payload dict"""
    try:
        project = payload.get("project")
        days = payload.get("days", 7)

        if not project:
            raise ValueError("project is required in payload")

        logger.info(f"Generating summary for project: {project}")

        result = asyncio.run(_generate_summary_async(project, days))

        logger.info(f"Successfully generated summary for project: {project}")
        return result

    except Exception as e:
        logger.error(f"Failed to generate summary for project {payload.get('project', 'unknown')}: {str(e)}")
        logger.error(traceback.format_exc())
        raise


async def _generate_summary_async(project: str, days: int) -> Dict[str, Any]:
    """Async function to generate summary"""
    from datetime import datetime, timedelta

    # Get dependencies
    async with get_session() as session:
        commit_repo: ICommitRepository = await get_commit_repository(session)
        ai_service: IAIAnalyzer = await get_ai_service()

        # Get recent commits
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        commits = await commit_repo.find_by_project(
            project=project,
            start_date=start_date,
            end_date=end_date,
            limit=50
        )

        if not commits:
            return {
                'status': 'success',
                'project': project,
                'summary': f"No commits found for {project} in the last {days} days.",
                'commits_count': 0
            }

        # Generate summary
        summary = await ai_service.generate_summary(commits)

        return {
            'status': 'success',
            'project': project,
            'summary': summary,
            'commits_count': len(commits),
            'period_days': days
        }
