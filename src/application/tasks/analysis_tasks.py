from celery import shared_task
from celery.utils.log import get_task_logger
from typing import Dict, Any
import traceback
import asyncio

from src.domain.repositories.commit_repository import ICommitRepository
from src.domain.services.ai_analyzer import IAIAnalyzer
from src.domain.services.event_dispatcher import IEventDispatcher
from src.infrastructure.persistence.database import get_session
from src.application.events.commit_events import (
    CommitAnalysisStartedEvent,
    CommitAnalysisCompletedEvent,
    CommitAnalysisFailedEvent
)
from src.interface.api.dependencies import get_commit_repository, get_ai_service, get_event_dispatcher

logger = get_task_logger(__name__)


@shared_task(bind=True, name='analyze_commit')
def analyze_commit_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze commit with AI - accepts payload dict.
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

        logger.info(f"Starting analysis for commit: {commit_id}")

        # Run async analysis
        result = asyncio.run(_analyze_commit_async(commit_id))

        logger.info(f"Successfully analyzed commit: {commit_id}")
        return result

    except Exception as e:
        logger.error(f"Failed to analyze commit {payload.get('commit_id', 'unknown')}: {str(e)}")
        logger.error(traceback.format_exc())

        # Fire failure event
        commit_id = payload.get("commit_id", "unknown")
        asyncio.run(_fire_analysis_failed_event(commit_id, str(e), self.request.retries))

        # Re-raise for Celery retry mechanism
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


async def _analyze_commit_async(commit_id: str) -> Dict[str, Any]:
    """Async function to analyze commit"""

    # Get dependencies
    async with get_session() as session:
        commit_repo: ICommitRepository = await get_commit_repository(session)
        ai_service: IAIAnalyzer = await get_ai_service()

        # Load commit
        commit = await commit_repo.find_by_id(commit_id)
        if not commit:
            return {
                'status': 'error',
                'error': f"Commit {commit_id} not found"
            }

        # Fire analysis started event
        # started_event = CommitAnalysisStartedEvent(
        #     commit_id=commit_id,
        #     model_name=ai_service.get_model_info()['name']
        # )

        # Generate AI analysis
        analysis_result = await ai_service.analyzer_commit(commit)

        # Update commit with analysis
        commit.mark_as_analyzed(
            summary=analysis_result.summary,
            tags=analysis_result.tags,
            sentiment=analysis_result.sentiment_score
        )

        updated_commit = await commit_repo.update(commit)

        # Fire analysis completed event
        completed_event = CommitAnalysisCompletedEvent(
            commit_id=commit_id,
            analysis_id=f"analysis_{commit_id}",
            summary=analysis_result.summary,
            tags=analysis_result.tags,
            processing_time_ms=1000  # Placeholder
        )
        event_dispatcher: IEventDispatcher = await get_event_dispatcher()
        await event_dispatcher.dispatch(completed_event)

        return {
            'status': 'success',
            'commit_id': commit_id,
            'summary': analysis_result.summary,
            'tags': analysis_result.tags,
            'sentiment_score': analysis_result.sentiment_score,
            'confidence_score': analysis_result.confidence_score
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
