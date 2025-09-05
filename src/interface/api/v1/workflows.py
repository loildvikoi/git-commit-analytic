# src/interface/api/v1/workflows.py
"""
API endpoints for automated workflows
"""
from fastapi import APIRouter, Depends, BackgroundTasks
import logging

router = APIRouter(prefix="/workflows", tags=["workflows"])
logger = logging.getLogger(__name__)


@router.post("/daily-insights")
async def trigger_daily_insights(
        background_tasks: BackgroundTasks,
        workflow=Depends(get_daily_insights_workflow)
):
    """Trigger daily insights workflow"""
    try:
        # Run in background
        background_tasks.add_task(workflow.execute)

        return {
            "status": "triggered",
            "message": "Daily insights workflow started in background"
        }

    except Exception as e:
        logger.error(f"Error triggering daily insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pr-review/{pr_id}")
async def automated_pr_review(
        pr_id: str,
        background_tasks: BackgroundTasks
):
    """Automated PR review using Phase 2 + 3"""
    try:
        # This would:
        # 1. Fetch PR commits (Phase 1 data)
        # 2. Search similar past PRs (Phase 2 RAG)
        # 3. Run multi-agent review (Phase 3 agents)

        background_tasks.add_task(
            run_pr_review_workflow,
            pr_id
        )

        return {
            "status": "triggered",
            "pr_id": pr_id,
            "message": "PR review started"
        }

    except Exception as e:
        logger.error(f"Error in PR review: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_pr_review_workflow(pr_id: str):
    """Run PR review workflow"""
    # Implementation here
    pass


async def get_daily_insights_workflow():
    """Get daily insights workflow"""
    from ...application.workflows.daily_insights_workflow import DailyInsightsWorkflow
    from ..dependencies import (
        get_workflow_service,
        get_search_service,
        get_agent_service,
        get_commit_repository
    )

    return DailyInsightsWorkflow(
        await get_workflow_service(),
        await get_search_service(),
        await get_agent_service(),
        await get_commit_repository()
    )