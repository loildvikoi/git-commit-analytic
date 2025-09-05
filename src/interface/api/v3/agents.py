# src/interface/api/v1/agents.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List
import logging
import time

from src.interface.dto.agent_dto import (
    CreateAgentDto,
    AgentTaskDto,
    CodeReviewRequestDto,
    MultiAgentAnalysisDto,
    ProactiveInsightsRequestDto,
    AgentResponseDto,
    WorkflowResponseDto
)
from src.interface.api.dependencies import (
    get_code_review_use_case,
    get_multi_agent_analysis_use_case,
    get_proactive_insights_use_case,
    get_agent_service
)

router = APIRouter(prefix="/agents", tags=["agents"])
logger = logging.getLogger(__name__)


@router.post("/create", response_model=AgentResponseDto)
async def create_agent(
        request: CreateAgentDto,
        agent_service=Depends(get_agent_service)
):
    """Create a new AI agent"""
    try:
        from ....domain.entities.agent import AgentRole

        agent = await agent_service.create_agent(
            name=request.name,
            role=AgentRole[request.role.value.upper()],
            model=request.model
        )

        return AgentResponseDto(
            agent_id=agent.id,
            agent_name=agent.name,
            role=agent.role.value,
            task_executed="creation",
            result={"status": "created"},
            confidence=1.0,
            execution_time_ms=0,
            tools_used=[],
            decisions_made=[]
        )

    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute-task", response_model=AgentResponseDto)
async def execute_agent_task(
        request: AgentTaskDto,
        agent_service=Depends(get_agent_service)
):
    """Execute a task with an agent"""
    try:
        start_time = time.time()

        # Get agent
        from ..dependencies import get_agent_repository
        agent_repo = await get_agent_repository()
        agent = await agent_repo.find_by_id(request.agent_id)

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Execute task
        result = await agent_service.execute_task(
            agent=agent,
            task=request.task,
            context=request.context
        )

        execution_time = int((time.time() - start_time) * 1000)

        return AgentResponseDto(
            agent_id=agent.id,
            agent_name=agent.name,
            role=agent.role.value,
            task_executed=request.task,
            result=result,
            confidence=result.get("confidence", 0.8),
            execution_time_ms=execution_time,
            tools_used=agent.tools_used,
            decisions_made=[d.__dict__ for d in agent.decisions]
        )

    except Exception as e:
        logger.error(f"Error executing task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/code-review")
async def review_code(
        request: CodeReviewRequestDto,
        background_tasks: BackgroundTasks,
        use_case=Depends(get_code_review_use_case)
):
    """Autonomous code review by AI agent"""
    try:
        # Execute review (could be async in background)
        result = await use_case.execute(
            commit_id=request.commit_id,
            review_depth=request.review_depth
        )

        return {
            "status": "success",
            "data": result
        }

    except Exception as e:
        logger.error(f"Error in code review: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi-agent-analysis")
async def multi_agent_analysis(
        request: MultiAgentAnalysisDto,
        use_case=Depends(get_multi_agent_analysis_use_case)
):
    """Run multi-agent collaborative analysis"""
    try:
        result = await use_case.execute(
            project=request.project,
            analysis_type=request.analysis_type
        )

        return {
            "status": "success",
            "data": result
        }

    except Exception as e:
        logger.error(f"Error in multi-agent analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/proactive-insights")
async def generate_proactive_insights(
        request: ProactiveInsightsRequestDto,
        use_case=Depends(get_proactive_insights_use_case)
):
    """Generate proactive insights using AI agents"""
    try:
        result = await use_case.execute(
            project=request.project,
            lookback_days=request.lookback_days
        )

        return {
            "status": "success",
            "data": result
        }

    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_agents(
        agent_service=Depends(get_agent_service)
):
    """List all available agents"""
    try:
        from ..dependencies import get_agent_repository
        agent_repo = await get_agent_repository()
        agents = await agent_repo.list_active_agents()

        return {
            "status": "success",
            "agents": [agent.to_dict() for agent in agents]
        }

    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def agents_health_check():
    """Check agents system health"""
    try:
        health_status = {
            "agent_service": "healthy",
            "workflow_service": "healthy",
            "tool_service": "healthy",
            "ollama_connection": "healthy"
        }

        # Check Ollama connection
        import httpx
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code != 200:
                    health_status["ollama_connection"] = "unhealthy"
            except:
                health_status["ollama_connection"] = "unreachable"

        all_healthy = all(status == "healthy" for status in health_status.values())

        return {
            "status": "healthy" if all_healthy else "degraded",
            "components": health_status
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
