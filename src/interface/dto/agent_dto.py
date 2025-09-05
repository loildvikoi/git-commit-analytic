# src/interface/dto/agent_dto.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class AgentRoleDto(str, Enum):
    CODE_REVIEWER = "code_reviewer"
    BUG_DETECTOR = "bug_detector"
    PERFORMANCE_ANALYZER = "performance_analyzer"
    SECURITY_SCANNER = "security_scanner"
    DOCUMENTATION_WRITER = "documentation_writer"
    GENERAL_ASSISTANT = "general_assistant"


class CreateAgentDto(BaseModel):
    """DTO for creating an agent"""
    name: str = Field(..., min_length=1, max_length=100)
    role: AgentRoleDto
    model: str = Field("llama3.2:1b", description="Ollama model to use")
    temperature: float = Field(0.3, ge=0, le=1)
    max_iterations: int = Field(5, ge=1, le=10)


class AgentTaskDto(BaseModel):
    """DTO for agent task execution"""
    agent_id: str = Field(..., description="Agent ID to execute task")
    task: str = Field(..., min_length=1, max_length=1000)
    context: Optional[Dict[str, Any]] = Field(None, description="Task context")
    timeout_seconds: int = Field(30, ge=1, le=300)


class CodeReviewRequestDto(BaseModel):
    """DTO for code review request"""
    commit_id: str = Field(..., description="Commit ID to review")
    review_depth: str = Field("standard", pattern="^(quick|standard|thorough)$")
    focus_areas: Optional[List[str]] = Field(None, description="Specific areas to focus on")


class MultiAgentAnalysisDto(BaseModel):
    """DTO for multi-agent analysis"""
    project: str = Field(..., description="Project to analyze")
    analysis_type: str = Field("comprehensive", pattern="^(quick|comprehensive|security|performance)$")
    lookback_days: int = Field(7, ge=1, le=90)
    include_predictions: bool = Field(True, description="Include predictive insights")


class ProactiveInsightsRequestDto(BaseModel):
    """DTO for proactive insights"""
    project: Optional[str] = Field(None, description="Specific project or all")
    lookback_days: int = Field(7, ge=1, le=30)
    insight_types: List[str] = Field(
        ["patterns", "risks", "recommendations"],
        description="Types of insights to generate"
    )


class AgentResponseDto(BaseModel):
    """Response DTO for agent operations"""
    agent_id: str
    agent_name: str
    role: str
    task_executed: str
    result: Dict[str, Any]
    confidence: float
    execution_time_ms: int
    tools_used: List[str]
    decisions_made: List[Dict[str, Any]]


class WorkflowResponseDto(BaseModel):
    """Response DTO for workflow execution"""
    workflow_id: str
    workflow_name: str
    status: str
    steps_completed: int
    total_steps: int
    results: Dict[str, Any]
    errors: List[str]
    execution_time_ms: int
