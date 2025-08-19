from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import List, Optional


class FileChangeDto(BaseModel):
    filename: str = Field(..., min_length=1, description="Name of the file changed")
    additions: int = Field(..., ge=0, description="Number of additions")
    deletions: int = Field(..., ge=0, description="Number of deletions")
    status: str = Field(..., description="Commit status")


class CommitDto(BaseModel):
    commit_hash: str = Field(..., min_length=6, description="Commit hash")
    author_email: str = Field(..., description="Author's email address")
    author_name: str = Field(..., min_length=1, description="Author's name")
    message: str = Field(..., min_length=1, description="Commit message")
    timestamp: datetime
    branch: str = Field(..., min_length=1, description="Branch")
    project: str = Field(..., min_length=1, description="Project")
    files_changed: List[FileChangeDto] = []
    issue_numbers: List[str] = []


class CommitResponseDto(BaseModel):
    id: str
    commit_hash: str
    author_email: str
    author_name: str
    message: str
    timestamp: datetime
    branch: str
    project: str
    files_count: int
    total_lines_changed: int
    complexity_score: Optional[float] = None
    summary: Optional[str] = None
    tags: List[str] = []
    sentiment_score: Optional[float] = None
    analyzed_at: Optional[datetime] = None
    created_at: datetime


class CommitSearchDto(BaseModel):
    query: Optional[str] = None
    project: Optional[str] = None
    author: Optional[str] = None
    branch: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    tags: List[str] = []
    limit: int = Field(50, ge=1, le=200, description="Number of results to return per page")
    offset: int = Field(0, ge=0, description="Pagination offset")

class ChatRequestDto(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Question to ask about the commits")
    context_project: Optional[str] = Field(None, description="Project context for the question")
    context_author: Optional[str] = Field(None, description="Author context for the question")
    context_days: int = Field(30, ge=1, le=365, description="Number of days to consider for context")


class ChatResponseDto(BaseModel):
    answer: str
    context_commits_count: int
    processing_time_ms: int
    model_used: str
