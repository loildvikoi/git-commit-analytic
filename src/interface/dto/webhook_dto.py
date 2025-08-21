from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class WebhookAuthorDto(BaseModel):
    name: str
    email: str
    username: Optional[str] = None


class WebhookCommitDto(BaseModel):
    id: str
    message: str
    timestamp: datetime
    author: WebhookAuthorDto
    added: List[str] = []
    modified: List[str] = []
    removed: List[str] = []
    url: Optional[str] = None


class WebhookRepositoryDto(BaseModel):
    name: str
    full_name: str
    url: str
    default_branch: str


class WebhookPayloadDto(BaseModel):
    ref: str  # refs/heads/main
    repository: WebhookRepositoryDto
    commits: List[WebhookCommitDto]
    pusher: WebhookAuthorDto
    head_commit: Optional[WebhookCommitDto] = None
