from dataclasses import dataclass
from typing import Dict, Any, List
from .base import DomainEvent


@dataclass
class CommitReceivedEvent(DomainEvent):
    """Event fired when a new commit is received"""
    commit_id: str
    commit_hash: str
    project: str
    author: str
    branch: str

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'commit_id': self.commit_id,
            'commit_hash': self.commit_hash,
            'project': self.project,
            'author': self.author,
            'branch': self.branch
        }


@dataclass
class CommitAnalysisStartedEvent(DomainEvent):
    """Event fired when commit analysis starts"""
    commit_id: str
    model_name: str

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'commit_id': self.commit_id,
            'model_name': self.model_name
        }


@dataclass
class CommitAnalysisCompletedEvent(DomainEvent):
    """Event fired when commit analysis completes"""
    commit_id: str
    analysis_id: str
    summary: str
    tags: List[str]
    processing_time_ms: int

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'commit_id': self.commit_id,
            'analysis_id': self.analysis_id,
            'summary': self.summary,
            'tags': self.tags,
            'processing_time_ms': self.processing_time_ms
        }


@dataclass
class CommitAnalysisFailedEvent(DomainEvent):
    """Event fired when commit analysis fails"""
    commit_id: str
    error_message: str
    retry_count: int

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'commit_id': self.commit_id,
            'error_message': self.error_message,
            'retry_count': self.retry_count
        }
