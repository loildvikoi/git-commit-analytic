from typing import Dict, Any, List
from src.domain.events.base import DomainEvent


class CommitReceivedEvent(DomainEvent):
    """Event fired when a new commit is received"""

    def __init__(self, commit_id: str, commit_hash: str, project: str, author: str, branch: str):
        super().__init__()  # Gọi parent __init__ để set event_id và occurred_at
        self.commit_id = commit_id
        self.commit_hash = commit_hash
        self.project = project
        self.author = author
        self.branch = branch

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'commit_id': self.commit_id,
            'commit_hash': self.commit_hash,
            'project': self.project,
            'author': self.author,
            'branch': self.branch
        }


class CommitAnalysisStartedEvent(DomainEvent):
    """Event fired when commit analysis starts"""

    def __init__(self, commit_id: str, model_name: str):
        super().__init__()
        self.commit_id = commit_id
        self.model_name = model_name

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'commit_id': self.commit_id,
            'model_name': self.model_name
        }


class CommitAnalysisCompletedEvent(DomainEvent):
    """Event fired when commit analysis completes"""

    def __init__(self, commit_id: str, analysis_id: str, summary: str, tags: List[str], processing_time_ms: int):
        super().__init__()
        self.commit_id = commit_id
        self.analysis_id = analysis_id
        self.summary = summary
        self.tags = tags
        self.processing_time_ms = processing_time_ms

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'commit_id': self.commit_id,
            'analysis_id': self.analysis_id,
            'summary': self.summary,
            'tags': self.tags,
            'processing_time_ms': self.processing_time_ms
        }


class CommitAnalysisFailedEvent(DomainEvent):
    """Event fired when commit analysis fails"""

    def __init__(self, commit_id: str, error_message: str, retry_count: int):
        super().__init__()
        self.commit_id = commit_id
        self.error_message = error_message
        self.retry_count = retry_count

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'commit_id': self.commit_id,
            'error_message': self.error_message,
            'retry_count': self.retry_count
        }
