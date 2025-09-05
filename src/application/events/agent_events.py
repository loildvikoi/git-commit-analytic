# src/application/events/agent_events.py
from typing import Dict, Any, List
from src.domain.events.base import DomainEvent


class AgentAnalysisRequestedEvent(DomainEvent):
    """Event when agent analysis is requested"""

    def __init__(
            self,
            commit_id: str,
            document_id: str,
            analysis_types: List[str]
    ):
        super().__init__()
        self.commit_id = commit_id
        self.document_id = document_id
        self.analysis_types = analysis_types

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'commit_id': self.commit_id,
            'document_id': self.document_id,
            'analysis_types': self.analysis_types
        }


class CriticalBugDetectedEvent(DomainEvent):
    """Event when critical bug is detected"""

    def __init__(
            self,
            commit_id: str,
            bug_analysis: str,
            severity: str
    ):
        super().__init__()
        self.commit_id = commit_id
        self.bug_analysis = bug_analysis
        self.severity = severity

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'commit_id': self.commit_id,
            'bug_analysis': self.bug_analysis,
            'severity': self.severity
        }
