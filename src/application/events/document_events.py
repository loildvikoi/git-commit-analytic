from typing import Dict, Any, List, Optional
from src.domain.events.base import DomainEvent


class DocumentIndexedEvent(DomainEvent):
    """Event fired when documents are indexed"""

    def __init__(
            self,
            document_ids: List[str],
            document_type: str,
            project: Optional[str] = None,
            chunks_count: int = 1
    ):
        super().__init__()
        self.document_ids = document_ids
        self.document_type = document_type
        self.project = project
        self.chunks_count = chunks_count

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'document_ids': self.document_ids,
            'document_type': self.document_type,
            'project': self.project,
            'chunks_count': self.chunks_count
        }


class DocumentSearchedEvent(DomainEvent):
    """Event fired when documents are searched"""

    def __init__(
            self,
            query: str,
            results_count: int,
            search_type: str,  # hybrid, semantic, keyword
            cached: bool = False
    ):
        super().__init__()
        self.query = query
        self.results_count = results_count
        self.search_type = search_type
        self.cached = cached

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'results_count': self.results_count,
            'search_type': self.search_type,
            'cached': self.cached
        }


class RAGQueryProcessedEvent(DomainEvent):
    """Event fired when RAG query is processed"""

    def __init__(
            self,
            question: str,
            documents_used: int,
            confidence: float,
            response_time_ms: int
    ):
        super().__init__()
        self.question = question
        self.documents_used = documents_used
        self.confidence = confidence
        self.response_time_ms = response_time_ms

    def _get_event_data(self) -> Dict[str, Any]:
        return {
            'question': self.question,
            'documents_used': self.documents_used,
            'confidence': self.confidence,
            'response_time_ms': self.response_time_ms
        }

