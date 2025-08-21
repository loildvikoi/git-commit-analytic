
# src/domain/services/search_service.py
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..entities.document import Document, DocumentType, SearchScore


class SearchQuery:
    """Domain model for search queries"""

    def __init__(
            self,
            text: str,
            semantic_weight: float = 0.7,
            use_hybrid: bool = True,
            use_reranking: bool = True,
            max_results: int = 10,
            min_score: float = 0.0
    ):
        self.text = text
        self.semantic_weight = semantic_weight  # Weight for semantic vs keyword search
        self.use_hybrid = use_hybrid
        self.use_reranking = use_reranking
        self.max_results = max_results
        self.min_score = min_score

        # Filters
        self.document_types: Optional[List[DocumentType]] = None
        self.projects: Optional[List[str]] = None
        self.authors: Optional[List[str]] = None
        self.date_range: Optional[tuple] = None  # (start_date, end_date)

    def with_filters(self, **kwargs) -> "SearchQuery":
        """Fluent interface for adding filters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class SearchResult:
    """Domain model for search results"""

    def __init__(
            self,
            document: Document,
            score: SearchScore,
            highlights: Optional[List[str]] = None,
            explanation: Optional[str] = None
    ):
        self.document = document
        self.score = score
        self.highlights = highlights or []
        self.explanation = explanation

    @property
    def relevance_score(self) -> float:
        """Get the final relevance score"""
        return self.score.combined_score


class ISearchService(ABC):
    """Service interface for advanced search capabilities"""

    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform search with hybrid approach"""
        pass

    @abstractmethod
    async def semantic_search(
            self,
            query_text: str,
            limit: int = 10,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Pure semantic/vector search"""
        pass

    @abstractmethod
    async def keyword_search(
            self,
            query_text: str,
            limit: int = 10,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Pure keyword/BM25 search"""
        pass

    @abstractmethod
    async def rerank(
            self,
            results: List[SearchResult],
            query: str
    ) -> List[SearchResult]:
        """Rerank search results for better relevance"""
        pass

    @abstractmethod
    async def get_similar_documents(
            self,
            document_id: str,
            limit: int = 5
    ) -> List[SearchResult]:
        """Find similar documents to a given document"""
        pass

