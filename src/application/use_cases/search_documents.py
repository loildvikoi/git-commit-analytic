from typing import List, Optional, Dict, Any
from ...domain.services.search_service import ISearchService, SearchQuery, SearchResult
from ...domain.services.cache_service import ICacheService
from ...domain.entities.document import DocumentType
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class SearchDocumentsUseCase:
    """Use case for searching documents with hybrid approach"""

    def __init__(
            self,
            search_service: ISearchService,
            cache_service: ICacheService
    ):
        self.search_service = search_service
        self.cache_service = cache_service

    async def execute(
            self,
            query_text: str,
            document_types: Optional[List[str]] = None,
            projects: Optional[List[str]] = None,
            authors: Optional[List[str]] = None,
            use_hybrid: bool = True,
            use_cache: bool = True,
            max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Execute document search"""

        # Create search query
        search_query = SearchQuery(
            text=query_text,
            use_hybrid=use_hybrid,
            max_results=max_results
        )

        # Add filters
        if document_types:
            search_query.document_types = [
                DocumentType[dt.upper()] for dt in document_types
            ]
        if projects:
            search_query.projects = projects
        if authors:
            search_query.authors = authors

        # Check cache
        if use_cache:
            cache_key = self._generate_cache_key(search_query)
            cached = await self.cache_service.get(cache_key)
            if cached:
                logger.info(f"Cache hit for search: {query_text[:50]}...")
                return cached

        # Perform search
        results = await self.search_service.search(search_query)

        # Convert to response format
        response = []
        for result in results:
            response.append({
                "document": result.document.to_dict(),
                "score": {
                    "semantic": result.score.semantic_score,
                    "keyword": result.score.keyword_score,
                    "combined": result.score.combined_score,
                    "confidence": result.score.confidence
                },
                "highlights": result.highlights,
                "explanation": result.explanation
            })

        # Cache results
        if use_cache and response:
            await self.cache_service.set(cache_key, response, ttl=300)

        return response

    def _generate_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for search query"""
        key_data = {
            "text": query.text,
            "types": [dt.value for dt in query.document_types] if query.document_types else None,
            "projects": query.projects,
            "authors": query.authors,
            "hybrid": query.use_hybrid
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return f"search:docs:{hashlib.md5(key_str.encode()).hexdigest()}"

