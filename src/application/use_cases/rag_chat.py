from typing import Dict, Any, Optional, List
from ...domain.services.rag_service import IRAGService
from ...domain.services.search_service import ISearchService
from ...domain.services.cache_service import ICacheService
from ...domain.repositories.document_repository import IDocumentRepository
import logging
import hashlib

logger = logging.getLogger(__name__)


class RAGChatUseCase:
    """Use case for RAG-based chat"""

    def __init__(
            self,
            rag_service: IRAGService,
            search_service: ISearchService,
            document_repository: IDocumentRepository,
            cache_service: ICacheService
    ):
        self.rag_service = rag_service
        self.search_service = search_service
        self.document_repository = document_repository
        self.cache_service = cache_service

    async def execute(
            self,
            question: str,
            context_project: Optional[str] = None,
            context_author: Optional[str] = None,
            max_documents: int = 5,
            use_cache: bool = True
    ) -> Dict[str, Any]:
        """Execute RAG chat query"""

        # Check cache
        if use_cache:
            cache_key = self._generate_cache_key(question, context_project, context_author)
            cached = await self.cache_service.get(cache_key)
            if cached:
                logger.info(f"Cache hit for RAG chat: {question[:50]}...")
                cached["cached"] = True
                return cached

        # Search for relevant documents first
        from ...domain.services.search_service import SearchQuery
        search_query = SearchQuery(
            text=question,
            use_hybrid=True,
            max_results=max_documents * 2  # Get more for better context
        )

        # Add filters if provided
        if context_project:
            search_query.projects = [context_project]
        if context_author:
            search_query.authors = [context_author]

        # Search for relevant documents
        search_results = await self.search_service.search(search_query)

        # Extract documents from search results
        context_documents = [result.document for result in search_results[:max_documents]]

        # Get answer from RAG service
        result = await self.rag_service.answer_question(
            question=question,
            context_documents=context_documents,
            search_first=False  # We already searched
        )

        # Enhance result with metadata
        enhanced_result = {
            "question": question,
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", 0.0),
            "sources": [
                {
                    "document_id": doc.id,
                    "title": doc.title,
                    "project": doc.project,
                    "author": doc.author,
                    "type": doc.document_type.value,
                    "relevance_score": next(
                        (r.score.combined_score for r in search_results if r.document.id == doc.id),
                        0.0
                    )
                }
                for doc in context_documents
            ],
            "context_used": len(context_documents),
            "method": "rag_hybrid_search",
            "cached": False
        }

        # Cache result
        if use_cache:
            await self.cache_service.set(cache_key, enhanced_result, ttl=600)

        return enhanced_result

    def _generate_cache_key(
            self,
            question: str,
            project: Optional[str],
            author: Optional[str]
    ) -> str:
        """Generate cache key"""
        key_str = f"{question}:{project or ''}:{author or ''}"
        return f"rag:chat:{hashlib.md5(key_str.encode()).hexdigest()}"

