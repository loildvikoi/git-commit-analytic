# src/infrastructure/rag/hybrid_search_service.py
from typing import List, Optional, Dict, Any
from rank_bm25 import BM25Okapi
import numpy as np
import logging
from ...domain.services.search_service import (
    ISearchService, SearchQuery, SearchResult
)
from ...domain.entities.document import Document, SearchScore
from ...domain.repositories.document_repository import IDocumentRepository
from ...domain.repositories.vector_repository import IVectorRepository
from ...domain.services.embedding_service import IEmbeddingService

logger = logging.getLogger(__name__)


class HybridSearchService(ISearchService):
    """Hybrid search service combining semantic and keyword search"""

    def __init__(
            self,
            document_repository: IDocumentRepository,
            vector_repository: IVectorRepository,
            embedding_service: IEmbeddingService
    ):
        self.document_repository = document_repository
        self.vector_repository = vector_repository
        self.embedding_service = embedding_service

        # BM25 index cache
        self.bm25_index = None
        self.indexed_documents = []
        self.last_index_update = None

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform hybrid search"""

        if query.use_hybrid:
            # Perform both semantic and keyword search
            semantic_results = await self.semantic_search(
                query.text,
                limit=query.max_results * 2,  # Get more for merging
                filters=self._build_filters(query)
            )

            keyword_results = await self.keyword_search(
                query.text,
                limit=query.max_results * 2,
                filters=self._build_filters(query)
            )

            # Merge results
            merged_results = self._merge_results(
                semantic_results,
                keyword_results,
                query.semantic_weight
            )

        else:
            # Pure semantic search
            merged_results = await self.semantic_search(
                query.text,
                limit=query.max_results,
                filters=self._build_filters(query)
            )

        # Apply reranking if requested
        if query.use_reranking:
            merged_results = await self.rerank(merged_results, query.text)

        # Filter by minimum score and limit
        final_results = [
                            r for r in merged_results
                            if r.score.combined_score >= query.min_score
                        ][:query.max_results]

        return final_results

    async def semantic_search(
            self,
            query_text: str,
            limit: int = 10,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Pure semantic/vector search"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(query_text)

            # Search in vector store
            similar_docs = await self.vector_repository.search_similar(
                query_embedding=query_embedding,
                limit=limit,
                filter_metadata=filters,
                min_score=0.0
            )

            # Fetch full documents
            results = []
            for doc_id, similarity_score in similar_docs:
                document = await self.document_repository.find_by_id(doc_id)
                if document:
                    score = SearchScore(
                        semantic_score=similarity_score,
                        keyword_score=0.0,
                        combined_score=similarity_score,
                        confidence=0.8
                    )

                    result = SearchResult(
                        document=document,
                        score=score,
                        highlights=self._extract_highlights(document.content, query_text),
                        explanation=f"Semantic similarity: {similarity_score:.3f}"
                    )
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []

    async def keyword_search(
            self,
            query_text: str,
            limit: int = 10,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Pure keyword/BM25 search"""
        try:
            # Get documents from repository
            all_documents = await self.document_repository.search(
                query="",  # Get all for BM25
                limit=1000  # Reasonable limit
            )

            # Apply filters
            if filters:
                filtered_docs = self._apply_filters(all_documents, filters)
            else:
                filtered_docs = all_documents

            if not filtered_docs:
                return []

            # Build or update BM25 index
            await self._update_bm25_index(filtered_docs)

            # Tokenize query
            query_tokens = query_text.lower().split()

            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)

            # Get top documents
            top_indices = np.argsort(scores)[::-1][:limit]

            results = []
            for idx in top_indices:
                if idx < len(self.indexed_documents) and scores[idx] > 0:
                    document = self.indexed_documents[idx]

                    # Normalize BM25 score to 0-1 range
                    normalized_score = scores[idx] / (scores[idx] + 1)

                    score = SearchScore(
                        semantic_score=0.0,
                        keyword_score=normalized_score,
                        combined_score=normalized_score,
                        confidence=0.7
                    )

                    result = SearchResult(
                        document=document,
                        score=score,
                        highlights=self._extract_highlights(document.content, query_text),
                        explanation=f"BM25 score: {normalized_score:.3f}"
                    )
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            return []

    async def rerank(
            self,
            results: List[SearchResult],
            query: str
    ) -> List[SearchResult]:
        """Rerank search results for better relevance"""
        try:
            from datetime import datetime, timedelta

            for result in results:
                doc = result.document
                boost = 1.0

                # Boost recent documents
                if doc.created_at:
                    age_days = (datetime.now() - doc.created_at).days
                    if age_days < 7:
                        boost *= 1.3  # Very recent
                    elif age_days < 30:
                        boost *= 1.1  # Recent
                    elif age_days > 365:
                        boost *= 0.9  # Old

                # Boost if query terms in title
                if doc.title and query.lower() in doc.title.lower():
                    boost *= 1.5

                # Boost if query terms in summary
                if doc.summary and query.lower() in doc.summary.lower():
                    boost *= 1.2

                # Boost based on document type (commits are often most relevant)
                if doc.document_type.value == "commit":
                    boost *= 1.1

                # Apply boost to combined score
                result.score.combined_score = min(1.0, result.score.combined_score * boost)

            # Re-sort by new scores
            results.sort(key=lambda x: x.score.combined_score, reverse=True)

            return results

        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            return results

    async def get_similar_documents(
            self,
            document_id: str,
            limit: int = 5
    ) -> List[SearchResult]:
        """Find similar documents to a given document"""
        try:
            # Get the document
            document = await self.document_repository.find_by_id(document_id)
            if not document:
                return []

            # Get its embedding
            embedding = await self.vector_repository.get_embedding(document_id)
            if not embedding:
                # Generate embedding if not found
                embedding = await self.embedding_service.generate_embedding(
                    document.searchable_content
                )

            # Search for similar documents
            similar_docs = await self.vector_repository.search_similar(
                query_embedding=embedding,
                limit=limit + 1,  # +1 to exclude self
                min_score=0.3
            )

            # Convert to results, excluding the original document
            results = []
            for doc_id, similarity_score in similar_docs:
                if doc_id != document_id:
                    similar_doc = await self.document_repository.find_by_id(doc_id)
                    if similar_doc:
                        score = SearchScore(
                            semantic_score=similarity_score,
                            keyword_score=0.0,
                            combined_score=similarity_score,
                            confidence=0.9
                        )

                        result = SearchResult(
                            document=similar_doc,
                            score=score,
                            highlights=[],
                            explanation=f"Similarity to {document.title or document_id[:8]}: {similarity_score:.3f}"
                        )
                        results.append(result)

            return results[:limit]

        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            return []

    def _merge_results(
            self,
            semantic_results: List[SearchResult],
            keyword_results: List[SearchResult],
            semantic_weight: float
    ) -> List[SearchResult]:
        """Merge semantic and keyword search results"""

        # Create a dictionary to track scores
        merged_scores = {}

        # Add semantic results
        for result in semantic_results:
            doc_id = result.document.id
            merged_scores[doc_id] = {
                'document': result.document,
                'semantic_score': result.score.semantic_score,
                'keyword_score': 0.0,
                'highlights': result.highlights
            }

        # Add/update with keyword results
        for result in keyword_results:
            doc_id = result.document.id
            if doc_id in merged_scores:
                merged_scores[doc_id]['keyword_score'] = result.score.keyword_score
            else:
                merged_scores[doc_id] = {
                    'document': result.document,
                    'semantic_score': 0.0,
                    'keyword_score': result.score.keyword_score,
                    'highlights': result.highlights
                }

        # Calculate combined scores and create results
        results = []
        for doc_id, scores in merged_scores.items():
            combined = (
                    semantic_weight * scores['semantic_score'] +
                    (1 - semantic_weight) * scores['keyword_score']
            )

            score = SearchScore(
                semantic_score=scores['semantic_score'],
                keyword_score=scores['keyword_score'],
                combined_score=combined,
                confidence=0.8 if scores['semantic_score'] > 0 and scores['keyword_score'] > 0 else 0.6
            )

            result = SearchResult(
                document=scores['document'],
                score=score,
                highlights=scores['highlights'],
                explanation=f"Hybrid search (S:{scores['semantic_score']:.2f}, K:{scores['keyword_score']:.2f})"
            )
            results.append(result)

        # Sort by combined score
        results.sort(key=lambda x: x.score.combined_score, reverse=True)

        return results

    async def _update_bm25_index(self, documents: List[Document]):
        """Update BM25 index with documents"""

        # Tokenize documents
        tokenized_docs = []
        for doc in documents:
            text = doc.searchable_content.lower()
            tokens = text.split()  # Simple tokenization
            tokenized_docs.append(tokens)

        # Build BM25 index
        self.bm25_index = BM25Okapi(tokenized_docs)
        self.indexed_documents = documents

        logger.debug(f"Updated BM25 index with {len(documents)} documents")

    def _build_filters(self, query: SearchQuery) -> Dict[str, Any]:
        """Build filters from search query"""
        filters = {}

        if query.document_types:
            filters['document_type'] = [dt.value for dt in query.document_types]
        if query.projects:
            filters['project'] = query.projects
        if query.authors:
            filters['author'] = query.authors

        return filters if filters else None

    def _apply_filters(
            self,
            documents: List[Document],
            filters: Dict[str, Any]
    ) -> List[Document]:
        """Apply filters to document list"""
        filtered = documents

        if 'document_type' in filters:
            allowed_types = filters['document_type']
            filtered = [d for d in filtered if d.document_type.value in allowed_types]

        if 'project' in filters:
            allowed_projects = filters['project']
            filtered = [d for d in filtered if d.project in allowed_projects]

        if 'author' in filters:
            allowed_authors = filters['author']
            filtered = [d for d in filtered if d.author in allowed_authors]

        return filtered

    def _extract_highlights(
            self,
            content: str,
            query: str,
            context_size: int = 50
    ) -> List[str]:
        """Extract relevant highlights from content"""
        highlights = []
        query_terms = query.lower().split()
        content_lower = content.lower()

        for term in query_terms:
            if term in content_lower:
                idx = content_lower.index(term)
                start = max(0, idx - context_size)
                end = min(len(content), idx + len(term) + context_size)

                highlight = content[start:end]
                if start > 0:
                    highlight = "..." + highlight
                if end < len(content):
                    highlight = highlight + "..."

                highlights.append(highlight)

        return highlights[:3]  # Return top 3 highlights