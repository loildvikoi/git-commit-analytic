
# src/domain/services/rag_service.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..entities.document import Document


class RAGContext:
    """Context for RAG queries"""

    def __init__(
            self,
            question: str,
            documents: List[Document],
            max_context_length: int = 2000,
            include_metadata: bool = True
    ):
        self.question = question
        self.documents = documents
        self.max_context_length = max_context_length
        self.include_metadata = include_metadata

    def build_context_text(self) -> str:
        """Build context text from documents"""
        context_parts = []
        current_length = 0

        for doc in self.documents:
            doc_text = doc.searchable_content

            if self.include_metadata:
                meta_parts = []
                if doc.project:
                    meta_parts.append(f"Project: {doc.project}")
                if doc.author:
                    meta_parts.append(f"Author: {doc.author}")
                if meta_parts:
                    doc_text = f"[{', '.join(meta_parts)}]\n{doc_text}"

            if current_length + len(doc_text) > self.max_context_length:
                # Truncate if needed
                remaining = self.max_context_length - current_length
                if remaining > 100:  # Only add if meaningful
                    context_parts.append(doc_text[:remaining] + "...")
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n\n---\n\n".join(context_parts)


class IRAGService(ABC):
    """Service interface for RAG operations"""

    @abstractmethod
    async def answer_question(
            self,
            question: str,
            context_documents: Optional[List[Document]] = None,
            search_first: bool = True,
            max_documents: int = 5
    ) -> Dict[str, Any]:
        """Answer a question using RAG approach
        Returns dict with 'answer', 'sources', 'confidence', etc."""
        pass

    @abstractmethod
    async def generate_context(
            self,
            question: str,
            max_documents: int = 5
    ) -> RAGContext:
        """Generate context for a question by finding relevant documents"""
        pass

    @abstractmethod
    async def augment_query(self, query: str) -> List[str]:
        """Augment query with synonyms, related terms, etc."""
        pass

    @abstractmethod
    async def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        pass