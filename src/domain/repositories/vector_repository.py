

# src/domain/repositories/vector_repository.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from ..entities.document import Document, SearchScore


class IVectorRepository(ABC):
    """Repository interface for vector operations"""

    @abstractmethod
    async def add_embedding(
            self,
            document_id: str,
            embedding: List[float],
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add document embedding to vector store"""
        pass

    @abstractmethod
    async def search_similar(
            self,
            query_embedding: List[float],
            limit: int = 10,
            filter_metadata: Optional[Dict[str, Any]] = None,
            min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Search for similar documents by embedding
        Returns list of (document_id, similarity_score) tuples"""
        pass

    @abstractmethod
    async def update_embedding(
            self,
            document_id: str,
            embedding: List[float]
    ) -> bool:
        """Update existing embedding"""
        pass

    @abstractmethod
    async def delete_embedding(self, document_id: str) -> bool:
        """Delete embedding from vector store"""
        pass

    @abstractmethod
    async def get_embedding(self, document_id: str) -> Optional[List[float]]:
        """Get embedding for a document"""
        pass

    @abstractmethod
    async def bulk_add_embeddings(
            self,
            embeddings: List[Tuple[str, List[float], Dict[str, Any]]]
    ) -> int:
        """Bulk add embeddings
        Input: List of (document_id, embedding, metadata) tuples
        Returns: Number of successfully added embeddings"""
        pass