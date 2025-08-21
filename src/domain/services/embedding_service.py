# src/domain/services/embedding_service.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..entities.document import Document


class IEmbeddingService(ABC):
    """Service interface for generating embeddings"""

    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        pass

    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batch)"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        pass

    @abstractmethod
    async def calculate_similarity(
            self,
            embedding1: List[float],
            embedding2: List[float]
    ) -> float:
        """Calculate similarity between two embeddings"""
        pass

