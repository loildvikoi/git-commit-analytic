# src/infrastructure/rag/sentence_transformer_service.py
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import logging
from ...domain.services.embedding_service import IEmbeddingService

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddingService(IEmbeddingService):
    """Embedding service using Sentence Transformers"""

    def __init__(
            self,
            model_name: str = "all-MiniLM-L6-v2",
            device: str = None,
            cache_folder: str = "./models"
    ):
        """
        Initialize embedding service

        Popular models:
        - all-MiniLM-L6-v2: Fast, good quality (384 dims)
        - all-mpnet-base-v2: Best quality (768 dims)
        - all-MiniLM-L12-v2: Good balance (384 dims)
        """
        self.model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        logger.info(f"Loading embedding model: {model_name} on {self.device}")
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            cache_folder=cache_folder
        )

        # Get model dimensions
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded: {self.embedding_dim} dimensions")

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            # Encode text
            embedding = self.model.encode(
                text,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            # Convert to list
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batch)"""
        try:
            if not texts:
                return []

            # Batch encode
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100
            )

            # Convert to list of lists
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        return {
            "model_name": self.model_name,
            "embedding_dimensions": self.embedding_dim,
            "device": self.device,
            "max_sequence_length": self.model.max_seq_length,
            "provider": "sentence-transformers"
        }

    async def calculate_similarity(
            self,
            embedding1: List[float],
            embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)

            # Ensure in range [0, 1]
            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0