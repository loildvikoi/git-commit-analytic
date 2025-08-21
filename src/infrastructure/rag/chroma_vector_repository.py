# src/infrastructure/rag/chroma_vector_repository.py
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import logging
import os
from ...domain.repositories.vector_repository import IVectorRepository

logger = logging.getLogger(__name__)


class ChromaVectorRepository(IVectorRepository):
    """ChromaDB implementation of vector repository"""

    def __init__(
            self,
            collection_name: str = "git_analytics_docs",
            persist_directory: str = "./data/chroma",
            embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """Initialize ChromaDB vector store"""

        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        # Using default embedding function for now (can be customized)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")

        self.collection_name = collection_name

    async def add_embedding(
            self,
            document_id: str,
            embedding: List[float],
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add document embedding to vector store"""
        try:
            # ChromaDB requires string values in metadata
            clean_metadata = self._clean_metadata(metadata) if metadata else {}

            # Add to collection
            self.collection.add(
                ids=[document_id],
                embeddings=[embedding],
                metadatas=[clean_metadata]
            )

            logger.debug(f"Added embedding for document: {document_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding embedding for {document_id}: {str(e)}")
            return False

    async def search_similar(
            self,
            query_embedding: List[float],
            limit: int = 10,
            filter_metadata: Optional[Dict[str, Any]] = None,
            min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Search for similar documents by embedding"""
        try:
            # Build where clause for filtering
            where_clause = None
            if filter_metadata:
                where_clause = self._build_where_clause(filter_metadata)

            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_clause,
                include=["distances"]
            )

            # Convert results to list of (id, score) tuples
            # ChromaDB returns distances, convert to similarity scores
            similar_docs = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for doc_id, distance in zip(results['ids'][0], results['distances'][0]):
                    # Convert distance to similarity (1 - distance for cosine)
                    similarity = 1.0 - distance
                    if similarity >= min_score:
                        similar_docs.append((doc_id, float(similarity)))

            return similar_docs

        except Exception as e:
            logger.error(f"Error searching similar documents: {str(e)}")
            return []

    async def update_embedding(
            self,
            document_id: str,
            embedding: List[float]
    ) -> bool:
        """Update existing embedding"""
        try:
            # ChromaDB update is actually an upsert
            self.collection.update(
                ids=[document_id],
                embeddings=[embedding]
            )

            logger.debug(f"Updated embedding for document: {document_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating embedding for {document_id}: {str(e)}")
            return False

    async def delete_embedding(self, document_id: str) -> bool:
        """Delete embedding from vector store"""
        try:
            self.collection.delete(ids=[document_id])
            logger.debug(f"Deleted embedding for document: {document_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting embedding for {document_id}: {str(e)}")
            return False

    async def get_embedding(self, document_id: str) -> Optional[List[float]]:
        """Get embedding for a document"""
        try:
            result = self.collection.get(
                ids=[document_id],
                include=["embeddings"]
            )

            if result['embeddings'] and len(result['embeddings']) > 0:
                return result['embeddings'][0]

            return None

        except Exception as e:
            logger.error(f"Error getting embedding for {document_id}: {str(e)}")
            return None

    async def bulk_add_embeddings(
            self,
            embeddings: List[Tuple[str, List[float], Dict[str, Any]]]
    ) -> int:
        """Bulk add embeddings"""
        if not embeddings:
            return 0

        try:
            ids = []
            embeds = []
            metadatas = []

            for doc_id, embedding, metadata in embeddings:
                ids.append(doc_id)
                embeds.append(embedding)
                metadatas.append(self._clean_metadata(metadata))

            # Add in batches if necessary
            batch_size = 1000
            added_count = 0

            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_embeds = embeds[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size]

                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeds,
                    metadatas=batch_metas
                )

                added_count += len(batch_ids)
                logger.info(f"Added batch of {len(batch_ids)} embeddings")

            return added_count

        except Exception as e:
            logger.error(f"Error in bulk add embeddings: {str(e)}")
            return 0

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata for ChromaDB (ensure all values are strings, numbers, or bools)"""
        clean = {}
        for key, value in metadata.items():
            if value is None:
                clean[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, list):
                # Convert list to comma-separated string
                clean[key] = ",".join(str(v) for v in value)
            else:
                clean[key] = str(value)
        return clean

    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters"""
        where = {}

        for key, value in filters.items():
            if value is not None:
                if isinstance(value, list):
                    # Use $in operator for lists
                    where[key] = {"$in": value}
                else:
                    where[key] = value

        return where if where else None

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_function": str(self.embedding_function)
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "error": str(e)
            }