import logging
from typing import List, Optional
from ...domain.entities.document import Document, DocumentType, DocumentMetadata
from ...domain.repositories.document_repository import IDocumentRepository
from ...domain.repositories.vector_repository import IVectorRepository
from ...domain.services.embedding_service import IEmbeddingService
from ...domain.services.event_dispatcher import IEventDispatcher

logger = logging.getLogger(__name__)


class IndexDocumentUseCase:
    """Use case for indexing documents into the RAG system"""

    def __init__(
            self,
            document_repository: IDocumentRepository,
            vector_repository: IVectorRepository,
            embedding_service: IEmbeddingService,
            event_dispatcher: IEventDispatcher
    ):
        self.document_repository = document_repository
        self.vector_repository = vector_repository
        self.embedding_service = embedding_service
        self.event_dispatcher = event_dispatcher

    async def execute(
            self,
            content: str,
            document_type: DocumentType,
            source_type: str,
            project: Optional[str] = None,
            author: Optional[str] = None,
            title: Optional[str] = None,
            chunk_size: int = 500,
            chunk_overlap: int = 50
    ) -> List[Document]:
        """Index a document with chunking and embedding"""

        # Create metadata
        from datetime import datetime
        metadata = DocumentMetadata(
            source_type=source_type,
            indexed_at=datetime.now(),
            version=1
        )

        # Create chunks if content is large
        chunks = self._create_chunks(content, chunk_size, chunk_overlap)

        documents = []
        parent_id = None

        for i, chunk_content in enumerate(chunks):
            # Create document
            doc = Document(
                content=chunk_content,
                document_type=document_type,
                metadata=metadata,
                project=project,
                author=author,
                title=f"{title} - Part {i + 1}" if len(chunks) > 1 and title else title,
                parent_id=parent_id if i > 0 else None,
                chunk_index=i
            )

            # Set parent_id for subsequent chunks
            if i == 0 and len(chunks) > 1:
                parent_id = doc.id

            # Save document
            saved_doc = await self.document_repository.save(doc)

            # Generate embedding
            embedding = await self.embedding_service.generate_embedding(
                saved_doc.searchable_content
            )

            # Save embedding to vector store
            await self.vector_repository.add_embedding(
                document_id=saved_doc.id,
                embedding=embedding,
                metadata={
                    "document_type": document_type.value,
                    "project": project,
                    "author": author,
                    "chunk_index": i
                }
            )

            # Update document with embedding flag
            saved_doc.embedding = embedding
            await self.document_repository.update(saved_doc)

            documents.append(saved_doc)

            logger.info(f"Indexed document chunk {i + 1}/{len(chunks)}: {saved_doc.id}")

        # Dispatch event
        from ..events.document_events import DocumentIndexedEvent
        event = DocumentIndexedEvent(
            document_ids=[doc.id for doc in documents],
            document_type=document_type.value,
            project=project,
            chunks_count=len(documents)
        )
        await self.event_dispatcher.dispatch(event)

        return documents

    def _create_chunks(
            self,
            content: str,
            chunk_size: int,
            chunk_overlap: int
    ) -> List[str]:
        """Split content into overlapping chunks"""
        if len(content) <= chunk_size:
            return [content]

        chunks = []
        start = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk = content[start:end]
            chunks.append(chunk)

            # Move start position with overlap
            start += chunk_size - chunk_overlap

            # Avoid tiny last chunk
            if len(content) - start < chunk_overlap:
                break

        return chunks

