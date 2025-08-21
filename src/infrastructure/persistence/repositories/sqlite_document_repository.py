from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc
from ....domain.repositories.document_repository import IDocumentRepository
from ....domain.entities.document import Document, DocumentType, DocumentMetadata
from ..models.document_model import DocumentModel, DocumentTypeEnum
import logging

logger = logging.getLogger(__name__)


class SqliteDocumentRepository(IDocumentRepository):
    """SQLite implementation of document repository"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, document: Document) -> Document:
        """Save a document"""
        try:
            db_document = DocumentModel(
                id=document.id,
                content=document.content,
                document_type=DocumentTypeEnum[document.document_type.name],
                source_type=document.metadata.source_type,
                source_url=document.metadata.source_url,
                indexed_at=document.metadata.indexed_at,
                last_updated=document.metadata.last_updated,
                version=document.metadata.version,
                project=document.project,
                author=document.author,
                title=document.title,
                parent_id=document.parent_id,
                chunk_index=document.chunk_index,
                summary=document.summary,
                keywords=document.keywords,
                entities=document.entities,
                tags=document.tags,
                has_embedding=1 if document.embedding else 0,
                created_at=document.created_at,
                updated_at=document.updated_at
            )

            self.session.add(db_document)
            await self.session.commit()
            await self.session.refresh(db_document)

            logger.info(f"Saved document: {document.id}")
            return self._to_domain(db_document)

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error saving document {document.id}: {str(e)}")
            raise

    async def find_by_id(self, document_id: str) -> Optional[Document]:
        """Find document by ID"""
        try:
            stmt = select(DocumentModel).where(DocumentModel.id == document_id)
            result = await self.session.execute(stmt)
            db_document = result.scalar_one_or_none()

            return self._to_domain(db_document) if db_document else None
        except Exception as e:
            logger.error(f"Error finding document {document_id}: {str(e)}")
            return None

    async def find_by_parent_id(self, parent_id: str) -> List[Document]:
        """Find all chunks of a parent document"""
        try:
            stmt = select(DocumentModel).where(
                DocumentModel.parent_id == parent_id
            ).order_by(DocumentModel.chunk_index)

            result = await self.session.execute(stmt)
            db_documents = result.scalars().all()

            return [self._to_domain(doc) for doc in db_documents]
        except Exception as e:
            logger.error(f"Error finding chunks for parent {parent_id}: {str(e)}")
            return []

    async def search(
            self,
            query: str,
            document_types: Optional[List[DocumentType]] = None,
            project: Optional[str] = None,
            author: Optional[str] = None,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            limit: int = 50
    ) -> List[Document]:
        """Search documents"""
        try:
            stmt = select(DocumentModel)
            conditions = []

            # Text search
            if query:
                search_conditions = [
                    DocumentModel.content.ilike(f"%{query}%"),
                    DocumentModel.title.ilike(f"%{query}%"),
                    DocumentModel.summary.ilike(f"%{query}%")
                ]
                conditions.append(or_(*search_conditions))

            # Type filter
            if document_types:
                type_enums = [DocumentTypeEnum[dt.name] for dt in document_types]
                conditions.append(DocumentModel.document_type.in_(type_enums))

            # Other filters
            if project:
                conditions.append(DocumentModel.project == project)
            if author:
                conditions.append(DocumentModel.author == author)
            if start_date:
                conditions.append(DocumentModel.created_at >= start_date)
            if end_date:
                conditions.append(DocumentModel.created_at <= end_date)

            if conditions:
                stmt = stmt.where(and_(*conditions))

            stmt = stmt.order_by(desc(DocumentModel.created_at)).limit(limit)

            result = await self.session.execute(stmt)
            db_documents = result.scalars().all()

            return [self._to_domain(doc) for doc in db_documents]

        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

    async def update(self, document: Document) -> Document:
        """Update document"""
        try:
            stmt = select(DocumentModel).where(DocumentModel.id == document.id)
            result = await self.session.execute(stmt)
            db_document = result.scalar_one_or_none()

            if not db_document:
                raise ValueError(f"Document not found: {document.id}")

            # Update fields
            db_document.content = document.content
            db_document.summary = document.summary
            db_document.keywords = document.keywords
            db_document.entities = document.entities
            db_document.tags = document.tags
            db_document.has_embedding = 1 if document.embedding else 0
            db_document.updated_at = datetime.now()

            await self.session.commit()
            await self.session.refresh(db_document)

            logger.info(f"Updated document: {document.id}")
            return self._to_domain(db_document)

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating document {document.id}: {str(e)}")
            raise

    async def delete(self, document_id: str) -> bool:
        """Delete document"""
        try:
            stmt = select(DocumentModel).where(DocumentModel.id == document_id)
            result = await self.session.execute(stmt)
            db_document = result.scalar_one_or_none()

            if not db_document:
                return False

            await self.session.delete(db_document)
            await self.session.commit()

            logger.info(f"Deleted document: {document_id}")
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False

    async def bulk_save(self, documents: List[Document]) -> List[Document]:
        """Save multiple documents"""
        saved = []
        for document in documents:
            try:
                saved_doc = await self.save(document)
                saved.append(saved_doc)
            except Exception as e:
                logger.error(f"Error in bulk save for document {document.id}: {str(e)}")

        return saved

    def _to_domain(self, db_model: DocumentModel) -> Document:
        """Convert SQLAlchemy model to domain entity"""
        metadata = DocumentMetadata(
            source_type=db_model.source_type,
            source_url=db_model.source_url,
            indexed_at=db_model.indexed_at,
            last_updated=db_model.last_updated,
            version=db_model.version
        )

        document = Document(
            content=db_model.content,
            document_type=DocumentType[db_model.document_type.name],
            metadata=metadata,
            project=db_model.project,
            author=db_model.author,
            title=db_model.title,
            parent_id=db_model.parent_id,
            chunk_index=db_model.chunk_index
        )

        # Set ID and timestamps
        document.id = db_model.id
        document.created_at = db_model.created_at
        document.updated_at = db_model.updated_at

        # Set extracted info
        document.summary = db_model.summary
        document.keywords = db_model.keywords or []
        document.entities = db_model.entities or []
        document.tags = db_model.tags or []

        # Note: embeddings are stored separately in vector store
        document.embedding = None

        return document
