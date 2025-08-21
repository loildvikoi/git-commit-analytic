# src/domain/repositories/document_repository.py
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..entities.document import Document, DocumentType


class IDocumentRepository(ABC):
    """Repository interface for Document entity"""

    @abstractmethod
    async def save(self, document: Document) -> Document:
        """Save a document"""
        pass

    @abstractmethod
    async def find_by_id(self, document_id: str) -> Optional[Document]:
        """Find document by ID"""
        pass

    @abstractmethod
    async def find_by_parent_id(self, parent_id: str) -> List[Document]:
        """Find all chunks of a parent document"""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def update(self, document: Document) -> Document:
        """Update document"""
        pass

    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """Delete document"""
        pass

    @abstractmethod
    async def bulk_save(self, documents: List[Document]) -> List[Document]:
        """Save multiple documents"""
        pass
