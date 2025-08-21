# src/domain/entities/document.py
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
from .base import Entity, ValueObject


class DocumentType(Enum):
    """Types of documents in the system"""
    COMMIT = "commit"
    PULL_REQUEST = "pull_request"
    ISSUE = "issue"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    DISCUSSION = "discussion"


@dataclass
class DocumentMetadata(ValueObject):
    """Value object for document metadata"""
    source_type: str  # github, gitlab, gitlog, manual
    source_url: Optional[str] = None
    indexed_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    version: int = 1
    language: Optional[str] = None

    def _validate(self):
        if self.source_type not in ['github', 'gitlab', 'gitlog', 'manual', 'api']:
            raise ValueError(f"Invalid source type: {self.source_type}")
        if self.version < 1:
            raise ValueError("Version must be positive")


@dataclass
class SearchScore(ValueObject):
    """Value object for search scoring"""
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0
    confidence: float = 0.0

    def _validate(self):
        for score in [self.semantic_score, self.keyword_score, self.combined_score, self.confidence]:
            if not (0 <= score <= 1):
                raise ValueError(f"Score must be between 0 and 1, got {score}")


class Document(Entity):
    """Document entity for RAG system - represents searchable content"""

    def __init__(
            self,
            content: str,
            document_type: DocumentType,
            metadata: DocumentMetadata,
            project: Optional[str] = None,
            author: Optional[str] = None,
            title: Optional[str] = None,
            embedding: Optional[List[float]] = None,
            parent_id: Optional[str] = None,
            chunk_index: int = 0
    ):
        super().__init__()
        self.content = content
        self.document_type = document_type
        self.metadata = metadata
        self.project = project
        self.author = author
        self.title = title
        self.embedding = embedding
        self.parent_id = parent_id  # For chunked documents
        self.chunk_index = chunk_index

        # Extracted information
        self.summary: Optional[str] = None
        self.keywords: List[str] = []
        self.entities: List[str] = []
        self.tags: List[str] = []

        # Search-related
        self._score: Optional[SearchScore] = None

    @property
    def is_chunk(self) -> bool:
        """Check if this is a chunk of a larger document"""
        return self.parent_id is not None

    @property
    def searchable_content(self) -> str:
        """Generate content optimized for search"""
        parts = []

        # Title/summary first for relevance
        if self.title:
            parts.append(f"Title: {self.title}")
        if self.summary:
            parts.append(f"Summary: {self.summary}")

        # Main content
        parts.append(self.content)

        # Metadata for context
        if self.project:
            parts.append(f"Project: {self.project}")
        if self.author:
            parts.append(f"Author: {self.author}")
        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")

        return "\n".join(parts)

    def set_search_score(self, score: SearchScore):
        """Set search relevance score"""
        self._score = score

    def get_search_score(self) -> Optional[SearchScore]:
        """Get search relevance score"""
        return self._score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "content": self.content,
            "document_type": self.document_type.value,
            "metadata": {
                "source_type": self.metadata.source_type,
                "source_url": self.metadata.source_url,
                "indexed_at": self.metadata.indexed_at.isoformat() if self.metadata.indexed_at else None,
                "version": self.metadata.version
            },
            "project": self.project,
            "author": self.author,
            "title": self.title,
            "parent_id": self.parent_id,
            "chunk_index": self.chunk_index,
            "summary": self.summary,
            "keywords": self.keywords,
            "entities": self.entities,
            "tags": self.tags,
            "has_embedding": self.embedding is not None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_commit(cls, commit, source_type: str = "api") -> "Document":
        """Factory method to create Document from Commit entity"""
        from datetime import datetime

        # Build content from commit
        content_parts = [
            f"Commit: {commit.commit_hash.value}",
            f"Message: {commit.message}",
            f"Author: {commit.author_name} <{commit.author_email}>",
            f"Branch: {commit.branch}",
            f"Files changed: {len(commit.files_changed)}"
        ]

        if commit.summary:
            content_parts.append(f"AI Summary: {commit.summary}")

        content = "\n".join(content_parts)

        # Create metadata
        metadata = DocumentMetadata(
            source_type=source_type,
            source_url=f"commit/{commit.commit_hash.value}",
            indexed_at=datetime.now(),
            version=1
        )

        # Create document
        doc = cls(
            content=content,
            document_type=DocumentType.COMMIT,
            metadata=metadata,
            project=commit.project,
            author=commit.author_email,
            title=commit.message[:100]  # First 100 chars as title
        )

        # Set extracted info
        doc.summary = commit.summary
        doc.tags = commit.tags if commit.tags else []
        doc.entities = commit.issue_numbers if commit.issue_numbers else []

        return doc