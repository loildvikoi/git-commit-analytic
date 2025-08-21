from sqlalchemy import Column, String, Text, Integer, Float, JSON, DateTime, Enum as SQLEnum
from sqlalchemy.sql import func
from ..database import Base
import enum


class DocumentTypeEnum(enum.Enum):
    COMMIT = "commit"
    PULL_REQUEST = "pull_request"
    ISSUE = "issue"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    DISCUSSION = "discussion"


class DocumentModel(Base):
    """SQLAlchemy model for documents"""

    __tablename__ = "documents"

    # Primary fields
    id = Column(String(36), primary_key=True)
    content = Column(Text, nullable=False)
    document_type = Column(SQLEnum(DocumentTypeEnum), nullable=False, index=True)

    # Metadata
    source_type = Column(String(50), nullable=False, index=True)
    source_url = Column(String(500))
    indexed_at = Column(DateTime, index=True)
    last_updated = Column(DateTime)
    version = Column(Integer, default=1)

    # Key fields for searching
    project = Column(String(255), index=True)
    author = Column(String(255), index=True)
    title = Column(String(500))

    # Chunking
    parent_id = Column(String(36), index=True)
    chunk_index = Column(Integer, default=0)

    # Extracted information
    summary = Column(Text)
    keywords = Column(JSON, default=list)
    entities = Column(JSON, default=list)
    tags = Column(JSON, default=list)

    # Vector search
    has_embedding = Column(Integer, default=0)  # Boolean as integer for SQLite
    embedding_model = Column(String(100))

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<DocumentModel(id={self.id}, type={self.document_type}, project={self.project})>"

