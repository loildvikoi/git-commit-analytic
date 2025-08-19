from sqlalchemy import Column, String, DateTime, Text, Integer, Float, JSON
from sqlalchemy.sql import func
from ..database import Base


class CommitModel(Base):
    """SQLAlchemy model for commits"""

    __tablename__ = "commits"

    # Primary fields
    id = Column(Integer, primary_key=True)
    commit_hash = Column(String(40), unique=True, nullable=False, index=True)
    author_email = Column(String(255), nullable=False, index=True)
    author_name = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    branch = Column(String(255), nullable=False, index=True)
    project = Column(String(255), nullable=False, index=True)

    # File changes (stored as JSON)
    files_changed = Column(JSON, nullable=False, default=list)
    issue_numbers = Column(JSON, nullable=False, default=list)

    # Metrics
    total_lines_changed = Column(Integer, default=0)
    files_count = Column(Integer, default=0)
    complexity_score = Column(Float)
    impact_score = Column(Float)

    # Analysis results
    summary = Column(Text)
    tags = Column(JSON, default=list)
    sentiment_score = Column(Float)
    embedding_id = Column(String(255))
    analyzed_at = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<CommitModel(hash={self.commit_hash}, project={self.project})>"


class AnalysisModel(Base):
    """SQLAlchemy model for analysis results"""

    __tablename__ = "analyses"

    id = Column(String(36), primary_key=True)
    commit_id = Column(String(36), nullable=False, index=True)

    # Model information
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    prompt_version = Column(String(50), nullable=False)

    # Analysis results
    summary = Column(Text, nullable=False)
    tags = Column(JSON, default=list)
    sentiment_score = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    extracted_entities = Column(JSON, default=list)

    # Performance metrics
    processing_time_ms = Column(Integer, nullable=False)
    tokens_used = Column(Integer, default=0)
    status = Column(String(20), default="completed")

    # Timestamps
    created_at = Column(DateTime, default=func.now(), nullable=False)

    def __repr__(self):
        return f"<AnalysisModel(commit_id={self.commit_id}, model={self.model_name})>"
