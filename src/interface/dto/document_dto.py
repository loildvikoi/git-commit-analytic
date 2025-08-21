from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class DocumentTypeDto(str, Enum):
    COMMIT = "commit"
    PULL_REQUEST = "pull_request"
    ISSUE = "issue"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    DISCUSSION = "discussion"


class IndexDocumentDto(BaseModel):
    """DTO for indexing a document"""
    content: str = Field(..., min_length=1, description="Document content")
    document_type: DocumentTypeDto = Field(..., description="Type of document")
    source_type: str = Field(..., description="Source of document (github, gitlab, etc)")
    project: Optional[str] = Field(None, description="Project name")
    author: Optional[str] = Field(None, description="Author email or username")
    title: Optional[str] = Field(None, description="Document title")
    chunk_size: int = Field(500, ge=100, le=2000, description="Chunk size for splitting")
    chunk_overlap: int = Field(50, ge=0, le=200, description="Overlap between chunks")


class DocumentResponseDto(BaseModel):
    """Response DTO for document"""
    id: str
    content: str
    document_type: str
    project: Optional[str]
    author: Optional[str]
    title: Optional[str]
    parent_id: Optional[str]
    chunk_index: int
    has_embedding: bool
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class SearchDocumentsDto(BaseModel):
    """DTO for searching documents"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    document_types: Optional[List[DocumentTypeDto]] = Field(None, description="Filter by document types")
    projects: Optional[List[str]] = Field(None, description="Filter by projects")
    authors: Optional[List[str]] = Field(None, description="Filter by authors")
    use_hybrid: bool = Field(True, description="Use hybrid search (semantic + keyword)")
    use_cache: bool = Field(True, description="Use cached results if available")
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of results")


class SearchResultDto(BaseModel):
    """Response DTO for search result"""
    document: DocumentResponseDto
    score: Dict[str, float]  # semantic, keyword, combined, confidence
    highlights: List[str]
    explanation: Optional[str]


class RAGChatDto(BaseModel):
    """DTO for RAG chat requests"""
    question: str = Field(..., min_length=1, max_length=1000, description="Question to ask")
    context_project: Optional[str] = Field(None, description="Project context")
    context_author: Optional[str] = Field(None, description="Author context")
    max_documents: int = Field(5, ge=1, le=20, description="Max documents to use as context")
    use_cache: bool = Field(True, description="Use cached results if available")


class RAGChatResponseDto(BaseModel):
    """Response DTO for RAG chat"""
    question: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    context_used: int
    method: str
    cached: bool


class SyncCommitsDto(BaseModel):
    """DTO for syncing commits to documents"""
    project: Optional[str] = Field(None, description="Project to sync")
    limit: int = Field(100, ge=1, le=1000, description="Number of commits to sync")
    skip_existing: bool = Field(True, description="Skip already synced commits")