# src/interface/api/v1/documents.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List
import logging

from src.interface.dto.document_dto import (
    IndexDocumentDto,
    DocumentResponseDto,
    SearchDocumentsDto,
    SearchResultDto,
    SyncCommitsDto
)
from src.interface.api.dependencies import (
    get_index_document_use_case,
    get_search_documents_use_case,
    get_sync_commits_use_case
)

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


@router.post("/index", response_model=List[DocumentResponseDto])
async def index_document(
        request: IndexDocumentDto,
        use_case=Depends(get_index_document_use_case)
):
    """Index a new document into the RAG system"""
    try:
        from ....domain.entities.document import DocumentType

        documents = await use_case.execute(
            content=request.content,
            document_type=DocumentType[request.document_type.value.upper()],
            source_type=request.source_type,
            project=request.project,
            author=request.author,
            title=request.title,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )

        # Convert to response DTOs
        return [
            DocumentResponseDto(
                id=doc.id,
                content=doc.content,
                document_type=doc.document_type.value,
                project=doc.project,
                author=doc.author,
                title=doc.title,
                parent_id=doc.parent_id,
                chunk_index=doc.chunk_index,
                has_embedding=doc.embedding is not None,
                metadata={
                    "source_type": doc.metadata.source_type,
                    "source_url": doc.metadata.source_url,
                    "version": doc.metadata.version
                },
                created_at=doc.created_at,
                updated_at=doc.updated_at
            )
            for doc in documents
        ]

    except Exception as e:
        logger.error(f"Error indexing document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=List[SearchResultDto])
async def search_documents(
        request: SearchDocumentsDto,
        use_case=Depends(get_search_documents_use_case)
):
    """Search documents using hybrid search"""
    try:
        results = await use_case.execute(
            query_text=request.query,
            document_types=[dt.value for dt in request.document_types] if request.document_types else None,
            projects=request.projects,
            authors=request.authors,
            use_hybrid=request.use_hybrid,
            use_cache=request.use_cache,
            max_results=request.max_results
        )

        # Convert to response DTOs
        return [
            SearchResultDto(
                document=DocumentResponseDto(
                    id=result["document"]["id"],
                    content=result["document"]["content"],
                    document_type=result["document"]["document_type"],
                    project=result["document"]["project"],
                    author=result["document"]["author"],
                    title=result["document"]["title"],
                    parent_id=result["document"]["parent_id"],
                    chunk_index=result["document"]["chunk_index"],
                    has_embedding=result["document"]["has_embedding"],
                    metadata=result["document"]["metadata"],
                    created_at=result["document"]["created_at"],
                    updated_at=result["document"]["updated_at"]
                ),
                score=result["score"],
                highlights=result["highlights"],
                explanation=result["explanation"]
            )
            for result in results
        ]

    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync-commits")
async def sync_commits_to_documents(
        request: SyncCommitsDto,
        use_case=Depends(get_sync_commits_use_case)
):
    """Sync existing commits to document store for RAG"""
    try:
        result = await use_case.execute(
            project=request.project,
            limit=request.limit,
            skip_existing=request.skip_existing
        )

        return {
            "status": "success",
            "data": result
        }

    except Exception as e:
        logger.error(f"Error syncing commits: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

