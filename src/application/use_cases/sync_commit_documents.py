from typing import List, Optional, Dict, Any
from ...domain.repositories.commit_repository import ICommitRepository
from ...domain.repositories.document_repository import IDocumentRepository
from ...domain.services.embedding_service import IEmbeddingService
from ...domain.repositories.vector_repository import IVectorRepository
from ...domain.entities.document import Document, DocumentType
import logging

logger = logging.getLogger(__name__)


class SyncCommitDocumentsUseCase:
    """Sync existing commits to document store for RAG"""

    def __init__(
            self,
            commit_repository: ICommitRepository,
            document_repository: IDocumentRepository,
            embedding_service: IEmbeddingService,
            vector_repository: IVectorRepository
    ):
        self.commit_repository = commit_repository
        self.document_repository = document_repository
        self.embedding_service = embedding_service
        self.vector_repository = vector_repository

    async def execute(
            self,
            project: Optional[str] = None,
            limit: int = 100,
            skip_existing: bool = True
    ) -> Dict[str, Any]:
        """Sync commits to document store"""

        # Get commits to sync
        if project:
            commits = await self.commit_repository.find_by_project(
                project=project,
                limit=limit
            )
        else:
            commits = await self.commit_repository.search(
                query="",
                limit=limit
            )

        results = {
            "total_commits": len(commits),
            "synced": 0,
            "skipped": 0,
            "failed": 0,
            "errors": []
        }

        for commit in commits:
            try:
                # Convert commit to document
                await self.commit2document(commit, skip_existing=skip_existing)
                results["synced"] += 1
                logger.info(f"Synced commit {commit.commit_hash.value[:8]} to documents")

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Commit {commit.commit_hash.value[:8]}: {str(e)}")
                logger.error(f"Failed to sync commit {commit.commit_hash.value}: {str(e)}", exc_info=True)

        return results

    async def commit2document(self, commit, skip_existing: bool = True):
        """Convert commit to document entity"""
        # Check if already exists
        if skip_existing:
            existing = await self.document_repository.search(
                query=commit.commit_hash.value,
                document_types=[DocumentType.COMMIT],
                limit=1
            )
            if existing:
                logger.info(f"Document already exists for commit {commit.commit_hash.value[:8]}")
                return existing[0]

        logger.info(f"Converting commit {commit.commit_hash.value} to document")

        # Create document from commit
        doc = Document.from_commit(commit, source_type="github")

        # Save document
        saved_doc = await self.document_repository.save(doc)

        # Generate embedding
        embedding = await self.embedding_service.generate_embedding(
            saved_doc.searchable_content
        )

        # Save to vector store
        await self.vector_repository.add_embedding(
            document_id=saved_doc.id,
            embedding=embedding,
            metadata={
                "document_type": DocumentType.COMMIT.value,
                "project": commit.project,
                "author": commit.author_email,
                "commit_hash": commit.commit_hash.value
            }
        )

        logger.info(f"Saved document {saved_doc.id} for commit {commit.commit_hash.value[:8]}")
        return saved_doc
