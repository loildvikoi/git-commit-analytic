# src/application/use_cases/enhanced_process_commit.py
"""
Enhanced commit processing that integrates Phase 1, 2, and 3
"""
import logging
from typing import Dict, Any, List
from ...domain.repositories.commit_repository import ICommitRepository
from ...domain.repositories.document_repository import IDocumentRepository
from ...domain.services.embedding_service import IEmbeddingService
from ...domain.services.agent_service import IAgentService
from ...domain.services.event_dispatcher import IEventDispatcher
from ...domain.entities.agent import AgentRole

logger = logging.getLogger(__name__)


class EnhancedProcessCommitUseCase:
    """
    Integrates all three phases:
    - Phase 1: Store commit and basic AI analysis
    - Phase 2: Create document with embeddings for RAG
    - Phase 3: Agent-based deep analysis and insights
    """

    def __init__(
            self,
            commit_repository: ICommitRepository,
            document_repository: IDocumentRepository,
            embedding_service: IEmbeddingService,
            agent_service: IAgentService,
            event_dispatcher: IEventDispatcher
    ):
        self.commit_repository = commit_repository
        self.document_repository = document_repository
        self.embedding_service = embedding_service
        self.agent_service = agent_service
        self.event_dispatcher = event_dispatcher

    async def execute(self, commit_dto) -> Dict[str, Any]:
        """Process commit through all three phases"""

        # Phase 1: Store commit
        commit = await self._phase1_store_commit(commit_dto)

        # Phase 2: Create searchable document
        document = await self._phase2_create_document(commit)

        # Phase 3: Agent analysis (async - don't block response)
        await self._phase3_agent_analysis(commit, document)

        return {
            "commit_id": commit.id,
            "commit_hash": commit.commit_hash,
            "document_id": document.id,
            "status": "processing",
            "phases_completed": ["storage", "indexing", "analysis_queued"]
        }

    async def _phase1_store_commit(self, commit_dto):
        """Phase 1: Basic storage"""
        # Convert DTO to domain entity
        commit = self._to_domain(commit_dto)

        # Save to database
        saved_commit = await self.commit_repository.save(commit)

        # Fire event for background processing
        from ...application.events.commit_events import CommitReceivedEvent
        event = CommitReceivedEvent(
            commit_id=saved_commit.id,
            commit_hash=saved_commit.commit_hash.value,
            project=saved_commit.project,
            author=saved_commit.author_email,
            branch=saved_commit.branch
        )
        await self.event_dispatcher.dispatch(event)

        logger.info(f"Phase 1: Stored commit {saved_commit.commit_hash.value}")
        return saved_commit

    async def _phase2_create_document(self, commit):
        """Phase 2: Create searchable document with embeddings"""
        from ...domain.entities.document import Document

        # Create document from commit
        document = Document.from_commit(commit, source_type="webhook")

        # Save document
        saved_doc = await self.document_repository.save(document)

        # Generate embedding for semantic search
        embedding = await self.embedding_service.generate_embedding(
            document.searchable_content
        )

        # Store embedding in vector database
        from ...interface.api.dependencies import get_vector_repository
        vector_repo = await get_vector_repository()

        await vector_repo.add_embedding(
            document_id=saved_doc.id,
            embedding=embedding,
            metadata={
                "commit_hash": commit.commit_hash.value,
                "project": commit.project,
                "author": commit.author_email,
                "document_type": "commit"
            }
        )

        logger.info(f"Phase 2: Created document {saved_doc.id} with embeddings")
        return saved_doc

    async def _phase3_agent_analysis(self, commit, document):
        """Phase 3: Agent-based analysis"""
        # Fire event for agent processing
        from ..events.agent_events import AgentAnalysisRequestedEvent
        event = AgentAnalysisRequestedEvent(
            commit_id=commit.id,
            document_id=document.id,
            analysis_types=["code_quality", "bug_detection", "insights"]
        )
        await self.event_dispatcher.dispatch(event)

        logger.info(f"Phase 3: Queued agent analysis for {commit.id}")

    def _to_domain(self, dto):
        """Convert DTO to domain entity"""
        from ...domain.entities.commit import Commit, CommitHash, FileChange

        return Commit(
            commit_hash=CommitHash(dto.commit_hash),
            author_email=dto.author_email,
            author_name=dto.author_name,
            message=dto.message,
            timestamp=dto.timestamp,
            branch=dto.branch,
            project=dto.project,
            files_changed=[
                FileChange(
                    filename=fc.filename,
                    additions=fc.additions,
                    deletions=fc.deletions,
                    status=fc.status
                ) for fc in dto.files_changed
            ],
            issue_numbers=dto.issue_numbers
        )
