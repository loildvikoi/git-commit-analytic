import logging

from fastapi import Depends

from ...interface.api.dependencies import get_sync_commits_use_case
from ...interface.dto.commit_dto import CommitDto, CommitResponseDto
from ...domain.repositories.commit_repository import ICommitRepository
from ...domain.entities.commit import Commit, CommitHash, FileChange
from src.application.events.commit_events import CommitReceivedEvent
from ...domain.services.event_dispatcher import IEventDispatcher

logger = logging.getLogger(__name__)


class ProcessCommitUseCase:
    """Use case for processing incoming commits - priority is to save the commit, also need to response as quickly as possible"""

    def __init__(
        self,
        commit_repository: ICommitRepository,
        event_dispatcher: IEventDispatcher,
        sync_commit_documents_use_case=None
    ):
        self.commit_repository = commit_repository
        self.event_dispatcher = event_dispatcher
        self.sync_commit_documents_use_case = sync_commit_documents_use_case

    async def execute(self, commit_dto: CommitDto) -> CommitResponseDto:
        """
        Process a commit DTO - single responsibility: save business input to persistence layer.
        Side effects (analysis, caching) are handled by event handlers
        """

        # Check if commit already exists
        existing = await self.commit_repository.find_by_hash(commit_dto.commit_hash)
        if existing:
            logger.info(f"Commit {commit_dto.commit_hash} already exists")
            return self._to_response_dto(existing)

        # Convert DTO to domain entity
        commit = self._to_domain(commit_dto)

        # Save the commit (this is the primary responsibility)
        saved_commit = await self.commit_repository.save(commit)
        logger.info(f"Saved commit {saved_commit.commit_hash.value}")

        # If saving failed, raise an exception
        if not saved_commit:
            logger.error(f"Failed to save commit {commit.commit_hash.value}")
            raise Exception("Failed to save commit")

        # Dispatch domain event (let handlers handle side effects)
        event = CommitReceivedEvent(
            commit_id=saved_commit.id,
            commit_hash=saved_commit.commit_hash.value,
            project=saved_commit.project,
            author=saved_commit.author_email,
            branch=saved_commit.branch
        )

        await self.event_dispatcher.dispatch(event)
        logger.info(f"Dispatched CommitReceivedEvent for {saved_commit.commit_hash.value}")

        # Sync commit to documents

        if self.sync_commit_documents_use_case:
            logger.info(f"Syncing commit {saved_commit.commit_hash.value} to documents")
            await self.sync_commit_documents_use_case.commit2document(saved_commit, skip_existing=True)

        return self._to_response_dto(saved_commit)

    def _to_domain(self, dto: CommitDto) -> Commit:
        """Convert DTO to domain entity"""
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

    def _to_response_dto(self, commit: Commit) -> CommitResponseDto:
        """Convert domain entity to response DTO"""
        return CommitResponseDto(
            id=commit.id,
            commit_hash=commit.commit_hash.value,
            author_email=commit.author_email,
            author_name=commit.author_name,
            message=commit.message,
            timestamp=commit.timestamp,
            branch=commit.branch,
            project=commit.project,
            files_count=commit.metrics.files_count,
            total_lines_changed=commit.metrics.total_lines_changed,
            complexity_score=commit.metrics.complexity_score,
            summary=commit.summary,
            tags=commit.tags,
            sentiment_score=commit.sentiment_score,
            analyzed_at=commit.analyzed_at,
            created_at=commit.created_at
        )
