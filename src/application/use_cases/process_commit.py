from typing import Optional
from datetime import datetime
from ..dto.commit_dto import CommitDto, CommitResponseDto
from ...domain.repositories.commit_repository import ICommitRepository
from ...domain.entities.commit import Commit, CommitHash, FileChange
from ...domain.events.commit_events import CommitReceivedEvent
from ...domain.services.cache_service import ICacheService
from ...domain.services.event_dispatcher import IEventDispatcher
from ...domain.services.message_queue import IMessageQueue


class ProcessCommitUseCase:
    """Use case for processing incoming commits"""

    def __init__(
        self,
        commit_repository: ICommitRepository,
        event_dispatcher: IEventDispatcher,
        cache_service: ICacheService,
        message_queue: IMessageQueue
    ):
        self.commit_repository = commit_repository
        self.event_dispatcher = event_dispatcher
        self.cache_service = cache_service
        self.message_queue = message_queue

    async def execute(self, commit_dto: CommitDto) -> CommitResponseDto:
        """
        Process a commit DTO, save it to the repository, and dispatch an event.
        """

        # Check if the commit already exists
        existing = await self.commit_repository.find_by_hash(commit_dto.commit_hash)
        if existing:
            return self._to_response_dto(existing)

        # Convert DTO to domain entity
        commit = self._to_domain(commit_dto)

        # Save the commit to the repository
        saved_commit = await self.commit_repository.save(commit)

        # Dispatch an event for the received commit
        event = CommitReceivedEvent(
            commit_id=saved_commit.id,
            commit_hash=saved_commit.commit_hash.value,
            project=saved_commit.project,
            author=saved_commit.author_email,
            branch=saved_commit.branch
        )
        await self.event_dispatcher.dispatch(event)

        # Do async analysis by queue
        await self.message_queue.enqueue(
            "analyze_commit",
            {"commit_id": saved_commit.id}
        )

        # Invalidate relevant caches
        await self.cache_service.invalidate_pattern(f"commits:project:{commit.project}:*")
        await self.cache_service.invalidate_pattern(f"commits:author:{commit.author_email}:*")

        # Return a response DTO
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
