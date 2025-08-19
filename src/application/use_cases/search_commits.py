from typing import List
from ..dto.commit_dto import CommitSearchDto, CommitResponseDto
from ...domain.repositories.commit_repository import ICommitRepository
from ...domain.services.cache_service import ICacheService
import hashlib
import json


class SearchCommitsUseCase:
    """Use case for searching commits based on various criteria."""

    def __init(self, commit_repository: ICommitRepository, cache_service: ICacheService):
        self.commit_repository = commit_repository
        self.cache_service = cache_service

    async def execute(self, search_dto: CommitSearchDto) -> List[CommitResponseDto]:
        """Search commits with caching."""

        # Create a unique cache key based on the search criteria
        cache_key = self._generate_cache_key(search_dto)

        # Check if the results are cached
        cached_results = await self.cache_service.get(cache_key)
        if cached_results:
            return cached_results

        # Perform the search in the repository
        commits = await self.commit_repository.search(
            query=search_dto.query,
            filters={
                'project': search_dto.project,
                'author': search_dto.author,
                'branch': search_dto.branch,
                'start_date': search_dto.start_date,
                'end_date': search_dto.end_date,
                'tags': search_dto.tags
            },
            limit=search_dto.limit,
        )

        # Convert domain entities to response DTOs
        result = [self._to_response_dto(commit) for commit in commits]

        # Cache the results
        await self.cache_service.set(cache_key, result, ttl=300)  # Cache for 5 minutes

        return result

    def _generate_cache_key(self, search_dto: CommitSearchDto) -> str:
        """Generate cache key for search parameters"""
        search_hash = hashlib.md5(
            json.dumps(search_dto.dict(), sort_keys=True, default=str).encode()
        ).hexdigest()
        return f"search:commits:{search_hash}"

    def _to_response_dto(self, commit) -> CommitResponseDto:
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
