from typing import List
from datetime import datetime, timedelta
import time
from ..dto.commit_dto import ChatRequestDto, ChatResponseDto
from ...domain.repositories.commit_repository import ICommitRepository
from ...domain.services.ai_analyzer import IAIAnalyzer
from ...domain.services.cache_service import ICacheService
import hashlib


class ChatWithAIUseCase:
    """Use case for interacting with AI to analyze commits and answer questions."""

    def __init__(
            self,
            commit_repository: ICommitRepository,
            ai_analyzer: IAIAnalyzer,
            cache_service: ICacheService
    ):
        self.commit_repository = commit_repository
        self.ai_analyzer = ai_analyzer
        self.cache_service = cache_service

    async def execute(self, chat_dto: ChatRequestDto) -> ChatResponseDto:
        """Answer a question about commits."""

        start_time = time.time()

        # Create a unique cache key based on the question and context
        cache_key = self._generate_cache_key(chat_dto)

        # Check if the answer is cached
        cached_result = await self.cache_service.get(cache_key)
        if cached_result:
            return cached_result

        # Fetch commits based on the provided context
        end_date = datetime.now()
        start_date = end_date - timedelta(days=chat_dto.context_days)

        context_commits = await self.commit_repository.find_by_project(
            project=chat_dto.project,
            start_date=start_date,
            end_date=end_date,
            limit=100
        )

        if chat_dto.context_author:
            author_commits = await self.commit_repository.find_by_author(
                author=chat_dto.context_author,
                project=chat_dto.context_project,
                start_date=start_date,
                end_date=end_date,
                limit=50
            )
            context_commits.extend(author_commits)

        # Remove duplicates
        unique_commits = {commit.id: commit for commit in context_commits}
        context_commits = list(unique_commits.values())

        # Ask the AI to answer the question
        answer = await self.ai_analyzer.answer_question(
            question=chat_dto.question,
            context_commits=context_commits
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        result = ChatResponseDto(
            answer=answer,
            processing_time_ms=processing_time_ms,
            model_used=self.ai_analyzer.get_model_info()['name'],
            context_commits_count=len(context_commits),
        )

        # Cache the result
        await self.cache_service.set(cache_key, result, ttl=600)  # Cache for 10 minutes

        return result

    def _generate_cache_key(self, chat_dto: ChatRequestDto) -> str:
        """Generate cache key for chat request"""
        chat_hash = hashlib.md5(
            f"{chat_dto.question}:{chat_dto.context_project}:{chat_dto.context_author}:{chat_dto.context_days}".encode()
        ).hexdigest()
        return f"chat:response:{chat_hash}"
