from abc import abstractmethod, ABC
from datetime import datetime
from typing import Optional, Any, Dict, List

from src.domain.entities.commit import Commit


class ICommitRepository(ABC):
    @abstractmethod
    async def save(self, commit: Commit) -> Commit:
        """Save a commit to the repository."""
        pass

    @abstractmethod
    async def find_by_id(self, commit_id: str) -> Optional[Commit]:
        """Find a commit by its ID."""
        pass

    @abstractmethod
    async def find_by_hash(self, commit_hash: str) -> Optional[Commit]:
        """Find a commit by its hash."""
        pass

    @abstractmethod
    async def find_by_author(
            self,
            author_email: str,
            project: Optional[str] = None,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            limit: int = 100
    ) -> List[Commit]:
        """Find commits by author with filters"""
        pass

    @abstractmethod
    async def find_by_project(
            self,
            project: str,
            branch: Optional[str] = None,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            limit: int = 100
    ) -> List[Commit]:
        """Find commits by project with filters"""
        pass

    @abstractmethod
    async def search(
            self,
            query: str,
            filters: Optional[Dict[str, Any]] = None,
            limit: int = 50
    ) -> List[Commit]:
        """Search commits by message, author, or other fields"""
        pass

    @abstractmethod
    async def update(self, commit: Commit) -> Commit:
        """Update commit with analysis results"""
        pass

    @abstractmethod
    async def get_statistics(
            self,
            project: Optional[str] = None,
            author: Optional[str] = None,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get commit statistics"""
        pass
