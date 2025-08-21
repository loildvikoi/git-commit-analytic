from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func
from sqlalchemy.orm import selectinload
from ....domain.repositories.commit_repository import ICommitRepository
from ....domain.entities.commit import Commit, CommitHash, FileChange
from ..models.commit_model import CommitModel
import logging

logger = logging.getLogger(__name__)


class SqliteCommitRepository(ICommitRepository):
    """SQLite/PostgreSQL implementation of commit repository"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, commit: Commit) -> Commit:
        """Save commit to database"""
        try:
            # Calculate metrics
            metrics = commit.metrics

            # Create database model
            db_commit = CommitModel(
                # id=commit.id,
                commit_hash=commit.commit_hash.value,
                author_email=commit.author_email,
                author_name=commit.author_name,
                message=commit.message,
                timestamp=commit.timestamp,
                branch=commit.branch,
                project=commit.project,
                files_changed=[
                    {
                        'filename': fc.filename,
                        'additions': fc.additions,
                        'deletions': fc.deletions,
                        'status': fc.status
                    } for fc in commit.files_changed
                ],
                issue_numbers=commit.issue_numbers,
                total_lines_changed=metrics.total_lines_changed,
                files_count=metrics.files_count,
                complexity_score=metrics.complexity_score,
                impact_score=metrics.impact_score,
                summary=commit.summary,
                tags=commit.tags,
                sentiment_score=commit.sentiment_score,
                embedding_id=commit.embedding_id,
                analyzed_at=commit.analyzed_at,
                created_at=commit.created_at,
                updated_at=commit.updated_at
            )

            self.session.add(db_commit)
            await self.session.commit()
            await self.session.refresh(db_commit)

            logger.info(f"Saved commit: {commit.commit_hash.value}")
            return self._to_domain(db_commit)

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error saving commit {commit.commit_hash.value}: {str(e)}", exc_info=True)
            raise

    async def find_by_id(self, commit_id: str) -> Optional[Commit]:
        """Find commit by ID"""
        try:
            stmt = select(CommitModel).where(CommitModel.id == commit_id)
            result = await self.session.execute(stmt)
            db_commit = result.scalar_one_or_none()

            return self._to_domain(db_commit) if db_commit else None
        except Exception as e:
            logger.debug(f"Error finding commit by ID {commit_id}: {str(e)}", exc_info=True)
            return None

    async def find_by_hash(self, commit_hash: str) -> Optional[Commit]:
        """Find commit by hash"""
        try:
            stmt = select(CommitModel).where(CommitModel.commit_hash == commit_hash)
            result = await self.session.execute(stmt)
            db_commit = result.scalar_one_or_none()

            return self._to_domain(db_commit) if db_commit else None
        except Exception as e:
            logger.debug(f"Error finding commit by hash {commit_hash}: {str(e)}", exc_info=True)
            return None

    async def find_by_author(
            self,
            author_email: str,
            project: Optional[str] = None,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            limit: int = 100
    ) -> List[Commit]:
        """Find commits by author with filters"""
        try:
            stmt = select(CommitModel).where(CommitModel.author_email == author_email)

            if project:
                stmt = stmt.where(CommitModel.project == project)
            if start_date:
                stmt = stmt.where(CommitModel.timestamp >= start_date)
            if end_date:
                stmt = stmt.where(CommitModel.timestamp <= end_date)

            stmt = stmt.order_by(desc(CommitModel.timestamp)).limit(limit)

            result = await self.session.execute(stmt)
            db_commits = result.scalars().all()

            return [self._to_domain(db_commit) for db_commit in db_commits]
        except Exception as e:
            logger.debug(f"Error finding commits by author {author_email}: {str(e)}", exc_info=True)
            return []

    async def find_by_project(
            self,
            project: str,
            branch: Optional[str] = None,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            limit: int = 100
    ) -> List[Commit]:
        """Find commits by project with filters"""
        try:
            stmt = select(CommitModel).where(CommitModel.project == project)

            if branch:
                stmt = stmt.where(CommitModel.branch == branch)
            if start_date:
                stmt = stmt.where(CommitModel.timestamp >= start_date)
            if end_date:
                stmt = stmt.where(CommitModel.timestamp <= end_date)

            stmt = stmt.order_by(desc(CommitModel.timestamp)).limit(limit)

            result = await self.session.execute(stmt)
            db_commits = result.scalars().all()

            return [self._to_domain(db_commit) for db_commit in db_commits]
        except Exception as e:
            logger.debug(f"Error finding commits by project {project}: {str(e)}", exc_info=True)
            return []

    async def search(
            self,
            query: str,
            filters: Optional[Dict[str, Any]] = None,
            limit: int = 50
    ) -> List[Commit]:
        """Search commits by message, author, or other fields"""
        try:
            stmt = select(CommitModel)
            conditions = []

            # Text search in message, author name, or commit hash
            if query:
                search_conditions = [
                    CommitModel.message.ilike(f"%{query}%"),
                    CommitModel.author_name.ilike(f"%{query}%"),
                    CommitModel.commit_hash.ilike(f"%{query}%")
                ]
                conditions.append(or_(*search_conditions))

            # Apply filters
            if filters:
                if filters.get('project'):
                    conditions.append(CommitModel.project == filters['project'])
                if filters.get('author'):
                    conditions.append(CommitModel.author_email == filters['author'])
                if filters.get('branch'):
                    conditions.append(CommitModel.branch == filters['branch'])
                if filters.get('start_date'):
                    conditions.append(CommitModel.timestamp >= filters['start_date'])
                if filters.get('end_date'):
                    conditions.append(CommitModel.timestamp <= filters['end_date'])
                if filters.get('tags'):
                    # JSON contains any of the tags
                    for tag in filters['tags']:
                        conditions.append(CommitModel.tags.contains([tag]))

            if conditions:
                stmt = stmt.where(and_(*conditions))

            stmt = stmt.order_by(desc(CommitModel.timestamp)).limit(limit)

            result = await self.session.execute(stmt)
            db_commits = result.scalars().all()

            return [self._to_domain(db_commit) for db_commit in db_commits]
        except Exception as e:
            logger.debug(f"Error searching commits with query '{query}': {str(e)}", exc_info=True)
            return []

    async def update(self, commit: Commit) -> Commit:
        """Update commit with analysis results"""
        try:
            stmt = select(CommitModel).where(CommitModel.id == commit.id)
            result = await self.session.execute(stmt)
            db_commit = result.scalar_one_or_none()

            if not db_commit:
                raise ValueError(f"Commit not found: {commit.id}")

            # Update analysis fields
            db_commit.summary = commit.summary
            db_commit.tags = commit.tags
            db_commit.sentiment_score = commit.sentiment_score
            db_commit.embedding_id = commit.embedding_id
            db_commit.analyzed_at = commit.analyzed_at
            db_commit.updated_at = datetime.utcnow()

            await self.session.commit()
            await self.session.refresh(db_commit)

            logger.info(f"Updated commit: {commit.commit_hash.value}")
            return self._to_domain(db_commit)
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating commit {commit.id}: {str(e)}", exc_info=True)
            raise

    async def get_statistics(
            self,
            project: Optional[str] = None,
            author: Optional[str] = None,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get commit statistics"""
        try:
            base_query = select(CommitModel)
            conditions = []

            if project:
                conditions.append(CommitModel.project == project)
            if author:
                conditions.append(CommitModel.author_email == author)
            if start_date:
                conditions.append(CommitModel.timestamp >= start_date)
            if end_date:
                conditions.append(CommitModel.timestamp <= end_date)

            if conditions:
                base_query = base_query.where(and_(*conditions))

            # Total commits
            count_result = await self.session.execute(
                select(func.count(CommitModel.id)).select_from(base_query.subquery())
            )
            total_commits = count_result.scalar()

            # Average metrics
            metrics_result = await self.session.execute(
                select(
                    func.avg(CommitModel.total_lines_changed),
                    func.avg(CommitModel.files_count),
                    func.avg(CommitModel.complexity_score),
                    func.avg(CommitModel.sentiment_score)
                ).select_from(base_query.subquery())
            )
            avg_lines, avg_files, avg_complexity, avg_sentiment = metrics_result.first()

            # Top authors
            authors_result = await self.session.execute(
                select(
                    CommitModel.author_email,
                    CommitModel.author_name,
                    func.count(CommitModel.id).label('commit_count')
                ).select_from(base_query.subquery())
                .group_by(CommitModel.author_email, CommitModel.author_name)
                .order_by(desc('commit_count'))
                .limit(10)
            )
            top_authors = [
                {
                    'email': row.author_email,
                    'name': row.author_name,
                    'commits': row.commit_count
                }
                for row in authors_result
            ]

            return {
                'total_commits': total_commits or 0,
                'avg_lines_changed': float(avg_lines or 0),
                'avg_files_per_commit': float(avg_files or 0),
                'avg_complexity_score': float(avg_complexity or 0),
                'avg_sentiment_score': float(avg_sentiment or 0),
                'top_authors': top_authors
            }
        except Exception as e:
            logger.debug(f"Error getting statistics: {str(e)}", exc_info=True)
            return {}

    def _to_domain(self, db_model: CommitModel) -> Commit:
        """Convert SQLAlchemy model to domain entity"""
        commit = Commit(
            commit_hash=CommitHash(db_model.commit_hash),
            author_email=db_model.author_email,
            author_name=db_model.author_name,
            message=db_model.message,
            timestamp=db_model.timestamp,
            branch=db_model.branch,
            project=db_model.project,
            files_changed=[
                FileChange(**fc) for fc in (db_model.files_changed or [])
            ],
            issue_numbers=db_model.issue_numbers or []
        )

        # Set entity ID and timestamps from database
        commit.id = db_model.id
        commit.created_at = db_model.created_at
        commit.updated_at = db_model.updated_at

        # Set analysis results
        commit.summary = db_model.summary
        commit.tags = db_model.tags or []
        commit.sentiment_score = db_model.sentiment_score
        commit.embedding_id = db_model.embedding_id
        commit.analyzed_at = db_model.analyzed_at

        return commit
