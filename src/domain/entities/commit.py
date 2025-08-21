from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

from src.domain.entities.base import ValueObject, Entity


@dataclass
class CommitHash(ValueObject):
    value: str

    def _validate(self):
        if not self.value or len(self.value) < 6:
            raise ValueError("Commit hash must be at least 6 characters long.")


@dataclass
class FileChange(ValueObject):
    filename: str
    additions: int
    deletions: int
    status: str

    def _validate(self):
        if self.additions < 0 or self.deletions < 0:
            raise ValueError("Additions and deletions must be non-negative")
        if self.status not in ['added', 'modified', 'deleted', 'renamed']:
            raise ValueError("Invalid file status")


@dataclass
class CommitMetrics(ValueObject):
    """Value object for commit metrics"""
    total_lines_changed: int
    files_count: int
    complexity_score: Optional[float] = None
    impact_score: Optional[float] = None



class Commit(Entity):
    def __init__(
        self,
        commit_hash: CommitHash,
        author_email: str,
        author_name: str,
        message: str,
        timestamp: datetime,
        branch: str,
        project: str,
        files_changed: List[FileChange],
        issue_numbers: List[str] = None
    ):
        super().__init__()
        self.commit_hash: CommitHash = commit_hash
        self.author_email: str = author_email
        self.author_name: str = author_name
        self.message: str = message
        self.timestamp: datetime = timestamp
        self.branch: str = branch
        self.project: str = project
        self.files_changed: List[FileChange] = files_changed or []
        self.issue_numbers: List[str] = issue_numbers or []

        # AI responses
        self.summary: Optional[str] = None
        self.tags: List[str] = []
        self.sentiment_score: Optional[float] = None
        self.embedding_id: Optional[str] = None
        self.analyzed_at: Optional[datetime] = None

        # Computed metrics
        self._metrics: Optional[CommitMetrics] = None

    @property
    def metrics(self) -> CommitMetrics:
        """Calculate and cache commit metrics"""
        if self._metrics is None:
            total_changes = sum(f.additions + f.deletions for f in self.files_changed)
            self._metrics = CommitMetrics(
                total_lines_changed=total_changes,
                files_count=len(self.files_changed),
                complexity_score=self._calculate_complexity(),
                impact_score=self._calculate_impact()
            )
        return self._metrics

    def _calculate_complexity(self) -> float:
        """Business rule: calculate commit complexity"""
        if not self.files_changed:
            return 0.0

        # Complex = many files + many changes + certain file types
        file_score = min(len(self.files_changed) / 10, 1.0)  # Max 1.0 for 10+ files

        total_changes = sum(f.additions + f.deletions for f in self.files_changed)
        change_score = min(total_changes / 500, 1.0)  # Max 1.0 for 500+ lines

        # File type complexity
        complex_extensions = {'.sql', '.yaml', '.yml', '.json', '.xml'}
        type_score = sum(1 for f in self.files_changed
                         if any(f.filename.endswith(ext) for ext in complex_extensions)) / len(self.files_changed)

        return (file_score + change_score + type_score) / 3

    def _calculate_impact(self) -> float:
        """Business rule: calculate commit impact"""
        # High impact = core files, many deletions, config changes
        core_patterns = ['src/', 'lib/', 'core/', 'main', 'index']
        config_patterns = ['config', 'setting', '.env', 'docker', 'requirement']

        core_files = sum(1 for f in self.files_changed
                         if any(pattern in f.filename.lower() for pattern in core_patterns))
        config_files = sum(1 for f in self.files_changed
                           if any(pattern in f.filename.lower() for pattern in config_patterns))

        core_score = min(core_files / len(self.files_changed), 1.0) if self.files_changed else 0
        config_score = min(config_files / len(self.files_changed), 1.0) if self.files_changed else 0

        # Deletions often indicate refactoring/cleanup
        total_deletions = sum(f.deletions for f in self.files_changed)
        deletion_score = min(total_deletions / 200, 1.0)

        return (core_score + config_score + deletion_score) / 3

    def is_hotfix(self) -> bool:
        """Business rule: identify hotfix commits"""
        hotfix_keywords = ['hotfix', 'urgent', 'critical', 'emergency', 'fix bug', 'patch']
        message_lower = self.message.lower()
        return any(keyword in message_lower for keyword in hotfix_keywords)

    def is_feature(self) -> bool:
        """Business rule: identify feature commits"""
        feature_keywords = ['feat', 'feature', 'add', 'implement', 'new']
        message_lower = self.message.lower()
        return any(keyword in message_lower for keyword in feature_keywords)

    def mark_as_analyzed(self, summary: str, tags: List[str], sentiment: float):
        """Mark commit as analyzed with results"""
        self.summary = summary
        self.tags = tags
        self.sentiment_score = sentiment
        self.analyzed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def to_dict(self):
        """Convert commit to dictionary representation"""
        return {
            "id": self.id,
            "commit_hash": self.commit_hash.value,
            "author_email": self.author_email,
            "author_name": self.author_name,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "branch": self.branch,
            "project": self.project,
            "files_changed": [f.__dict__ for f in self.files_changed],
            "issue_numbers": self.issue_numbers,
            "summary": self.summary,
            "tags": self.tags,
            "sentiment_score": self.sentiment_score,
            "embedding_id": self.embedding_id,
            "analyzed_at": self.analyzed_at.isoformat() if self.analyzed_at else None,
            "metrics": {
                "total_lines_changed": self.metrics.total_lines_changed,
                "files_count": self.metrics.files_count,
                "complexity_score": self.metrics.complexity_score,
                "impact_score": self.metrics.impact_score
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }