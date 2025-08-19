    # Git AI Analytics - Full Implementation
    
    ## Project Structure
    ```
    git-ai-analytics/
    ├── src/
    │   ├── domain/                    # Domain Layer
    │   │   ├── entities/
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py
    │   │   │   ├── commit.py
    │   │   │   ├── developer.py
    │   │   │   └── analysis.py
    │   │   ├── events/
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py
    │   │   │   └── commit_events.py
    │   │   ├── repositories/
    │   │   │   ├── __init__.py
    │   │   │   ├── commit_repository.py
    │   │   │   └── analysis_repository.py
    │   │   └── services/
    │   │       ├── __init__.py
    │   │       └── ai_analyzer.py
    │   ├── application/               # Application Layer
    │   │   ├── use_cases/
    │   │   │   ├── __init__.py
    │   │   │   ├── process_commit.py
    │   │   │   ├── analyze_commit.py
    │   │   │   └── search_commits.py
    │   │   ├── dto/
    │   │   │   ├── __init__.py
    │   │   │   ├── commit_dto.py
    │   │   │   └── webhook_dto.py
    │   │   ├── services/
    │   │   │   ├── __init__.py
    │   │   │   └── event_dispatcher.py
    │   │   └── interfaces/
    │   │       ├── __init__.py
    │   │       ├── cache_interface.py
    │   │       ├── ai_interface.py
    │   │       └── messaging_interface.py
    │   ├── infrastructure/           # Infrastructure Layer
    │   │   ├── persistence/
    │   │   │   ├── __init__.py
    │   │   │   ├── models/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── commit_model.py
    │   │   │   ├── repositories/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── sqlite_commit_repository.py
    │   │   │   └── database.py
    │   │   ├── ai/
    │   │   │   ├── __init__.py
    │   │   │   ├── ollama_service.py
    │   │   │   └── prompts/
    │   │   │       └── commit_prompts.py
    │   │   ├── cache/
    │   │   │   ├── __init__.py
    │   │   │   ├── memory_cache.py
    │   │   │   └── redis_cache.py
    │   │   ├── messaging/
    │   │   │   ├── __init__.py
    │   │   │   ├── celery_config.py
    │   │   │   └── tasks/
    │   │   │       ├── __init__.py
    │   │   │       └── analysis_tasks.py
    │   │   └── events/
    │   │       ├── __init__.py
    │   │       └── event_bus.py
    │   ├── presentation/             # Presentation Layer
    │   │   ├── api/
    │   │   │   ├── __init__.py
    │   │   │   ├── v1/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── webhooks.py
    │   │   │   │   ├── commits.py
    │   │   │   │   └── chat.py
    │   │   │   └── dependencies.py
    │   │   └── websocket/
    │   │       ├── __init__.py
    │   │       └── handlers.py
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── config.py
    │   │   └── logging.py
    │   └── main.py
    ├── configs/
    │   ├── default.yaml
    │   ├── development.yaml
    │   └── production.yaml
    ├── tests/
    ├── migrations/
    ├── docker/
    │   ├── Dockerfile
    │   └── docker-compose.yml
    ├── requirements/
    │   ├── base.txt
    │   ├── development.txt
    │   └── production.txt
    ├── .env.example
    └── README.md
    ```
    
    ## 1. Domain Layer
    
    ### src/domain/entities/base.py
    ```python
    from abc import ABC
    from datetime import datetime
    from typing import Any
    from uuid import uuid4, UUID
    from dataclasses import dataclass, field
    
    @dataclass
    class ValueObject(ABC):
        """Base class for value objects"""
        
        def __post_init__(self):
            self._validate()
        
        def _validate(self):
            """Override to add validation logic"""
            pass
    
    class Entity(ABC):
        """Base class for entities"""
        
        def __init__(self):
            self.id: str = str(uuid4())
            self.created_at: datetime = datetime.utcnow()
            self.updated_at: datetime = datetime.utcnow()
        
        def __eq__(self, other):
            if not isinstance(other, self.__class__):
                return False
            return self.id == other.id
        
        def __hash__(self):
            return hash(self.id)
    ```
    
    ### src/domain/entities/commit.py
    ```python
    from dataclasses import dataclass
    from datetime import datetime
    from typing import List, Optional
    from .base import Entity, ValueObject
    
    @dataclass
    class CommitHash(ValueObject):
        """Value object for commit hash"""
        value: str
        
        def _validate(self):
            if not self.value or len(self.value) < 6:
                raise ValueError("Invalid commit hash")
    
    @dataclass
    class FileChange(ValueObject):
        """Value object for file changes"""
        filename: str
        additions: int
        deletions: int
        status: str  # added, modified, deleted, renamed
        
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
        """Commit domain entity"""
        
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
            self.commit_hash = commit_hash
            self.author_email = author_email
            self.author_name = author_name
            self.message = message
            self.timestamp = timestamp
            self.branch = branch
            self.project = project
            self.files_changed = files_changed or []
            self.issue_numbers = issue_numbers or []
            
            # Analysis results (populated later)
            self.summary: Optional[str] = None
            self.tags: List[str] = []
            self.sentiment_score: Optional[float] = None
            self.embedding_id: Optional[str] = None
            self.analyzed_at: Optional[datetime] = None
            
            # Computed properties
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
    ```
    
    ### src/domain/entities/analysis.py
    ```python
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, List, Optional
    from .base import Entity, ValueObject
    
    @dataclass
    class AnalysisResult(ValueObject):
        """Value object for AI analysis results"""
        summary: str
        tags: List[str]
        sentiment_score: float
        confidence_score: float
        extracted_entities: List[str]
        
        def _validate(self):
            if not (-1 <= self.sentiment_score <= 1):
                raise ValueError("Sentiment score must be between -1 and 1")
            if not (0 <= self.confidence_score <= 1):
                raise ValueError("Confidence score must be between 0 and 1")
    
    class Analysis(Entity):
        """Analysis domain entity"""
        
        def __init__(
            self,
            commit_id: str,
            model_name: str,
            model_version: str,
            prompt_version: str,
            result: AnalysisResult,
            processing_time_ms: int,
            tokens_used: int = 0
        ):
            super().__init__()
            self.commit_id = commit_id
            self.model_name = model_name
            self.model_version = model_version
            self.prompt_version = prompt_version
            self.result = result
            self.processing_time_ms = processing_time_ms
            self.tokens_used = tokens_used
            self.status = "completed"
        
        def is_high_confidence(self) -> bool:
            """Check if analysis has high confidence"""
            return self.result.confidence_score >= 0.8
    ```
    
    ### src/domain/events/base.py
    ```python
    from abc import ABC
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, Any
    import json
    from uuid import uuid4
    
    @dataclass
    class DomainEvent(ABC):
        """Base class for domain events"""
        event_id: str = None
        occurred_at: datetime = None
        
        def __post_init__(self):
            if self.event_id is None:
                self.event_id = str(uuid4())
            if self.occurred_at is None:
                self.occurred_at = datetime.utcnow()
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert event to dictionary"""
            return {
                'event_type': self.__class__.__name__,
                'event_id': self.event_id,
                'occurred_at': self.occurred_at.isoformat(),
                'data': self._get_event_data()
            }
        
        def to_json(self) -> str:
            """Convert event to JSON string"""
            return json.dumps(self.to_dict(), default=str)
        
        def _get_event_data(self) -> Dict[str, Any]:
            """Override to provide event-specific data"""
            return {}
    ```
    
    ### src/domain/events/commit_events.py
    ```python
    from dataclasses import dataclass
    from typing import Dict, Any, List
    from .base import DomainEvent
    
    @dataclass
    class CommitReceivedEvent(DomainEvent):
        """Event fired when a new commit is received"""
        commit_id: str
        commit_hash: str
        project: str
        author: str
        branch: str
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'commit_hash': self.commit_hash,
                'project': self.project,
                'author': self.author,
                'branch': self.branch
            }
    
    @dataclass 
    class CommitAnalysisStartedEvent(DomainEvent):
        """Event fired when commit analysis starts"""
        commit_id: str
        model_name: str
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'model_name': self.model_name
            }
    
    @dataclass
    class CommitAnalysisCompletedEvent(DomainEvent):
        """Event fired when commit analysis completes"""
        commit_id: str
        analysis_id: str
        summary: str
        tags: List[str]
        processing_time_ms: int
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'analysis_id': self.analysis_id,
                'summary': self.summary,
                'tags': self.tags,
                'processing_time_ms': self.processing_time_ms
            }
    
    @dataclass
    class CommitAnalysisFailedEvent(DomainEvent):
        """Event fired when commit analysis fails"""
        commit_id: str
        error_message: str
        retry_count: int
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'error_message': self.error_message,
                'retry_count': self.retry_count
            }
    ```
    
    ### src/domain/repositories/commit_repository.py
    ```python
    from abc import ABC, abstractmethod
    from typing import List, Optional, Dict, Any
    from datetime import datetime
    from ..entities.commit import Commit
    
    class ICommitRepository(ABC):
        """Repository interface for Commit entity"""
        
        @abstractmethod
        async def save(self, commit: Commit) -> Commit:
            """Save a commit"""
            pass
        
        @abstractmethod
        async def find_by_id(self, commit_id: str) -> Optional[Commit]:
            """Find commit by ID"""
            pass
        
        @abstractmethod
        async def find_by_hash(self, commit_hash: str) -> Optional[Commit]:
            """Find commit by hash"""
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
    ```
    
    ### src/domain/services/ai_analyzer.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, List
    from ..entities.commit import Commit
    from ..entities.analysis import AnalysisResult
    
    class IAIAnalyzer(ABC):
        """Domain service interface for AI analysis"""
        
        @abstractmethod
        async def analyze_commit(self, commit: Commit) -> AnalysisResult:
            """Analyze a commit using AI"""
            pass
        
        @abstractmethod
        async def generate_summary(self, commits: List[Commit]) -> str:
            """Generate summary for multiple commits"""
            pass
        
        @abstractmethod
        async def answer_question(
            self, 
            question: str, 
            context_commits: List[Commit]
        ) -> str:
            """Answer question about commits"""
            pass
        
        @abstractmethod
        async def extract_entities(self, text: str) -> List[str]:
            """Extract named entities from text"""
            pass
        
        @abstractmethod
        def get_model_info(self) -> Dict[str, str]:
            """Get current model information"""
            pass
    ```
    
    ## 2. Application Layer
    
    ### src/application/dto/commit_dto.py
    ```python
    from pydantic import BaseModel, Field, validator
    from datetime import datetime
    from typing import List, Optional
    
    class FileChangeDto(BaseModel):
        """DTO for file changes"""
        filename: str = Field(..., min_length=1)
        additions: int = Field(..., ge=0)
        deletions: int = Field(..., ge=0)
        status: str = Field(..., regex="^(added|modified|deleted|renamed)$")
    
    class CommitDto(BaseModel):
        """DTO for commit data"""
        commit_hash: str = Field(..., min_length=6)
        author_email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
        author_name: str = Field(..., min_length=1)
        message: str = Field(..., min_length=1)
        timestamp: datetime
        branch: str = Field(..., min_length=1)
        project: str = Field(..., min_length=1)
        files_changed: List[FileChangeDto] = []
        issue_numbers: List[str] = []
    
    class CommitResponseDto(BaseModel):
        """DTO for commit response"""
        id: str
        commit_hash: str
        author_email: str
        author_name: str
        message: str
        timestamp: datetime
        branch: str
        project: str
        files_count: int
        total_lines_changed: int
        complexity_score: Optional[float] = None
        summary: Optional[str] = None
        tags: List[str] = []
        sentiment_score: Optional[float] = None
        analyzed_at: Optional[datetime] = None
        created_at: datetime
    
    class CommitSearchDto(BaseModel):
        """DTO for commit search"""
        query: Optional[str] = None
        project: Optional[str] = None
        author: Optional[str] = None
        branch: Optional[str] = None
        start_date: Optional[datetime] = None
        end_date: Optional[datetime] = None
        tags: List[str] = []
        limit: int = Field(50, ge=1, le=200)
        offset: int = Field(0, ge=0)
    
    class ChatRequestDto(BaseModel):
        """DTO for chat requests"""
        question: str = Field(..., min_length=1, max_length=1000)
        context_project: Optional[str] = None
        context_author: Optional[str] = None
        context_days: int = Field(30, ge=1, le=365)
    
    class ChatResponseDto(BaseModel):
        """DTO for chat responses"""
        answer: str
        context_commits_count: int
        processing_time_ms: int
        model_used: str
    ```
    
    ### src/application/dto/webhook_dto.py
    ```python
    from pydantic import BaseModel
    from typing import List, Optional
    from datetime import datetime
    
    class WebhookAuthorDto(BaseModel):
        """DTO for webhook author data"""
        name: str
        email: str
        username: Optional[str] = None
    
    class WebhookCommitDto(BaseModel):
        """DTO for webhook commit data"""
        id: str
        message: str
        timestamp: datetime
        author: WebhookAuthorDto
        added: List[str] = []
        modified: List[str] = []
        removed: List[str] = []
        url: Optional[str] = None
    
    class WebhookRepositoryDto(BaseModel):
        """DTO for webhook repository data"""
        name: str
        full_name: str
        url: str
        default_branch: str
    
    class WebhookPayloadDto(BaseModel):
        """DTO for webhook payload"""
        ref: str  # refs/heads/main
        repository: WebhookRepositoryDto
        commits: List[WebhookCommitDto]
        pusher: WebhookAuthorDto
        head_commit: Optional[WebhookCommitDto] = None
    ```
    
    ### src/application/use_cases/process_commit.py
    ```python
    from typing import Optional
    from datetime import datetime
    from ..dto.commit_dto import CommitDto, CommitResponseDto
    from ..services.event_dispatcher import IEventDispatcher
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.entities.commit import Commit, CommitHash, FileChange
    from ...domain.events.commit_events import CommitReceivedEvent
    from ...application.interfaces.cache_interface import ICacheService
    from ...application.interfaces.messaging_interface import IMessageQueue
    
    class ProcessCommitUseCase:
        """Use case for processing incoming commits"""
        
        def __init__(
            self,
            commit_repository: ICommitRepository,
            event_dispatcher: IEventDispatcher,
            message_queue: IMessageQueue,
            cache_service: ICacheService
        ):
            self.commit_repository = commit_repository
            self.event_dispatcher = event_dispatcher
            self.message_queue = message_queue
            self.cache_service = cache_service
        
        async def execute(self, commit_dto: CommitDto) -> CommitResponseDto:
            """Process a new commit"""
            
            # Check if commit already exists (idempotency)
            existing = await self.commit_repository.find_by_hash(commit_dto.commit_hash)
            if existing:
                return self._to_response_dto(existing)
            
            # Create domain entity
            commit = self._to_domain(commit_dto)
            
            # Save to repository
            saved_commit = await self.commit_repository.save(commit)
            
            # Dispatch domain event
            event = CommitReceivedEvent(
                commit_id=saved_commit.id,
                commit_hash=saved_commit.commit_hash.value,
                project=saved_commit.project,
                author=saved_commit.author_email,
                branch=saved_commit.branch
            )
            await self.event_dispatcher.dispatch(event)
            
            # Queue for async analysis
            await self.message_queue.enqueue(
                "analyze_commit",
                {"commit_id": saved_commit.id}
            )
            
            # Invalidate relevant caches
            await self.cache_service.invalidate_pattern(f"commits:project:{commit.project}:*")
            await self.cache_service.invalidate_pattern(f"commits:author:{commit.author_email}:*")
            
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
    ```
    
    ### src/application/use_cases/search_commits.py
    ```python
    from typing import List
    from ..dto.commit_dto import CommitSearchDto, CommitResponseDto
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...application.interfaces.cache_interface import ICacheService
    import hashlib
    import json
    
    class SearchCommitsUseCase:
        """Use case for searching commits"""
        
        def __init__(
            self,
            commit_repository: ICommitRepository,
            cache_service: ICacheService
        ):
            self.commit_repository = commit_repository
            self.cache_service = cache_service
        
        async def execute(self, search_dto: CommitSearchDto) -> List[CommitResponseDto]:
            """Search commits with caching"""
            
            # Generate cache key
            cache_key = self._generate_cache_key(search_dto)
            
            # Try cache first
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                return cached_result
            
            # Search in repository
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
                limit=search_dto.limit
            )
            
            # Convert to DTOs
            result = [self._to_response_dto(commit) for commit in commits]
            
            # Cache result for 5 minutes
            await self.cache_service.set(cache_key, result, ttl=300)
            
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
    ```
    
    ### src/application/use_cases/chat_with_ai.py
    ```python
    from typing import List
    from datetime import datetime, timedelta
    import time
    from ..dto.commit_dto import ChatRequestDto, ChatResponseDto
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.services.ai_analyzer import IAIAnalyzer
    from ...application.interfaces.cache_interface import ICacheService
    import hashlib
    
    class ChatWithAIUseCase:
        """Use case for chatting with AI about commits"""
        
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
            """Answer question about commits using AI"""
            
            start_time = time.time()
            
            # Generate cache key
            cache_key = self._generate_cache_key(chat_dto)
            
            # Try cache first
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                return cached_result
            
            # Get context commits
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=chat_dto.context_days)
            
            context_commits = await self.commit_repository.find_by_project(
                project=chat_dto.context_project,
                start_date=start_date,
                end_date=end_date,
                limit=100
            )
            
            if chat_dto.context_author:
                author_commits = await self.commit_repository.find_by_author(
                    author_email=chat_dto.context_author,
                    project=chat_dto.context_project,
                    start_date=start_date,
                    end_date=end_date,
                    limit=50
                )
                context_commits.extend(author_commits)
            
            # Remove duplicates
            unique_commits = {commit.id: commit for commit in context_commits}
            context_commits = list(unique_commits.values())
            
            # Ask AI
            answer = await self.ai_analyzer.answer_question(
                question=chat_dto.question,
                context_commits=context_commits
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            result = ChatResponseDto(
                answer=answer,
                context_commits_count=len(context_commits),
                processing_time_ms=processing_time_ms,
                model_used=self.ai_analyzer.get_model_info()['name']
            )
            
            # Cache result for 10 minutes
            await self.cache_service.set(cache_key, result, ttl=600)
            
            return result
        
        def _generate_cache_key(self, chat_dto: ChatRequestDto) -> str:
            """Generate cache key for chat request"""
            chat_hash = hashlib.md5(
                f"{chat_dto.question}:{chat_dto.context_project}:{chat_dto.context_author}:{chat_dto.context_days}".encode()
            ).hexdigest()
            return f"chat:response:{chat_hash}"

Git AI Analytics - Complete Implementation

Code ∙ Version 2 

class ChatWithAIUseCase: """Use case for chatting with AI about commits""" def \_\_init\_\_( self, commit\_repository: ICommitRepository, ai\_analyzer: IAIAnalyzer, cache\_service: ICacheService ): se

    # Git AI Analytics - Full Implementation
    
    ## Project Structure
    ```
    git-ai-analytics/
    ├── src/
    │   ├── domain/                    # Domain Layer
    │   │   ├── entities/
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py
    │   │   │   ├── commit.py
    │   │   │   ├── developer.py
    │   │   │   └── analysis.py
    │   │   ├── events/
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py
    │   │   │   └── commit_events.py
    │   │   ├── repositories/
    │   │   │   ├── __init__.py
    │   │   │   ├── commit_repository.py
    │   │   │   └── analysis_repository.py
    │   │   └── services/
    │   │       ├── __init__.py
    │   │       └── ai_analyzer.py
    │   ├── application/               # Application Layer
    │   │   ├── use_cases/
    │   │   │   ├── __init__.py
    │   │   │   ├── process_commit.py
    │   │   │   ├── analyze_commit.py
    │   │   │   └── search_commits.py
    │   │   ├── dto/
    │   │   │   ├── __init__.py
    │   │   │   ├── commit_dto.py
    │   │   │   └── webhook_dto.py
    │   │   ├── services/
    │   │   │   ├── __init__.py
    │   │   │   └── event_dispatcher.py
    │   │   └── interfaces/
    │   │       ├── __init__.py
    │   │       ├── cache_interface.py
    │   │       ├── ai_interface.py
    │   │       └── messaging_interface.py
    │   ├── infrastructure/           # Infrastructure Layer
    │   │   ├── persistence/
    │   │   │   ├── __init__.py
    │   │   │   ├── models/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── commit_model.py
    │   │   │   ├── repositories/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── sqlite_commit_repository.py
    │   │   │   └── database.py
    │   │   ├── ai/
    │   │   │   ├── __init__.py
    │   │   │   ├── ollama_service.py
    │   │   │   └── prompts/
    │   │   │       └── commit_prompts.py
    │   │   ├── cache/
    │   │   │   ├── __init__.py
    │   │   │   ├── memory_cache.py
    │   │   │   └── redis_cache.py
    │   │   ├── messaging/
    │   │   │   ├── __init__.py
    │   │   │   ├── celery_config.py
    │   │   │   └── tasks/
    │   │   │       ├── __init__.py
    │   │   │       └── analysis_tasks.py
    │   │   └── events/
    │   │       ├── __init__.py
    │   │       └── event_bus.py
    │   ├── presentation/             # Presentation Layer
    │   │   ├── api/
    │   │   │   ├── __init__.py
    │   │   │   ├── v1/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── webhooks.py
    │   │   │   │   ├── commits.py
    │   │   │   │   └── chat.py
    │   │   │   └── dependencies.py
    │   │   └── websocket/
    │   │       ├── __init__.py
    │   │       └── handlers.py
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── config.py
    │   │   └── logging.py
    │   └── main.py
    ├── configs/
    │   ├── default.yaml
    │   ├── development.yaml
    │   └── production.yaml
    ├── tests/
    ├── migrations/
    ├── docker/
    │   ├── Dockerfile
    │   └── docker-compose.yml
    ├── requirements/
    │   ├── base.txt
    │   ├── development.txt
    │   └── production.txt
    ├── .env.example
    └── README.md
    ```
    
    ## 1. Domain Layer
    
    ### src/domain/entities/base.py
    ```python
    from abc import ABC
    from datetime import datetime
    from typing import Any
    from uuid import uuid4, UUID
    from dataclasses import dataclass, field
    
    @dataclass
    class ValueObject(ABC):
        """Base class for value objects"""
        
        def __post_init__(self):
            self._validate()
        
        def _validate(self):
            """Override to add validation logic"""
            pass
    
    class Entity(ABC):
        """Base class for entities"""
        
        def __init__(self):
            self.id: str = str(uuid4())
            self.created_at: datetime = datetime.utcnow()
            self.updated_at: datetime = datetime.utcnow()
        
        def __eq__(self, other):
            if not isinstance(other, self.__class__):
                return False
            return self.id == other.id
        
        def __hash__(self):
            return hash(self.id)
    ```
    
    ### src/domain/entities/commit.py
    ```python
    from dataclasses import dataclass
    from datetime import datetime
    from typing import List, Optional
    from .base import Entity, ValueObject
    
    @dataclass
    class CommitHash(ValueObject):
        """Value object for commit hash"""
        value: str
        
        def _validate(self):
            if not self.value or len(self.value) < 6:
                raise ValueError("Invalid commit hash")
    
    @dataclass
    class FileChange(ValueObject):
        """Value object for file changes"""
        filename: str
        additions: int
        deletions: int
        status: str  # added, modified, deleted, renamed
        
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
        """Commit domain entity"""
        
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
            self.commit_hash = commit_hash
            self.author_email = author_email
            self.author_name = author_name
            self.message = message
            self.timestamp = timestamp
            self.branch = branch
            self.project = project
            self.files_changed = files_changed or []
            self.issue_numbers = issue_numbers or []
            
            # Analysis results (populated later)
            self.summary: Optional[str] = None
            self.tags: List[str] = []
            self.sentiment_score: Optional[float] = None
            self.embedding_id: Optional[str] = None
            self.analyzed_at: Optional[datetime] = None
            
            # Computed properties
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
    ```
    
    ### src/domain/entities/analysis.py
    ```python
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, List, Optional
    from .base import Entity, ValueObject
    
    @dataclass
    class AnalysisResult(ValueObject):
        """Value object for AI analysis results"""
        summary: str
        tags: List[str]
        sentiment_score: float
        confidence_score: float
        extracted_entities: List[str]
        
        def _validate(self):
            if not (-1 <= self.sentiment_score <= 1):
                raise ValueError("Sentiment score must be between -1 and 1")
            if not (0 <= self.confidence_score <= 1):
                raise ValueError("Confidence score must be between 0 and 1")
    
    class Analysis(Entity):
        """Analysis domain entity"""
        
        def __init__(
            self,
            commit_id: str,
            model_name: str,
            model_version: str,
            prompt_version: str,
            result: AnalysisResult,
            processing_time_ms: int,
            tokens_used: int = 0
        ):
            super().__init__()
            self.commit_id = commit_id
            self.model_name = model_name
            self.model_version = model_version
            self.prompt_version = prompt_version
            self.result = result
            self.processing_time_ms = processing_time_ms
            self.tokens_used = tokens_used
            self.status = "completed"
        
        def is_high_confidence(self) -> bool:
            """Check if analysis has high confidence"""
            return self.result.confidence_score >= 0.8
    ```
    
    ### src/domain/events/base.py
    ```python
    from abc import ABC
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, Any
    import json
    from uuid import uuid4
    
    @dataclass
    class DomainEvent(ABC):
        """Base class for domain events"""
        event_id: str = None
        occurred_at: datetime = None
        
        def __post_init__(self):
            if self.event_id is None:
                self.event_id = str(uuid4())
            if self.occurred_at is None:
                self.occurred_at = datetime.utcnow()
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert event to dictionary"""
            return {
                'event_type': self.__class__.__name__,
                'event_id': self.event_id,
                'occurred_at': self.occurred_at.isoformat(),
                'data': self._get_event_data()
            }
        
        def to_json(self) -> str:
            """Convert event to JSON string"""
            return json.dumps(self.to_dict(), default=str)
        
        def _get_event_data(self) -> Dict[str, Any]:
            """Override to provide event-specific data"""
            return {}
    ```
    
    ### src/domain/events/commit_events.py
    ```python
    from dataclasses import dataclass
    from typing import Dict, Any, List
    from .base import DomainEvent
    
    @dataclass
    class CommitReceivedEvent(DomainEvent):
        """Event fired when a new commit is received"""
        commit_id: str
        commit_hash: str
        project: str
        author: str
        branch: str
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'commit_hash': self.commit_hash,
                'project': self.project,
                'author': self.author,
                'branch': self.branch
            }
    
    @dataclass 
    class CommitAnalysisStartedEvent(DomainEvent):
        """Event fired when commit analysis starts"""
        commit_id: str
        model_name: str
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'model_name': self.model_name
            }
    
    @dataclass
    class CommitAnalysisCompletedEvent(DomainEvent):
        """Event fired when commit analysis completes"""
        commit_id: str
        analysis_id: str
        summary: str
        tags: List[str]
        processing_time_ms: int
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'analysis_id': self.analysis_id,
                'summary': self.summary,
                'tags': self.tags,
                'processing_time_ms': self.processing_time_ms
            }
    
    @dataclass
    class CommitAnalysisFailedEvent(DomainEvent):
        """Event fired when commit analysis fails"""
        commit_id: str
        error_message: str
        retry_count: int
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'error_message': self.error_message,
                'retry_count': self.retry_count
            }
    ```
    
    ### src/domain/repositories/commit_repository.py
    ```python
    from abc import ABC, abstractmethod
    from typing import List, Optional, Dict, Any
    from datetime import datetime
    from ..entities.commit import Commit
    
    class ICommitRepository(ABC):
        """Repository interface for Commit entity"""
        
        @abstractmethod
        async def save(self, commit: Commit) -> Commit:
            """Save a commit"""
            pass
        
        @abstractmethod
        async def find_by_id(self, commit_id: str) -> Optional[Commit]:
            """Find commit by ID"""
            pass
        
        @abstractmethod
        async def find_by_hash(self, commit_hash: str) -> Optional[Commit]:
            """Find commit by hash"""
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
    ```
    
    ### src/domain/services/ai_analyzer.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, List
    from ..entities.commit import Commit
    from ..entities.analysis import AnalysisResult
    
    class IAIAnalyzer(ABC):
        """Domain service interface for AI analysis"""
        
        @abstractmethod
        async def analyze_commit(self, commit: Commit) -> AnalysisResult:
            """Analyze a commit using AI"""
            pass
        
        @abstractmethod
        async def generate_summary(self, commits: List[Commit]) -> str:
            """Generate summary for multiple commits"""
            pass
        
        @abstractmethod
        async def answer_question(
            self, 
            question: str, 
            context_commits: List[Commit]
        ) -> str:
            """Answer question about commits"""
            pass
        
        @abstractmethod
        async def extract_entities(self, text: str) -> List[str]:
            """Extract named entities from text"""
            pass
        
        @abstractmethod
        def get_model_info(self) -> Dict[str, str]:
            """Get current model information"""
            pass
    ```
    
    ## 2. Application Layer
    
    ### src/application/dto/commit_dto.py
    ```python
    from pydantic import BaseModel, Field, validator
    from datetime import datetime
    from typing import List, Optional
    
    class FileChangeDto(BaseModel):
        """DTO for file changes"""
        filename: str = Field(..., min_length=1)
        additions: int = Field(..., ge=0)
        deletions: int = Field(..., ge=0)
        status: str = Field(..., regex="^(added|modified|deleted|renamed)$")
    
    class CommitDto(BaseModel):
        """DTO for commit data"""
        commit_hash: str = Field(..., min_length=6)
        author_email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
        author_name: str = Field(..., min_length=1)
        message: str = Field(..., min_length=1)
        timestamp: datetime
        branch: str = Field(..., min_length=1)
        project: str = Field(..., min_length=1)
        files_changed: List[FileChangeDto] = []
        issue_numbers: List[str] = []
    
    class CommitResponseDto(BaseModel):
        """DTO for commit response"""
        id: str
        commit_hash: str
        author_email: str
        author_name: str
        message: str
        timestamp: datetime
        branch: str
        project: str
        files_count: int
        total_lines_changed: int
        complexity_score: Optional[float] = None
        summary: Optional[str] = None
        tags: List[str] = []
        sentiment_score: Optional[float] = None
        analyzed_at: Optional[datetime] = None
        created_at: datetime
    
    class CommitSearchDto(BaseModel):
        """DTO for commit search"""
        query: Optional[str] = None
        project: Optional[str] = None
        author: Optional[str] = None
        branch: Optional[str] = None
        start_date: Optional[datetime] = None
        end_date: Optional[datetime] = None
        tags: List[str] = []
        limit: int = Field(50, ge=1, le=200)
        offset: int = Field(0, ge=0)
    
    class ChatRequestDto(BaseModel):
        """DTO for chat requests"""
        question: str = Field(..., min_length=1, max_length=1000)
        context_project: Optional[str] = None
        context_author: Optional[str] = None
        context_days: int = Field(30, ge=1, le=365)
    
    class ChatResponseDto(BaseModel):
        """DTO for chat responses"""
        answer: str
        context_commits_count: int
        processing_time_ms: int
        model_used: str
    ```
    
    ### src/application/dto/webhook_dto.py
    ```python
    from pydantic import BaseModel
    from typing import List, Optional
    from datetime import datetime
    
    class WebhookAuthorDto(BaseModel):
        """DTO for webhook author data"""
        name: str
        email: str
        username: Optional[str] = None
    
    class WebhookCommitDto(BaseModel):
        """DTO for webhook commit data"""
        id: str
        message: str
        timestamp: datetime
        author: WebhookAuthorDto
        added: List[str] = []
        modified: List[str] = []
        removed: List[str] = []
        url: Optional[str] = None
    
    class WebhookRepositoryDto(BaseModel):
        """DTO for webhook repository data"""
        name: str
        full_name: str
        url: str
        default_branch: str
    
    class WebhookPayloadDto(BaseModel):
        """DTO for webhook payload"""
        ref: str  # refs/heads/main
        repository: WebhookRepositoryDto
        commits: List[WebhookCommitDto]
        pusher: WebhookAuthorDto
        head_commit: Optional[WebhookCommitDto] = None
    ```
    
    ### src/application/use_cases/process_commit.py
    ```python
    from typing import Optional
    from datetime import datetime
    from ..dto.commit_dto import CommitDto, CommitResponseDto
    from ..services.event_dispatcher import IEventDispatcher
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.entities.commit import Commit, CommitHash, FileChange
    from ...domain.events.commit_events import CommitReceivedEvent
    from ...application.interfaces.cache_interface import ICacheService
    from ...application.interfaces.messaging_interface import IMessageQueue
    
    class ProcessCommitUseCase:
        """Use case for processing incoming commits"""
        
        def __init__(
            self,
            commit_repository: ICommitRepository,
            event_dispatcher: IEventDispatcher,
            message_queue: IMessageQueue,
            cache_service: ICacheService
        ):
            self.commit_repository = commit_repository
            self.event_dispatcher = event_dispatcher
            self.message_queue = message_queue
            self.cache_service = cache_service
        
        async def execute(self, commit_dto: CommitDto) -> CommitResponseDto:
            """Process a new commit"""
            
            # Check if commit already exists (idempotency)
            existing = await self.commit_repository.find_by_hash(commit_dto.commit_hash)
            if existing:
                return self._to_response_dto(existing)
            
            # Create domain entity
            commit = self._to_domain(commit_dto)
            
            # Save to repository
            saved_commit = await self.commit_repository.save(commit)
            
            # Dispatch domain event
            event = CommitReceivedEvent(
                commit_id=saved_commit.id,
                commit_hash=saved_commit.commit_hash.value,
                project=saved_commit.project,
                author=saved_commit.author_email,
                branch=saved_commit.branch
            )
            await self.event_dispatcher.dispatch(event)
            
            # Queue for async analysis
            await self.message_queue.enqueue(
                "analyze_commit",
                {"commit_id": saved_commit.id}
            )
            
            # Invalidate relevant caches
            await self.cache_service.invalidate_pattern(f"commits:project:{commit.project}:*")
            await self.cache_service.invalidate_pattern(f"commits:author:{commit.author_email}:*")
            
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
    ```
    
    ### src/application/use_cases/search_commits.py
    ```python
    from typing import List
    from ..dto.commit_dto import CommitSearchDto, CommitResponseDto
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...application.interfaces.cache_interface import ICacheService
    import hashlib
    import json
    
    class SearchCommitsUseCase:
        """Use case for searching commits"""
        
        def __init__(
            self,
            commit_repository: ICommitRepository,
            cache_service: ICacheService
        ):
            self.commit_repository = commit_repository
            self.cache_service = cache_service
        
        async def execute(self, search_dto: CommitSearchDto) -> List[CommitResponseDto]:
            """Search commits with caching"""
            
            # Generate cache key
            cache_key = self._generate_cache_key(search_dto)
            
            # Try cache first
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                return cached_result
            
            # Search in repository
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
                limit=search_dto.limit
            )
            
            # Convert to DTOs
            result = [self._to_response_dto(commit) for commit in commits]
            
            # Cache result for 5 minutes
            await self.cache_service.set(cache_key, result, ttl=300)
            
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
    ```
    
    ### src/application/use_cases/chat_with_ai.py
    ```python
    from typing import List
    from datetime import datetime, timedelta
    import time
    from ..dto.commit_dto import ChatRequestDto, ChatResponseDto
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.services.ai_analyzer import IAIAnalyzer
    from ...application.interfaces.cache_interface import ICacheService
    import hashlib
    
    class ChatWithAIUseCase:
        """Use case for chatting with AI about commits"""
        
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
            """Answer question about commits using AI"""
            
            start_time = time.time()
            
            # Generate cache key
            cache_key = self._generate_cache_key(chat_dto)
            
            # Try cache first
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                return cached_result
            
            # Get context commits
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=chat_dto.context_days)
            
            context_commits = await self.commit_repository.find_by_project(
                project=chat_dto.context_project,
                start_date=start_date,
                end_date=end_date,
                limit=100
            )
            
            if chat_dto.context_author:
                author_commits = await self.commit_repository.find_by_author(
                    author_email=chat_dto.context_author,
                    project=chat_dto.context_project,
                    start_date=start_date,
                    end_date=end_date,
                    limit=50
                )
                context_commits.extend(author_commits)
            
            # Remove duplicates
            unique_commits = {commit.id: commit for commit in context_commits}
            context_commits = list(unique_commits.values())
            
            # Ask AI
            answer = await self.ai_analyzer.answer_question(
                question=chat_dto.question,
                context_commits=context_commits
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            result = ChatResponseDto(
                answer=answer,
                context_commits_count=len(context_commits),
                processing_time_ms=processing_time_ms,
                model_used=self.ai_analyzer.get_model_info()['name']
            )
            
            # Cache result for 10 minutes
            await self.cache_service.set(cache_key, result, ttl=600)
            
            return result
        
        def _generate_cache_key(self, chat_dto: ChatRequestDto) -> str:
            """Generate cache key for chat request"""
            chat_hash = hashlib.md5(
                f"{chat_dto.question}:{chat_dto.context_project}:{chat_dto.context_author}:{chat_dto.context_days}".encode()
            ).hexdigest()
    ### src/application/services/event_dispatcher.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, List, Callable, Any
    from ...domain.events.base import DomainEvent
    import asyncio
    import logging
    
    logger = logging.getLogger(__name__)
    
    class IEventDispatcher(ABC):
        """Event dispatcher interface"""
        
        @abstractmethod
        async def dispatch(self, event: DomainEvent):
            """Dispatch event to all registered handlers"""
            pass
        
        @abstractmethod
        def register_handler(self, event_type: str, handler: Callable):
            """Register event handler"""
            pass
    
    class EventDispatcher(IEventDispatcher):
        """Implementation of event dispatcher"""
        
        def __init__(self):
            self.handlers: Dict[str, List[Callable]] = {}
        
        def register_handler(self, event_type: str, handler: Callable):
            """Register event handler"""
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
            logger.info(f"Registered handler for event type: {event_type}")
        
        async def dispatch(self, event: DomainEvent):
            """Dispatch event to all registered handlers"""
            event_type = event.__class__.__name__
            
            if event_type not in self.handlers:
                logger.debug(f"No handlers registered for event type: {event_type}")
                return
            
            logger.info(f"Dispatching event: {event_type} (ID: {event.event_id})")
            
            tasks = []
            for handler in self.handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        # Run sync handler in thread pool
                        loop = asyncio.get_event_loop()
                        tasks.append(loop.run_in_executor(None, handler, event))
                except Exception as e:
                    logger.error(f"Error preparing handler for {event_type}: {str(e)}")
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log any handler exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Handler {i} failed for {event_type}: {str(result)}")
    
    ### src/application/interfaces/cache_interface.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Any, Optional, List
    
    class ICacheService(ABC):
        """Interface for cache service"""
        
        @abstractmethod
        async def get(self, key: str) -> Optional[Any]:
            """Get value from cache"""
            pass
        
        @abstractmethod
        async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
            """Set value in cache with TTL in seconds"""
            pass
        
        @abstractmethod
        async def delete(self, key: str) -> bool:
            """Delete key from cache"""
            pass
        
        @abstractmethod
        async def exists(self, key: str) -> bool:
            """Check if key exists in cache"""
            pass
        
        @abstractmethod
        async def invalidate_pattern(self, pattern: str) -> int:
            """Invalidate all keys matching pattern (with * wildcard)"""
            pass
        
        @abstractmethod
        async def get_many(self, keys: List[str]) -> Dict[str, Any]:
            """Get multiple values from cache"""
            pass
        
        @abstractmethod
        async def set_many(self, data: Dict[str, Any], ttl: int = 3600) -> bool:
            """Set multiple values in cache"""
            pass
    
    ### src/application/interfaces/ai_interface.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, List, Optional
    from ...domain.entities.analysis import AnalysisResult
    
    class IAIService(ABC):
        """Interface for AI service"""
        
        @abstractmethod
        async def analyze_text(
            self,
            text: str,
            context: Optional[str] = None,
            temperature: float = 0.7
        ) -> AnalysisResult:
            """Analyze text and return structured results"""
            pass
        
        @abstractmethod
        async def generate_response(
            self,
            prompt: str,
            context: Optional[str] = None,
            max_tokens: int = 500
        ) -> str:
            """Generate text response"""
            pass
        
        @abstractmethod
        async def create_embedding(self, text: str) -> List[float]:
            """Create text embedding"""
            pass
        
        @abstractmethod
        async def health_check(self) -> bool:
            """Check if AI service is healthy"""
            pass
        
        @abstractmethod
        def get_model_info(self) -> Dict[str, str]:
            """Get model information"""
            pass
    
    ### src/application/interfaces/messaging_interface.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, Optional, Callable
    from enum import Enum
    
    class TaskStatus(Enum):
        PENDING = "PENDING"
        STARTED = "STARTED"
        SUCCESS = "SUCCESS"
        FAILURE = "FAILURE"
        RETRY = "RETRY"
    
    class IMessageQueue(ABC):
        """Interface for message queue"""
        
        @abstractmethod
        async def enqueue(
            self,
            task_name: str,
            payload: Dict[str, Any],
            delay_seconds: int = 0,
            priority: int = 5
        ) -> str:
            """Enqueue task for processing"""
            pass
        
        @abstractmethod
        async def get_task_status(self, task_id: str) -> TaskStatus:
            """Get task status"""
            pass
        
        @abstractmethod
        async def get_task_result(self, task_id: str) -> Optional[Any]:
            """Get task result"""
            pass
        
        @abstractmethod
        async def cancel_task(self, task_id: str) -> bool:
            """Cancel pending task"""
            pass
        
        @abstractmethod
        def register_task(self, task_name: str, handler: Callable):
            """Register task handler"""
            pass
    
    ## 3. Infrastructure Layer
    
    ### src/infrastructure/persistence/database.py
    ```python
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy import MetaData
    from contextlib import asynccontextmanager
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    # SQLAlchemy base
    Base = declarative_base()
    metadata = MetaData()
    
    # Global engine and session maker
    engine = None
    SessionLocal = None
    
    def get_database_url() -> str:
        """Get database URL from environment"""
        db_provider = os.getenv("DB_PROVIDER", "sqlite")
        
        if db_provider == "sqlite":
            db_path = os.getenv("DB_PATH", "data/git_analytics.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            return f"sqlite+aiosqlite:///{db_path}"
        
        elif db_provider == "postgresql":
            host = os.getenv("DB_HOST", "localhost")
            port = os.getenv("DB_PORT", "5432")
            name = os.getenv("DB_NAME", "git_analytics")
            user = os.getenv("DB_USER", "postgres")
            password = os.getenv("DB_PASSWORD", "")
            return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"
        
        else:
            raise ValueError(f"Unsupported database provider: {db_provider}")
    
    async def init_database():
        """Initialize database connection"""
        global engine, SessionLocal
        
        if engine is None:
            database_url = get_database_url()
            logger.info(f"Connecting to database: {database_url.split('://')[0]}://...")
            
            engine = create_async_engine(
                database_url,
                echo=os.getenv("DB_ECHO", "false").lower() == "true",
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            SessionLocal = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database initialized successfully")
    
    async def close_database():
        """Close database connection"""
        global engine
        if engine:
            await engine.dispose()
            logger.info("Database connection closed")
    
    @asynccontextmanager
    async def get_session() -> AsyncSession:
        """Get database session"""
        if SessionLocal is None:
            await init_database()
        
        async with SessionLocal() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    ### src/infrastructure/persistence/models/commit_model.py
    ```python
    from sqlalchemy import Column, String, DateTime, Text, Integer, Float, JSON
    from sqlalchemy.sql import func
    from ..database import Base
    
    class CommitModel(Base):
        """SQLAlchemy model for commits"""
        
        __tablename__ = "commits"
        
        # Primary fields
        id = Column(String(36), primary_key=True)
        commit_hash = Column(String(40), unique=True, nullable=False, index=True)
        author_email = Column(String(255), nullable=False, index=True)
        author_name = Column(String(255), nullable=False)
        message = Column(Text, nullable=False)
        timestamp = Column(DateTime, nullable=False, index=True)
        branch = Column(String(255), nullable=False, index=True)
        project = Column(String(255), nullable=False, index=True)
        
        # File changes (stored as JSON)
        files_changed = Column(JSON, nullable=False, default=list)
        issue_numbers = Column(JSON, nullable=False, default=list)
        
        # Metrics
        total_lines_changed = Column(Integer, default=0)
        files_count = Column(Integer, default=0)
        complexity_score = Column(Float)
        impact_score = Column(Float)
        
        # Analysis results
        summary = Column(Text)
        tags = Column(JSON, default=list)
        sentiment_score = Column(Float)
        embedding_id = Column(String(255))
        analyzed_at = Column(DateTime)
        
        # Timestamps
        created_at = Column(DateTime, default=func.now(), nullable=False)
        updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
        
        def __repr__(self):
            return f"<CommitModel(hash={self.commit_hash}, project={self.project})>"
    
    class AnalysisModel(Base):
        """SQLAlchemy model for analysis results"""
        
        __tablename__ = "analyses"
        
        id = Column(String(36), primary_key=True)
        commit_id = Column(String(36), nullable=False, index=True)
        
        # Model information
        model_name = Column(String(100), nullable=False)
        model_version = Column(String(50), nullable=False)
        prompt_version = Column(String(50), nullable=False)
        
        # Analysis results
        summary = Column(Text, nullable=False)
        tags = Column(JSON, default=list)
        sentiment_score = Column(Float, nullable=False)
        confidence_score = Column(Float, nullable=False)
        extracted_entities = Column(JSON, default=list)
        
        # Performance metrics
        processing_time_ms = Column(Integer, nullable=False)
        tokens_used = Column(Integer, default=0)
        status = Column(String(20), default="completed")
        
        # Timestamps
        created_at = Column(DateTime, default=func.now(), nullable=False)
        
        def __repr__(self):
            return f"<AnalysisModel(commit_id={self.commit_id}, model={self.model_name})>"

Git AI Analytics - Complete Implementation

Code ∙ Version 3 

\### src/application/services/event\_dispatcher.py \`\`\`python from abc import ABC, abstractmethod from typing import Dict, List, Callable, Any from ...domain.events.base import DomainEvent import asyncio import logging logger = logging.getLogger(\_\_name

    ### src/infrastructure/persistence/repositories/sqlite_commit_repository.py
    ```python
    from typing import List, Optional, Dict, Any
    from datetime import datetime
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import select, and_, or_, desc, func
    from sqlalchemy.orm import selectinload
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.entities.commit import Commit, CommitHash, FileChange
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
                    id=commit.id,
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
                logger.error(f"Error saving commit {commit.commit_hash.value}: {str(e)}")
                raise
        
        async def find_by_id(self, commit_id: str) -> Optional[Commit]:
            """Find commit by ID"""
            try:
                stmt = select(CommitModel).where(CommitModel.id == commit_id)
                result = await self.session.execute(stmt)
                db_commit = result.scalar_one_or_none()
                
                return self._to_domain(db_commit) if db_commit else None
            except Exception as e:
                logger.error(f"Error finding commit by ID {commit_id}: {str(e)}")
                return None
        
        async def find_by_hash(self, commit_hash: str) -> Optional[Commit]:
            """Find commit by hash"""
            try:
                stmt = select(CommitModel).where(CommitModel.commit_hash == commit_hash)
                result = await self.session.execute(stmt)
                db_commit = result.scalar_one_or_none()
                
                return self._to_domain(db_commit) if db_commit else None
            except Exception as e:
                logger.error(f"Error finding commit by hash {commit_hash}: {str(e)}")
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
                logger.error(f"Error finding commits by author {author_email}: {str(e)}")
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
                logger.error(f"Error finding commits by project {project}: {str(e)}")
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
                logger.error(f"Error searching commits with query '{query}': {str(e)}")
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
                logger.error(f"Error updating commit {commit.id}: {str(e)}")
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
                logger.error(f"Error getting statistics: {str(e)}")
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
    
    ### src/infrastructure/ai/ollama_service.py
    ```python
    import httpx
    import json
    import logging
    from typing import Dict, Any, List, Optional
    from ...domain.services.ai_analyzer import IAIAnalyzer
    from ...domain.entities.commit import Commit
    from ...domain.entities.analysis import AnalysisResult
    from .prompts.commit_prompts import CommitPrompts
    import time
    import os
    
    logger = logging.getLogger(__name__)
    
    class OllamaService(IAIAnalyzer):
        """Ollama AI service implementation"""
        
        def __init__(
            self,
            base_url: str = None,
            model: str = None,
            timeout: int = 30
        ):
            self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
            self.timeout = timeout
            self.prompts = CommitPrompts()
            
            # Remove trailing slash
            self.base_url = self.base_url.rstrip('/')
        
        async def analyze_commit(self, commit: Commit) -> AnalysisResult:
            """Analyze a commit using Ollama"""
            prompt = self.prompts.get_commit_analysis_prompt(commit)
            
            try:
                start_time = time.time()
                response = await self._generate_response(prompt, temperature=0.3)
                processing_time = int((time.time() - start_time) * 1000)
                
                # Parse structured response
                analysis_data = self._parse_analysis_response(response)
                
                return AnalysisResult(
                    summary=analysis_data.get('summary', 'No summary available'),
                    tags=analysis_data.get('tags', []),
                    sentiment_score=analysis_data.get('sentiment', 0.0),
                    confidence_score=analysis_data.get('confidence', 0.5),
                    extracted_entities=analysis_data.get('entities', [])
                )
                
            except Exception as e:
                logger.error(f"Error analyzing commit {commit.commit_hash.value}: {str(e)}")
                # Return default analysis on error
                return AnalysisResult(
                    summary=f"Analysis failed: {str(e)}",
                    tags=['error'],
                    sentiment_score=0.0,
                    confidence_score=0.0,
                    extracted_entities=[]
                )
        
        async def generate_summary(self, commits: List[Commit]) -> str:
            """Generate summary for multiple commits"""
            if not commits:
                return "No commits to summarize."
            
            # Git AI Analytics - Full Implementation
    
    ## Project Structure
    ```
    git-ai-analytics/
    ├── src/
    │   ├── domain/                    # Domain Layer
    │   │   ├── entities/
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py
    │   │   │   ├── commit.py
    │   │   │   ├── developer.py
    │   │   │   └── analysis.py
    │   │   ├── events/
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py
    │   │   │   └── commit_events.py
    │   │   ├── repositories/
    │   │   │   ├── __init__.py
    │   │   │   ├── commit_repository.py
    │   │   │   └── analysis_repository.py
    │   │   └── services/
    │   │       ├── __init__.py
    │   │       └── ai_analyzer.py
    │   ├── application/               # Application Layer
    │   │   ├── use_cases/
    │   │   │   ├── __init__.py
    │   │   │   ├── process_commit.py
    │   │   │   ├── analyze_commit.py
    │   │   │   └── search_commits.py
    │   │   ├── dto/
    │   │   │   ├── __init__.py
    │   │   │   ├── commit_dto.py
    │   │   │   └── webhook_dto.py
    │   │   ├── services/
    │   │   │   ├── __init__.py
    │   │   │   └── event_dispatcher.py
    │   │   └── interfaces/
    │   │       ├── __init__.py
    │   │       ├── cache_interface.py
    │   │       ├── ai_interface.py
    │   │       └── messaging_interface.py
    │   ├── infrastructure/           # Infrastructure Layer
    │   │   ├── persistence/
    │   │   │   ├── __init__.py
    │   │   │   ├── models/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── commit_model.py
    │   │   │   ├── repositories/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── sqlite_commit_repository.py
    │   │   │   └── database.py
    │   │   ├── ai/
    │   │   │   ├── __init__.py
    │   │   │   ├── ollama_service.py
    │   │   │   └── prompts/
    │   │   │       └── commit_prompts.py
    │   │   ├── cache/
    │   │   │   ├── __init__.py
    │   │   │   ├── memory_cache.py
    │   │   │   └── redis_cache.py
    │   │   ├── messaging/
    │   │   │   ├── __init__.py
    │   │   │   ├── celery_config.py
    │   │   │   └── tasks/
    │   │   │       ├── __init__.py
    │   │   │       └── analysis_tasks.py
    │   │   └── events/
    │   │       ├── __init__.py
    │   │       └── event_bus.py
    │   ├── presentation/             # Presentation Layer
    │   │   ├── api/
    │   │   │   ├── __init__.py
    │   │   │   ├── v1/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── webhooks.py
    │   │   │   │   ├── commits.py
    │   │   │   │   └── chat.py
    │   │   │   └── dependencies.py
    │   │   └── websocket/
    │   │       ├── __init__.py
    │   │       └── handlers.py
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── config.py
    │   │   └── logging.py
    │   └── main.py
    ├── configs/
    │   ├── default.yaml
    │   ├── development.yaml
    │   └── production.yaml
    ├── tests/
    ├── migrations/
    ├── docker/
    │   ├── Dockerfile
    │   └── docker-compose.yml
    ├── requirements/
    │   ├── base.txt
    │   ├── development.txt
    │   └── production.txt
    ├── .env.example
    └── README.md
    ```
    
    ## 1. Domain Layer
    
    ### src/domain/entities/base.py
    ```python
    from abc import ABC
    from datetime import datetime
    from typing import Any
    from uuid import uuid4, UUID
    from dataclasses import dataclass, field
    
    @dataclass
    class ValueObject(ABC):
        """Base class for value objects"""
        
        def __post_init__(self):
            self._validate()
        
        def _validate(self):
            """Override to add validation logic"""
            pass
    
    class Entity(ABC):
        """Base class for entities"""
        
        def __init__(self):
            self.id: str = str(uuid4())
            self.created_at: datetime = datetime.utcnow()
            self.updated_at: datetime = datetime.utcnow()
        
        def __eq__(self, other):
            if not isinstance(other, self.__class__):
                return False
            return self.id == other.id
        
        def __hash__(self):
            return hash(self.id)
    ```
    
    ### src/domain/entities/commit.py
    ```python
    from dataclasses import dataclass
    from datetime import datetime
    from typing import List, Optional
    from .base import Entity, ValueObject
    
    @dataclass
    class CommitHash(ValueObject):
        """Value object for commit hash"""
        value: str
        
        def _validate(self):
            if not self.value or len(self.value) < 6:
                raise ValueError("Invalid commit hash")
    
    @dataclass
    class FileChange(ValueObject):
        """Value object for file changes"""
        filename: str
        additions: int
        deletions: int
        status: str  # added, modified, deleted, renamed
        
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
        """Commit domain entity"""
        
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
            self.commit_hash = commit_hash
            self.author_email = author_email
            self.author_name = author_name
            self.message = message
            self.timestamp = timestamp
            self.branch = branch
            self.project = project
            self.files_changed = files_changed or []
            self.issue_numbers = issue_numbers or []
            
            # Analysis results (populated later)
            self.summary: Optional[str] = None
            self.tags: List[str] = []
            self.sentiment_score: Optional[float] = None
            self.embedding_id: Optional[str] = None
            self.analyzed_at: Optional[datetime] = None
            
            # Computed properties
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
    ```
    
    ### src/domain/entities/analysis.py
    ```python
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, List, Optional
    from .base import Entity, ValueObject
    
    @dataclass
    class AnalysisResult(ValueObject):
        """Value object for AI analysis results"""
        summary: str
        tags: List[str]
        sentiment_score: float
        confidence_score: float
        extracted_entities: List[str]
        
        def _validate(self):
            if not (-1 <= self.sentiment_score <= 1):
                raise ValueError("Sentiment score must be between -1 and 1")
            if not (0 <= self.confidence_score <= 1):
                raise ValueError("Confidence score must be between 0 and 1")
    
    class Analysis(Entity):
        """Analysis domain entity"""
        
        def __init__(
            self,
            commit_id: str,
            model_name: str,
            model_version: str,
            prompt_version: str,
            result: AnalysisResult,
            processing_time_ms: int,
            tokens_used: int = 0
        ):
            super().__init__()
            self.commit_id = commit_id
            self.model_name = model_name
            self.model_version = model_version
            self.prompt_version = prompt_version
            self.result = result
            self.processing_time_ms = processing_time_ms
            self.tokens_used = tokens_used
            self.status = "completed"
        
        def is_high_confidence(self) -> bool:
            """Check if analysis has high confidence"""
            return self.result.confidence_score >= 0.8
    ```
    
    ### src/domain/events/base.py
    ```python
    from abc import ABC
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, Any
    import json
    from uuid import uuid4
    
    @dataclass
    class DomainEvent(ABC):
        """Base class for domain events"""
        event_id: str = None
        occurred_at: datetime = None
        
        def __post_init__(self):
            if self.event_id is None:
                self.event_id = str(uuid4())
            if self.occurred_at is None:
                self.occurred_at = datetime.utcnow()
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert event to dictionary"""
            return {
                'event_type': self.__class__.__name__,
                'event_id': self.event_id,
                'occurred_at': self.occurred_at.isoformat(),
                'data': self._get_event_data()
            }
        
        def to_json(self) -> str:
            """Convert event to JSON string"""
            return json.dumps(self.to_dict(), default=str)
        
        def _get_event_data(self) -> Dict[str, Any]:
            """Override to provide event-specific data"""
            return {}
    ```
    
    ### src/domain/events/commit_events.py
    ```python
    from dataclasses import dataclass
    from typing import Dict, Any, List
    from .base import DomainEvent
    
    @dataclass
    class CommitReceivedEvent(DomainEvent):
        """Event fired when a new commit is received"""
        commit_id: str
        commit_hash: str
        project: str
        author: str
        branch: str
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'commit_hash': self.commit_hash,
                'project': self.project,
                'author': self.author,
                'branch': self.branch
            }
    
    @dataclass 
    class CommitAnalysisStartedEvent(DomainEvent):
        """Event fired when commit analysis starts"""
        commit_id: str
        model_name: str
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'model_name': self.model_name
            }
    
    @dataclass
    class CommitAnalysisCompletedEvent(DomainEvent):
        """Event fired when commit analysis completes"""
        commit_id: str
        analysis_id: str
        summary: str
        tags: List[str]
        processing_time_ms: int
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'analysis_id': self.analysis_id,
                'summary': self.summary,
                'tags': self.tags,
                'processing_time_ms': self.processing_time_ms
            }
    
    @dataclass
    class CommitAnalysisFailedEvent(DomainEvent):
        """Event fired when commit analysis fails"""
        commit_id: str
        error_message: str
        retry_count: int
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'error_message': self.error_message,
                'retry_count': self.retry_count
            }
    ```
    
    ### src/domain/repositories/commit_repository.py
    ```python
    from abc import ABC, abstractmethod
    from typing import List, Optional, Dict, Any
    from datetime import datetime
    from ..entities.commit import Commit
    
    class ICommitRepository(ABC):
        """Repository interface for Commit entity"""
        
        @abstractmethod
        async def save(self, commit: Commit) -> Commit:
            """Save a commit"""
            pass
        
        @abstractmethod
        async def find_by_id(self, commit_id: str) -> Optional[Commit]:
            """Find commit by ID"""
            pass
        
        @abstractmethod
        async def find_by_hash(self, commit_hash: str) -> Optional[Commit]:
            """Find commit by hash"""
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
    ```
    
    ### src/domain/services/ai_analyzer.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, List
    from ..entities.commit import Commit
    from ..entities.analysis import AnalysisResult
    
    class IAIAnalyzer(ABC):
        """Domain service interface for AI analysis"""
        
        @abstractmethod
        async def analyze_commit(self, commit: Commit) -> AnalysisResult:
            """Analyze a commit using AI"""
            pass
        
        @abstractmethod
        async def generate_summary(self, commits: List[Commit]) -> str:
            """Generate summary for multiple commits"""
            pass
        
        @abstractmethod
        async def answer_question(
            self, 
            question: str, 
            context_commits: List[Commit]
        ) -> str:
            """Answer question about commits"""
            pass
        
        @abstractmethod
        async def extract_entities(self, text: str) -> List[str]:
            """Extract named entities from text"""
            pass
        
        @abstractmethod
        def get_model_info(self) -> Dict[str, str]:
            """Get current model information"""
            pass
    ```
    
    ## 2. Application Layer
    
    ### src/application/dto/commit_dto.py
    ```python
    from pydantic import BaseModel, Field, validator
    from datetime import datetime
    from typing import List, Optional
    
    class FileChangeDto(BaseModel):
        """DTO for file changes"""
        filename: str = Field(..., min_length=1)
        additions: int = Field(..., ge=0)
        deletions: int = Field(..., ge=0)
        status: str = Field(..., regex="^(added|modified|deleted|renamed)$")
    
    class CommitDto(BaseModel):
        """DTO for commit data"""
        commit_hash: str = Field(..., min_length=6)
        author_email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
        author_name: str = Field(..., min_length=1)
        message: str = Field(..., min_length=1)
        timestamp: datetime
        branch: str = Field(..., min_length=1)
        project: str = Field(..., min_length=1)
        files_changed: List[FileChangeDto] = []
        issue_numbers: List[str] = []
    
    class CommitResponseDto(BaseModel):
        """DTO for commit response"""
        id: str
        commit_hash: str
        author_email: str
        author_name: str
        message: str
        timestamp: datetime
        branch: str
        project: str
        files_count: int
        total_lines_changed: int
        complexity_score: Optional[float] = None
        summary: Optional[str] = None
        tags: List[str] = []
        sentiment_score: Optional[float] = None
        analyzed_at: Optional[datetime] = None
        created_at: datetime
    
    class CommitSearchDto(BaseModel):
        """DTO for commit search"""
        query: Optional[str] = None
        project: Optional[str] = None
        author: Optional[str] = None
        branch: Optional[str] = None
        start_date: Optional[datetime] = None
        end_date: Optional[datetime] = None
        tags: List[str] = []
        limit: int = Field(50, ge=1, le=200)
        offset: int = Field(0, ge=0)
    
    class ChatRequestDto(BaseModel):
        """DTO for chat requests"""
        question: str = Field(..., min_length=1, max_length=1000)
        context_project: Optional[str] = None
        context_author: Optional[str] = None
        context_days: int = Field(30, ge=1, le=365)
    
    class ChatResponseDto(BaseModel):
        """DTO for chat responses"""
        answer: str
        context_commits_count: int
        processing_time_ms: int
        model_used: str
    ```
    
    ### src/application/dto/webhook_dto.py
    ```python
    from pydantic import BaseModel
    from typing import List, Optional
    from datetime import datetime
    
    class WebhookAuthorDto(BaseModel):
        """DTO for webhook author data"""
        name: str
        email: str
        username: Optional[str] = None
    
    class WebhookCommitDto(BaseModel):
        """DTO for webhook commit data"""
        id: str
        message: str
        timestamp: datetime
        author: WebhookAuthorDto
        added: List[str] = []
        modified: List[str] = []
        removed: List[str] = []
        url: Optional[str] = None
    
    class WebhookRepositoryDto(BaseModel):
        """DTO for webhook repository data"""
        name: str
        full_name: str
        url: str
        default_branch: str
    
    class WebhookPayloadDto(BaseModel):
        """DTO for webhook payload"""
        ref: str  # refs/heads/main
        repository: WebhookRepositoryDto
        commits: List[WebhookCommitDto]
        pusher: WebhookAuthorDto
        head_commit: Optional[WebhookCommitDto] = None
    ```
    
    ### src/application/use_cases/process_commit.py
    ```python
    from typing import Optional
    from datetime import datetime
    from ..dto.commit_dto import CommitDto, CommitResponseDto
    from ..services.event_dispatcher import IEventDispatcher
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.entities.commit import Commit, CommitHash, FileChange
    from ...domain.events.commit_events import CommitReceivedEvent
    from ...application.interfaces.cache_interface import ICacheService
    from ...application.interfaces.messaging_interface import IMessageQueue
    
    class ProcessCommitUseCase:
        """Use case for processing incoming commits"""
        
        def __init__(
            self,
            commit_repository: ICommitRepository,
            event_dispatcher: IEventDispatcher,
            message_queue: IMessageQueue,
            cache_service: ICacheService
        ):
            self.commit_repository = commit_repository
            self.event_dispatcher = event_dispatcher
            self.message_queue = message_queue
            self.cache_service = cache_service
        
        async def execute(self, commit_dto: CommitDto) -> CommitResponseDto:
            """Process a new commit"""
            
            # Check if commit already exists (idempotency)
            existing = await self.commit_repository.find_by_hash(commit_dto.commit_hash)
            if existing:
                return self._to_response_dto(existing)
            
            # Create domain entity
            commit = self._to_domain(commit_dto)
            
            # Save to repository
            saved_commit = await self.commit_repository.save(commit)
            
            # Dispatch domain event
            event = CommitReceivedEvent(
                commit_id=saved_commit.id,
                commit_hash=saved_commit.commit_hash.value,
                project=saved_commit.project,
                author=saved_commit.author_email,
                branch=saved_commit.branch
            )
            await self.event_dispatcher.dispatch(event)
            
            # Queue for async analysis
            await self.message_queue.enqueue(
                "analyze_commit",
                {"commit_id": saved_commit.id}
            )
            
            # Invalidate relevant caches
            await self.cache_service.invalidate_pattern(f"commits:project:{commit.project}:*")
            await self.cache_service.invalidate_pattern(f"commits:author:{commit.author_email}:*")
            
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
    ```
    
    ### src/application/use_cases/search_commits.py
    ```python
    from typing import List
    from ..dto.commit_dto import CommitSearchDto, CommitResponseDto
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...application.interfaces.cache_interface import ICacheService
    import hashlib
    import json
    
    class SearchCommitsUseCase:
        """Use case for searching commits"""
        
        def __init__(
            self,
            commit_repository: ICommitRepository,
            cache_service: ICacheService
        ):
            self.commit_repository = commit_repository
            self.cache_service = cache_service
        
        async def execute(self, search_dto: CommitSearchDto) -> List[CommitResponseDto]:
            """Search commits with caching"""
            
            # Generate cache key
            cache_key = self._generate_cache_key(search_dto)
            
            # Try cache first
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                return cached_result
            
            # Search in repository
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
                limit=search_dto.limit
            )
            
            # Convert to DTOs
            result = [self._to_response_dto(commit) for commit in commits]
            
            # Cache result for 5 minutes
            await self.cache_service.set(cache_key, result, ttl=300)
            
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
    ```
    
    ### src/application/use_cases/chat_with_ai.py
    ```python
    from typing import List
    from datetime import datetime, timedelta
    import time
    from ..dto.commit_dto import ChatRequestDto, ChatResponseDto
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.services.ai_analyzer import IAIAnalyzer
    from ...application.interfaces.cache_interface import ICacheService
    import hashlib
    
    class ChatWithAIUseCase:
        """Use case for chatting with AI about commits"""
        
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
            """Answer question about commits using AI"""
            
            start_time = time.time()
            
            # Generate cache key
            cache_key = self._generate_cache_key(chat_dto)
            
            # Try cache first
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                return cached_result
            
            # Get context commits
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=chat_dto.context_days)
            
            context_commits = await self.commit_repository.find_by_project(
                project=chat_dto.context_project,
                start_date=start_date,
                end_date=end_date,
                limit=100
            )
            
            if chat_dto.context_author:
                author_commits = await self.commit_repository.find_by_author(
                    author_email=chat_dto.context_author,
                    project=chat_dto.context_project,
                    start_date=start_date,
                    end_date=end_date,
                    limit=50
                )
                context_commits.extend(author_commits)
            
            # Remove duplicates
            unique_commits = {commit.id: commit for commit in context_commits}
            context_commits = list(unique_commits.values())
            
            # Ask AI
            answer = await self.ai_analyzer.answer_question(
                question=chat_dto.question,
                context_commits=context_commits
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            result = ChatResponseDto(
                answer=answer,
                context_commits_count=len(context_commits),
                processing_time_ms=processing_time_ms,
                model_used=self.ai_analyzer.get_model_info()['name']
            )
            
            # Cache result for 10 minutes
            await self.cache_service.set(cache_key, result, ttl=600)
            
            return result
        
        def _generate_cache_key(self, chat_dto: ChatRequestDto) -> str:
            """Generate cache key for chat request"""
            chat_hash = hashlib.md5(
                f"{chat_dto.question}:{chat_dto.context_project}:{chat_dto.context_author}:{chat_dto.context_days}".encode()
            ).hexdigest()
    ### src/application/services/event_dispatcher.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, List, Callable, Any
    from ...domain.events.base import DomainEvent
    import asyncio
    import logging
    
    logger = logging.getLogger(__name__)
    
    class IEventDispatcher(ABC):
        """Event dispatcher interface"""
        
        @abstractmethod
        async def dispatch(self, event: DomainEvent):
            """Dispatch event to all registered handlers"""
            pass
        
        @abstractmethod
        def register_handler(self, event_type: str, handler: Callable):
            """Register event handler"""
            pass
    
    class EventDispatcher(IEventDispatcher):
        """Implementation of event dispatcher"""
        
        def __init__(self):
            self.handlers: Dict[str, List[Callable]] = {}
        
        def register_handler(self, event_type: str, handler: Callable):
            """Register event handler"""
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
            logger.info(f"Registered handler for event type: {event_type}")
        
        async def dispatch(self, event: DomainEvent):
            """Dispatch event to all registered handlers"""
            event_type = event.__class__.__name__
            
            if event_type not in self.handlers:
                logger.debug(f"No handlers registered for event type: {event_type}")
                return
            
            logger.info(f"Dispatching event: {event_type} (ID: {event.event_id})")
            
            tasks = []
            for handler in self.handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        # Run sync handler in thread pool
                        loop = asyncio.get_event_loop()
                        tasks.append(loop.run_in_executor(None, handler, event))
                except Exception as e:
                    logger.error(f"Error preparing handler for {event_type}: {str(e)}")
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log any handler exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Handler {i} failed for {event_type}: {str(result)}")
    
    ### src/application/interfaces/cache_interface.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Any, Optional, List
    
    class ICacheService(ABC):
        """Interface for cache service"""
        
        @abstractmethod
        async def get(self, key: str) -> Optional[Any]:
            """Get value from cache"""
            pass
        
        @abstractmethod
        async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
            """Set value in cache with TTL in seconds"""
            pass
        
        @abstractmethod
        async def delete(self, key: str) -> bool:
            """Delete key from cache"""
            pass
        
        @abstractmethod
        async def exists(self, key: str) -> bool:
            """Check if key exists in cache"""
            pass
        
        @abstractmethod
        async def invalidate_pattern(self, pattern: str) -> int:
            """Invalidate all keys matching pattern (with * wildcard)"""
            pass
        
        @abstractmethod
        async def get_many(self, keys: List[str]) -> Dict[str, Any]:
            """Get multiple values from cache"""
            pass
        
        @abstractmethod
        async def set_many(self, data: Dict[str, Any], ttl: int = 3600) -> bool:
            """Set multiple values in cache"""
            pass
    
    ### src/application/interfaces/ai_interface.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, List, Optional
    from ...domain.entities.analysis import AnalysisResult
    
    class IAIService(ABC):
        """Interface for AI service"""
        
        @abstractmethod
        async def analyze_text(
            self,
            text: str,
            context: Optional[str] = None,
            temperature: float = 0.7
        ) -> AnalysisResult:
            """Analyze text and return structured results"""
            pass
        
        @abstractmethod
        async def generate_response(
            self,
            prompt: str,
            context: Optional[str] = None,
            max_tokens: int = 500
        ) -> str:
            """Generate text response"""
            pass
        
        @abstractmethod
        async def create_embedding(self, text: str) -> List[float]:
            """Create text embedding"""
            pass
        
        @abstractmethod
        async def health_check(self) -> bool:
            """Check if AI service is healthy"""
            pass
        
        @abstractmethod
        def get_model_info(self) -> Dict[str, str]:
            """Get model information"""
            pass
    
    ### src/application/interfaces/messaging_interface.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, Optional, Callable
    from enum import Enum
    
    class TaskStatus(Enum):
        PENDING = "PENDING"
        STARTED = "STARTED"
        SUCCESS = "SUCCESS"
        FAILURE = "FAILURE"
        RETRY = "RETRY"
    
    class IMessageQueue(ABC):
        """Interface for message queue"""
        
        @abstractmethod
        async def enqueue(
            self,
            task_name: str,
            payload: Dict[str, Any],
            delay_seconds: int = 0,
            priority: int = 5
        ) -> str:
            """Enqueue task for processing"""
            pass
        
        @abstractmethod
        async def get_task_status(self, task_id: str) -> TaskStatus:
            """Get task status"""
            pass
        
        @abstractmethod
        async def get_task_result(self, task_id: str) -> Optional[Any]:
            """Get task result"""
            pass
        
        @abstractmethod
        async def cancel_task(self, task_id: str) -> bool:
            """Cancel pending task"""
            pass
        
        @abstractmethod
        def register_task(self, task_name: str, handler: Callable):
            """Register task handler"""
            pass
    
    ## 3. Infrastructure Layer
    
    ### src/infrastructure/persistence/database.py
    ```python
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy import MetaData
    from contextlib import asynccontextmanager
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    # SQLAlchemy base
    Base = declarative_base()
    metadata = MetaData()
    
    # Global engine and session maker
    engine = None
    SessionLocal = None
    
    def get_database_url() -> str:
        """Get database URL from environment"""
        db_provider = os.getenv("DB_PROVIDER", "sqlite")
        
        if db_provider == "sqlite":
            db_path = os.getenv("DB_PATH", "data/git_analytics.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            return f"sqlite+aiosqlite:///{db_path}"
        
        elif db_provider == "postgresql":
            host = os.getenv("DB_HOST", "localhost")
            port = os.getenv("DB_PORT", "5432")
            name = os.getenv("DB_NAME", "git_analytics")
            user = os.getenv("DB_USER", "postgres")
            password = os.getenv("DB_PASSWORD", "")
            return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"
        
        else:
            raise ValueError(f"Unsupported database provider: {db_provider}")
    
    async def init_database():
        """Initialize database connection"""
        global engine, SessionLocal
        
        if engine is None:
            database_url = get_database_url()
            logger.info(f"Connecting to database: {database_url.split('://')[0]}://...")
            
            engine = create_async_engine(
                database_url,
                echo=os.getenv("DB_ECHO", "false").lower() == "true",
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            SessionLocal = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database initialized successfully")
    
    async def close_database():
        """Close database connection"""
        global engine
        if engine:
            await engine.dispose()
            logger.info("Database connection closed")
    
    @asynccontextmanager
    async def get_session() -> AsyncSession:
        """Get database session"""
        if SessionLocal is None:
            await init_database()
        
        async with SessionLocal() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    ### src/infrastructure/persistence/models/commit_model.py
    ```python
    from sqlalchemy import Column, String, DateTime, Text, Integer, Float, JSON
    from sqlalchemy.sql import func
    from ..database import Base
    
    class CommitModel(Base):
        """SQLAlchemy model for commits"""
        
        __tablename__ = "commits"
        
        # Primary fields
        id = Column(String(36), primary_key=True)
        commit_hash = Column(String(40), unique=True, nullable=False, index=True)
        author_email = Column(String(255), nullable=False, index=True)
        author_name = Column(String(255), nullable=False)
        message = Column(Text, nullable=False)
        timestamp = Column(DateTime, nullable=False, index=True)
        branch = Column(String(255), nullable=False, index=True)
        project = Column(String(255), nullable=False, index=True)
        
        # File changes (stored as JSON)
        files_changed = Column(JSON, nullable=False, default=list)
        issue_numbers = Column(JSON, nullable=False, default=list)
        
        # Metrics
        total_lines_changed = Column(Integer, default=0)
        files_count = Column(Integer, default=0)
        complexity_score = Column(Float)
        impact_score = Column(Float)
        
        # Analysis results
        summary = Column(Text)
        tags = Column(JSON, default=list)
        sentiment_score = Column(Float)
        embedding_id = Column(String(255))
        analyzed_at = Column(DateTime)
        
        # Timestamps
        created_at = Column(DateTime, default=func.now(), nullable=False)
        updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
        
        def __repr__(self):
            return f"<CommitModel(hash={self.commit_hash}, project={self.project})>"
    
    class AnalysisModel(Base):
        """SQLAlchemy model for analysis results"""
        
        __tablename__ = "analyses"
        
        id = Column(String(36), primary_key=True)
        commit_id = Column(String(36), nullable=False, index=True)
        
        # Model information
        model_name = Column(String(100), nullable=False)
        model_version = Column(String(50), nullable=False)
        prompt_version = Column(String(50), nullable=False)
        
        # Analysis results
        summary = Column(Text, nullable=False)
        tags = Column(JSON, default=list)
        sentiment_score = Column(Float, nullable=False)
        confidence_score = Column(Float, nullable=False)
        extracted_entities = Column(JSON, default=list)
        
        # Performance metrics
        processing_time_ms = Column(Integer, nullable=False)
        tokens_used = Column(Integer, default=0)
        status = Column(String(20), default="completed")
        
        # Timestamps
        created_at = Column(DateTime, default=func.now(), nullable=False)
        
        def __repr__(self):
            return f"<AnalysisModel(commit_id={self.commit_id}, model={self.model_name})>"

Git AI Analytics - Complete Implementation

Code ∙ Version 4 

\### src/infrastructure/persistence/repositories/sqlite\_commit\_repository.py \`\`\`python from typing import List, Optional, Dict, Any from datetime import datetime from sqlalchemy.ext.asyncio import AsyncSession from sqlalchemy import select, and\_, or\_,

Retry

---

You asked:
----------

LL

Continue

Edit

---

Claude Replied:
---------------

    ### src/infrastructure/persistence/repositories/sqlite_commit_repository.py
    ```python
    from typing import List, Optional, Dict, Any
    from datetime import datetime
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import select, and_, or_, desc, func
    from sqlalchemy.orm import selectinload
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.entities.commit import Commit, CommitHash, FileChange
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
                    id=commit.id,
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
                logger.error(f"Error saving commit {commit.commit_hash.value}: {str(e)}")
                raise
        
        async def find_by_id(self, commit_id: str) -> Optional[Commit]:
            """Find commit by ID"""
            try:
                stmt = select(CommitModel).where(CommitModel.id == commit_id)
                result = await self.session.execute(stmt)
                db_commit = result.scalar_one_or_none()
                
                return self._to_domain(db_commit) if db_commit else None
            except Exception as e:
                logger.error(f"Error finding commit by ID {commit_id}: {str(e)}")
                return None
        
        async def find_by_hash(self, commit_hash: str) -> Optional[Commit]:
            """Find commit by hash"""
            try:
                stmt = select(CommitModel).where(CommitModel.commit_hash == commit_hash)
                result = await self.session.execute(stmt)
                db_commit = result.scalar_one_or_none()
                
                return self._to_domain(db_commit) if db_commit else None
            except Exception as e:
                logger.error(f"Error finding commit by hash {commit_hash}: {str(e)}")
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
                logger.error(f"Error finding commits by author {author_email}: {str(e)}")
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
                logger.error(f"Error finding commits by project {project}: {str(e)}")
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
                logger.error(f"Error searching commits with query '{query}': {str(e)}")
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
                logger.error(f"Error updating commit {commit.id}: {str(e)}")
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
                logger.error(f"Error getting statistics: {str(e)}")
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
    
    ### src/infrastructure/ai/ollama_service.py
    ```python
    import httpx
    import json
    import logging
    from typing import Dict, Any, List, Optional
    from ...domain.services.ai_analyzer import IAIAnalyzer
    from ...domain.entities.commit import Commit
    from ...domain.entities.analysis import AnalysisResult
    from .prompts.commit_prompts import CommitPrompts
    import time
    import os
    
    logger = logging.getLogger(__name__)
    
    class OllamaService(IAIAnalyzer):
        """Ollama AI service implementation"""
        
        def __init__(
            self,
            base_url: str = None,
            model: str = None,
            timeout: int = 30
        ):
            self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
            self.timeout = timeout
            self.prompts = CommitPrompts()
            
            # Remove trailing slash
            self.base_url = self.base_url.rstrip('/')
        
        async def analyze_commit(self, commit: Commit) -> AnalysisResult:
            """Analyze a commit using Ollama"""
            prompt = self.prompts.get_commit_analysis_prompt(commit)
            
            try:
                start_time = time.time()
                response = await self._generate_response(prompt, temperature=0.3)
                processing_time = int((time.time() - start_time) * 1000)
                
                # Parse structured response
                analysis_data = self._parse_analysis_response(response)
                
                return AnalysisResult(
                    summary=analysis_data.get('summary', 'No summary available'),
                    tags=analysis_data.get('tags', []),
                    sentiment_score=analysis_data.get('sentiment', 0.0),
                    confidence_score=analysis_data.get('confidence', 0.5),
                    extracted_entities=analysis_data.get('entities', [])
                )
                
            except Exception as e:
                logger.error(f"Error analyzing commit {commit.commit_hash.value}: {str(e)}")
                # Return default analysis on error
                return AnalysisResult(
                    summary=f"Analysis failed: {str(e)}",
                    tags=['error'],
                    sentiment_score=0.0,
                    confidence_score=0.0,
                    extracted_entities=[]
                )
        
        async def generate_summary(self, commits: List[Commit]) -> str:
            """Generate summary for multiple commits"""
            if not commits:
                return "No commits to summarize."
            
            # Git AI Analytics - Full Implementation
    
    ## Project Structure
    ```
    git-ai-analytics/
    ├── src/
    │   ├── domain/                    # Domain Layer
    │   │   ├── entities/
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py
    │   │   │   ├── commit.py
    │   │   │   ├── developer.py
    │   │   │   └── analysis.py
    │   │   ├── events/
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py
    │   │   │   └── commit_events.py
    │   │   ├── repositories/
    │   │   │   ├── __init__.py
    │   │   │   ├── commit_repository.py
    │   │   │   └── analysis_repository.py
    │   │   └── services/
    │   │       ├── __init__.py
    │   │       └── ai_analyzer.py
    │   ├── application/               # Application Layer
    │   │   ├── use_cases/
    │   │   │   ├── __init__.py
    │   │   │   ├── process_commit.py
    │   │   │   ├── analyze_commit.py
    │   │   │   └── search_commits.py
    │   │   ├── dto/
    │   │   │   ├── __init__.py
    │   │   │   ├── commit_dto.py
    │   │   │   └── webhook_dto.py
    │   │   ├── services/
    │   │   │   ├── __init__.py
    │   │   │   └── event_dispatcher.py
    │   │   └── interfaces/
    │   │       ├── __init__.py
    │   │       ├── cache_interface.py
    │   │       ├── ai_interface.py
    │   │       └── messaging_interface.py
    │   ├── infrastructure/           # Infrastructure Layer
    │   │   ├── persistence/
    │   │   │   ├── __init__.py
    │   │   │   ├── models/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── commit_model.py
    │   │   │   ├── repositories/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── sqlite_commit_repository.py
    │   │   │   └── database.py
    │   │   ├── ai/
    │   │   │   ├── __init__.py
    │   │   │   ├── ollama_service.py
    │   │   │   └── prompts/
    │   │   │       └── commit_prompts.py
    │   │   ├── cache/
    │   │   │   ├── __init__.py
    │   │   │   ├── memory_cache.py
    │   │   │   └── redis_cache.py
    │   │   ├── messaging/
    │   │   │   ├── __init__.py
    │   │   │   ├── celery_config.py
    │   │   │   └── tasks/
    │   │   │       ├── __init__.py
    │   │   │       └── analysis_tasks.py
    │   │   └── events/
    │   │       ├── __init__.py
    │   │       └── event_bus.py
    │   ├── presentation/             # Presentation Layer
    │   │   ├── api/
    │   │   │   ├── __init__.py
    │   │   │   ├── v1/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── webhooks.py
    │   │   │   │   ├── commits.py
    │   │   │   │   └── chat.py
    │   │   │   └── dependencies.py
    │   │   └── websocket/
    │   │       ├── __init__.py
    │   │       └── handlers.py
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── config.py
    │   │   └── logging.py
    │   └── main.py
    ├── configs/
    │   ├── default.yaml
    │   ├── development.yaml
    │   └── production.yaml
    ├── tests/
    ├── migrations/
    ├── docker/
    │   ├── Dockerfile
    │   └── docker-compose.yml
    ├── requirements/
    │   ├── base.txt
    │   ├── development.txt
    │   └── production.txt
    ├── .env.example
    └── README.md
    ```
    
    ## 1. Domain Layer
    
    ### src/domain/entities/base.py
    ```python
    from abc import ABC
    from datetime import datetime
    from typing import Any
    from uuid import uuid4, UUID
    from dataclasses import dataclass, field
    
    @dataclass
    class ValueObject(ABC):
        """Base class for value objects"""
        
        def __post_init__(self):
            self._validate()
        
        def _validate(self):
            """Override to add validation logic"""
            pass
    
    class Entity(ABC):
        """Base class for entities"""
        
        def __init__(self):
            self.id: str = str(uuid4())
            self.created_at: datetime = datetime.utcnow()
            self.updated_at: datetime = datetime.utcnow()
        
        def __eq__(self, other):
            if not isinstance(other, self.__class__):
                return False
            return self.id == other.id
        
        def __hash__(self):
            return hash(self.id)
    ```
    
    ### src/domain/entities/commit.py
    ```python
    from dataclasses import dataclass
    from datetime import datetime
    from typing import List, Optional
    from .base import Entity, ValueObject
    
    @dataclass
    class CommitHash(ValueObject):
        """Value object for commit hash"""
        value: str
        
        def _validate(self):
            if not self.value or len(self.value) < 6:
                raise ValueError("Invalid commit hash")
    
    @dataclass
    class FileChange(ValueObject):
        """Value object for file changes"""
        filename: str
        additions: int
        deletions: int
        status: str  # added, modified, deleted, renamed
        
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
        """Commit domain entity"""
        
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
            self.commit_hash = commit_hash
            self.author_email = author_email
            self.author_name = author_name
            self.message = message
            self.timestamp = timestamp
            self.branch = branch
            self.project = project
            self.files_changed = files_changed or []
            self.issue_numbers = issue_numbers or []
            
            # Analysis results (populated later)
            self.summary: Optional[str] = None
            self.tags: List[str] = []
            self.sentiment_score: Optional[float] = None
            self.embedding_id: Optional[str] = None
            self.analyzed_at: Optional[datetime] = None
            
            # Computed properties
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
    ```
    
    ### src/domain/entities/analysis.py
    ```python
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, List, Optional
    from .base import Entity, ValueObject
    
    @dataclass
    class AnalysisResult(ValueObject):
        """Value object for AI analysis results"""
        summary: str
        tags: List[str]
        sentiment_score: float
        confidence_score: float
        extracted_entities: List[str]
        
        def _validate(self):
            if not (-1 <= self.sentiment_score <= 1):
                raise ValueError("Sentiment score must be between -1 and 1")
            if not (0 <= self.confidence_score <= 1):
                raise ValueError("Confidence score must be between 0 and 1")
    
    class Analysis(Entity):
        """Analysis domain entity"""
        
        def __init__(
            self,
            commit_id: str,
            model_name: str,
            model_version: str,
            prompt_version: str,
            result: AnalysisResult,
            processing_time_ms: int,
            tokens_used: int = 0
        ):
            super().__init__()
            self.commit_id = commit_id
            self.model_name = model_name
            self.model_version = model_version
            self.prompt_version = prompt_version
            self.result = result
            self.processing_time_ms = processing_time_ms
            self.tokens_used = tokens_used
            self.status = "completed"
        
        def is_high_confidence(self) -> bool:
            """Check if analysis has high confidence"""
            return self.result.confidence_score >= 0.8
    ```
    
    ### src/domain/events/base.py
    ```python
    from abc import ABC
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, Any
    import json
    from uuid import uuid4
    
    @dataclass
    class DomainEvent(ABC):
        """Base class for domain events"""
        event_id: str = None
        occurred_at: datetime = None
        
        def __post_init__(self):
            if self.event_id is None:
                self.event_id = str(uuid4())
            if self.occurred_at is None:
                self.occurred_at = datetime.utcnow()
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert event to dictionary"""
            return {
                'event_type': self.__class__.__name__,
                'event_id': self.event_id,
                'occurred_at': self.occurred_at.isoformat(),
                'data': self._get_event_data()
            }
        
        def to_json(self) -> str:
            """Convert event to JSON string"""
            return json.dumps(self.to_dict(), default=str)
        
        def _get_event_data(self) -> Dict[str, Any]:
            """Override to provide event-specific data"""
            return {}
    ```
    
    ### src/domain/events/commit_events.py
    ```python
    from dataclasses import dataclass
    from typing import Dict, Any, List
    from .base import DomainEvent
    
    @dataclass
    class CommitReceivedEvent(DomainEvent):
        """Event fired when a new commit is received"""
        commit_id: str
        commit_hash: str
        project: str
        author: str
        branch: str
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'commit_hash': self.commit_hash,
                'project': self.project,
                'author': self.author,
                'branch': self.branch
            }
    
    @dataclass 
    class CommitAnalysisStartedEvent(DomainEvent):
        """Event fired when commit analysis starts"""
        commit_id: str
        model_name: str
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'model_name': self.model_name
            }
    
    @dataclass
    class CommitAnalysisCompletedEvent(DomainEvent):
        """Event fired when commit analysis completes"""
        commit_id: str
        analysis_id: str
        summary: str
        tags: List[str]
        processing_time_ms: int
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'analysis_id': self.analysis_id,
                'summary': self.summary,
                'tags': self.tags,
                'processing_time_ms': self.processing_time_ms
            }
    
    @dataclass
    class CommitAnalysisFailedEvent(DomainEvent):
        """Event fired when commit analysis fails"""
        commit_id: str
        error_message: str
        retry_count: int
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'error_message': self.error_message,
                'retry_count': self.retry_count
            }
    ```
    
    ### src/domain/repositories/commit_repository.py
    ```python
    from abc import ABC, abstractmethod
    from typing import List, Optional, Dict, Any
    from datetime import datetime
    from ..entities.commit import Commit
    
    class ICommitRepository(ABC):
        """Repository interface for Commit entity"""
        
        @abstractmethod
        async def save(self, commit: Commit) -> Commit:
            """Save a commit"""
            pass
        
        @abstractmethod
        async def find_by_id(self, commit_id: str) -> Optional[Commit]:
            """Find commit by ID"""
            pass
        
        @abstractmethod
        async def find_by_hash(self, commit_hash: str) -> Optional[Commit]:
            """Find commit by hash"""
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
    ```
    
    ### src/domain/services/ai_analyzer.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, List
    from ..entities.commit import Commit
    from ..entities.analysis import AnalysisResult
    
    class IAIAnalyzer(ABC):
        """Domain service interface for AI analysis"""
        
        @abstractmethod
        async def analyze_commit(self, commit: Commit) -> AnalysisResult:
            """Analyze a commit using AI"""
            pass
        
        @abstractmethod
        async def generate_summary(self, commits: List[Commit]) -> str:
            """Generate summary for multiple commits"""
            pass
        
        @abstractmethod
        async def answer_question(
            self, 
            question: str, 
            context_commits: List[Commit]
        ) -> str:
            """Answer question about commits"""
            pass
        
        @abstractmethod
        async def extract_entities(self, text: str) -> List[str]:
            """Extract named entities from text"""
            pass
        
        @abstractmethod
        def get_model_info(self) -> Dict[str, str]:
            """Get current model information"""
            pass
    ```
    
    ## 2. Application Layer
    
    ### src/application/dto/commit_dto.py
    ```python
    from pydantic import BaseModel, Field, validator
    from datetime import datetime
    from typing import List, Optional
    
    class FileChangeDto(BaseModel):
        """DTO for file changes"""
        filename: str = Field(..., min_length=1)
        additions: int = Field(..., ge=0)
        deletions: int = Field(..., ge=0)
        status: str = Field(..., regex="^(added|modified|deleted|renamed)$")
    
    class CommitDto(BaseModel):
        """DTO for commit data"""
        commit_hash: str = Field(..., min_length=6)
        author_email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
        author_name: str = Field(..., min_length=1)
        message: str = Field(..., min_length=1)
        timestamp: datetime
        branch: str = Field(..., min_length=1)
        project: str = Field(..., min_length=1)
        files_changed: List[FileChangeDto] = []
        issue_numbers: List[str] = []
    
    class CommitResponseDto(BaseModel):
        """DTO for commit response"""
        id: str
        commit_hash: str
        author_email: str
        author_name: str
        message: str
        timestamp: datetime
        branch: str
        project: str
        files_count: int
        total_lines_changed: int
        complexity_score: Optional[float] = None
        summary: Optional[str] = None
        tags: List[str] = []
        sentiment_score: Optional[float] = None
        analyzed_at: Optional[datetime] = None
        created_at: datetime
    
    class CommitSearchDto(BaseModel):
        """DTO for commit search"""
        query: Optional[str] = None
        project: Optional[str] = None
        author: Optional[str] = None
        branch: Optional[str] = None
        start_date: Optional[datetime] = None
        end_date: Optional[datetime] = None
        tags: List[str] = []
        limit: int = Field(50, ge=1, le=200)
        offset: int = Field(0, ge=0)
    
    class ChatRequestDto(BaseModel):
        """DTO for chat requests"""
        question: str = Field(..., min_length=1, max_length=1000)
        context_project: Optional[str] = None
        context_author: Optional[str] = None
        context_days: int = Field(30, ge=1, le=365)
    
    class ChatResponseDto(BaseModel):
        """DTO for chat responses"""
        answer: str
        context_commits_count: int
        processing_time_ms: int
        model_used: str
    ```
    
    ### src/application/dto/webhook_dto.py
    ```python
    from pydantic import BaseModel
    from typing import List, Optional
    from datetime import datetime
    
    class WebhookAuthorDto(BaseModel):
        """DTO for webhook author data"""
        name: str
        email: str
        username: Optional[str] = None
    
    class WebhookCommitDto(BaseModel):
        """DTO for webhook commit data"""
        id: str
        message: str
        timestamp: datetime
        author: WebhookAuthorDto
        added: List[str] = []
        modified: List[str] = []
        removed: List[str] = []
        url: Optional[str] = None
    
    class WebhookRepositoryDto(BaseModel):
        """DTO for webhook repository data"""
        name: str
        full_name: str
        url: str
        default_branch: str
    
    class WebhookPayloadDto(BaseModel):
        """DTO for webhook payload"""
        ref: str  # refs/heads/main
        repository: WebhookRepositoryDto
        commits: List[WebhookCommitDto]
        pusher: WebhookAuthorDto
        head_commit: Optional[WebhookCommitDto] = None
    ```
    
    ### src/application/use_cases/process_commit.py
    ```python
    from typing import Optional
    from datetime import datetime
    from ..dto.commit_dto import CommitDto, CommitResponseDto
    from ..services.event_dispatcher import IEventDispatcher
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.entities.commit import Commit, CommitHash, FileChange
    from ...domain.events.commit_events import CommitReceivedEvent
    from ...application.interfaces.cache_interface import ICacheService
    from ...application.interfaces.messaging_interface import IMessageQueue
    
    class ProcessCommitUseCase:
        """Use case for processing incoming commits"""
        
        def __init__(
            self,
            commit_repository: ICommitRepository,
            event_dispatcher: IEventDispatcher,
            message_queue: IMessageQueue,
            cache_service: ICacheService
        ):
            self.commit_repository = commit_repository
            self.event_dispatcher = event_dispatcher
            self.message_queue = message_queue
            self.cache_service = cache_service
        
        async def execute(self, commit_dto: CommitDto) -> CommitResponseDto:
            """Process a new commit"""
            
            # Check if commit already exists (idempotency)
            existing = await self.commit_repository.find_by_hash(commit_dto.commit_hash)
            if existing:
                return self._to_response_dto(existing)
            
            # Create domain entity
            commit = self._to_domain(commit_dto)
            
            # Save to repository
            saved_commit = await self.commit_repository.save(commit)
            
            # Dispatch domain event
            event = CommitReceivedEvent(
                commit_id=saved_commit.id,
                commit_hash=saved_commit.commit_hash.value,
                project=saved_commit.project,
                author=saved_commit.author_email,
                branch=saved_commit.branch
            )
            await self.event_dispatcher.dispatch(event)
            
            # Queue for async analysis
            await self.message_queue.enqueue(
                "analyze_commit",
                {"commit_id": saved_commit.id}
            )
            
            # Invalidate relevant caches
            await self.cache_service.invalidate_pattern(f"commits:project:{commit.project}:*")
            await self.cache_service.invalidate_pattern(f"commits:author:{commit.author_email}:*")
            
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
    ```
    
    ### src/application/use_cases/search_commits.py
    ```python
    from typing import List
    from ..dto.commit_dto import CommitSearchDto, CommitResponseDto
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...application.interfaces.cache_interface import ICacheService
    import hashlib
    import json
    
    class SearchCommitsUseCase:
        """Use case for searching commits"""
        
        def __init__(
            self,
            commit_repository: ICommitRepository,
            cache_service: ICacheService
        ):
            self.commit_repository = commit_repository
            self.cache_service = cache_service
        
        async def execute(self, search_dto: CommitSearchDto) -> List[CommitResponseDto]:
            """Search commits with caching"""
            
            # Generate cache key
            cache_key = self._generate_cache_key(search_dto)
            
            # Try cache first
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                return cached_result
            
            # Search in repository
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
                limit=search_dto.limit
            )
            
            # Convert to DTOs
            result = [self._to_response_dto(commit) for commit in commits]
            
            # Cache result for 5 minutes
            await self.cache_service.set(cache_key, result, ttl=300)
            
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
    ```
    
    ### src/application/use_cases/chat_with_ai.py
    ```python
    from typing import List
    from datetime import datetime, timedelta
    import time
    from ..dto.commit_dto import ChatRequestDto, ChatResponseDto
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.services.ai_analyzer import IAIAnalyzer
    from ...application.interfaces.cache_interface import ICacheService
    import hashlib
    
    class ChatWithAIUseCase:
        """Use case for chatting with AI about commits"""
        
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
            """Answer question about commits using AI"""
            
            start_time = time.time()
            
            # Generate cache key
            cache_key = self._generate_cache_key(chat_dto)
            
            # Try cache first
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                return cached_result
            
            # Get context commits
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=chat_dto.context_days)
            
            context_commits = await self.commit_repository.find_by_project(
                project=chat_dto.context_project,
                start_date=start_date,
                end_date=end_date,
                limit=100
            )
            
            if chat_dto.context_author:
                author_commits = await self.commit_repository.find_by_author(
                    author_email=chat_dto.context_author,
                    project=chat_dto.context_project,
                    start_date=start_date,
                    end_date=end_date,
                    limit=50
                )
                context_commits.extend(author_commits)
            
            # Remove duplicates
            unique_commits = {commit.id: commit for commit in context_commits}
            context_commits = list(unique_commits.values())
            
            # Ask AI
            answer = await self.ai_analyzer.answer_question(
                question=chat_dto.question,
                context_commits=context_commits
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            result = ChatResponseDto(
                answer=answer,
                context_commits_count=len(context_commits),
                processing_time_ms=processing_time_ms,
                model_used=self.ai_analyzer.get_model_info()['name']
            )
            
            # Cache result for 10 minutes
            await self.cache_service.set(cache_key, result, ttl=600)
            
            return result
        
        def _generate_cache_key(self, chat_dto: ChatRequestDto) -> str:
            """Generate cache key for chat request"""
            chat_hash = hashlib.md5(
                f"{chat_dto.question}:{chat_dto.context_project}:{chat_dto.context_author}:{chat_dto.context_days}".encode()
            ).hexdigest()
    ### src/application/services/event_dispatcher.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, List, Callable, Any
    from ...domain.events.base import DomainEvent
    import asyncio
    import logging
    
    logger = logging.getLogger(__name__)
    
    class IEventDispatcher(ABC):
        """Event dispatcher interface"""
        
        @abstractmethod
        async def dispatch(self, event: DomainEvent):
            """Dispatch event to all registered handlers"""
            pass
        
        @abstractmethod
        def register_handler(self, event_type: str, handler: Callable):
            """Register event handler"""
            pass
    
    class EventDispatcher(IEventDispatcher):
        """Implementation of event dispatcher"""
        
        def __init__(self):
            self.handlers: Dict[str, List[Callable]] = {}
        
        def register_handler(self, event_type: str, handler: Callable):
            """Register event handler"""
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
            logger.info(f"Registered handler for event type: {event_type}")
        
        async def dispatch(self, event: DomainEvent):
            """Dispatch event to all registered handlers"""
            event_type = event.__class__.__name__
            
            if event_type not in self.handlers:
                logger.debug(f"No handlers registered for event type: {event_type}")
                return
            
            logger.info(f"Dispatching event: {event_type} (ID: {event.event_id})")
            
            tasks = []
            for handler in self.handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        # Run sync handler in thread pool
                        loop = asyncio.get_event_loop()
                        tasks.append(loop.run_in_executor(None, handler, event))
                except Exception as e:
                    logger.error(f"Error preparing handler for {event_type}: {str(e)}")
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log any handler exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Handler {i} failed for {event_type}: {str(result)}")
    
    ### src/application/interfaces/cache_interface.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Any, Optional, List
    
    class ICacheService(ABC):
        """Interface for cache service"""
        
        @abstractmethod
        async def get(self, key: str) -> Optional[Any]:
            """Get value from cache"""
            pass
        
        @abstractmethod
        async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
            """Set value in cache with TTL in seconds"""
            pass
        
        @abstractmethod
        async def delete(self, key: str) -> bool:
            """Delete key from cache"""
            pass
        
        @abstractmethod
        async def exists(self, key: str) -> bool:
            """Check if key exists in cache"""
            pass
        
        @abstractmethod
        async def invalidate_pattern(self, pattern: str) -> int:
            """Invalidate all keys matching pattern (with * wildcard)"""
            pass
        
        @abstractmethod
        async def get_many(self, keys: List[str]) -> Dict[str, Any]:
            """Get multiple values from cache"""
            pass
        
        @abstractmethod
        async def set_many(self, data: Dict[str, Any], ttl: int = 3600) -> bool:
            """Set multiple values in cache"""
            pass
    
    ### src/application/interfaces/ai_interface.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, List, Optional
    from ...domain.entities.analysis import AnalysisResult
    
    class IAIService(ABC):
        """Interface for AI service"""
        
        @abstractmethod
        async def analyze_text(
            self,
            text: str,
            context: Optional[str] = None,
            temperature: float = 0.7
        ) -> AnalysisResult:
            """Analyze text and return structured results"""
            pass
        
        @abstractmethod
        async def generate_response(
            self,
            prompt: str,
            context: Optional[str] = None,
            max_tokens: int = 500
        ) -> str:
            """Generate text response"""
            pass
        
        @abstractmethod
        async def create_embedding(self, text: str) -> List[float]:
            """Create text embedding"""
            pass
        
        @abstractmethod
        async def health_check(self) -> bool:
            """Check if AI service is healthy"""
            pass
        
        @abstractmethod
        def get_model_info(self) -> Dict[str, str]:
            """Get model information"""
            pass
    
    ### src/application/interfaces/messaging_interface.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, Optional, Callable
    from enum import Enum
    
    class TaskStatus(Enum):
        PENDING = "PENDING"
        STARTED = "STARTED"
        SUCCESS = "SUCCESS"
        FAILURE = "FAILURE"
        RETRY = "RETRY"
    
    class IMessageQueue(ABC):
        """Interface for message queue"""
        
        @abstractmethod
        async def enqueue(
            self,
            task_name: str,
            payload: Dict[str, Any],
            delay_seconds: int = 0,
            priority: int = 5
        ) -> str:
            """Enqueue task for processing"""
            pass
        
        @abstractmethod
        async def get_task_status(self, task_id: str) -> TaskStatus:
            """Get task status"""
            pass
        
        @abstractmethod
        async def get_task_result(self, task_id: str) -> Optional[Any]:
            """Get task result"""
            pass
        
        @abstractmethod
        async def cancel_task(self, task_id: str) -> bool:
            """Cancel pending task"""
            pass
        
        @abstractmethod
        def register_task(self, task_name: str, handler: Callable):
            """Register task handler"""
            pass
    
    ## 3. Infrastructure Layer
    
    ### src/infrastructure/persistence/database.py
    ```python
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy import MetaData
    from contextlib import asynccontextmanager
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    # SQLAlchemy base
    Base = declarative_base()
    metadata = MetaData()
    
    # Global engine and session maker
    engine = None
    SessionLocal = None
    
    def get_database_url() -> str:
        """Get database URL from environment"""
        db_provider = os.getenv("DB_PROVIDER", "sqlite")
        
        if db_provider == "sqlite":
            db_path = os.getenv("DB_PATH", "data/git_analytics.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            return f"sqlite+aiosqlite:///{db_path}"
        
        elif db_provider == "postgresql":
            host = os.getenv("DB_HOST", "localhost")
            port = os.getenv("DB_PORT", "5432")
            name = os.getenv("DB_NAME", "git_analytics")
            user = os.getenv("DB_USER", "postgres")
            password = os.getenv("DB_PASSWORD", "")
            return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"
        
        else:
            raise ValueError(f"Unsupported database provider: {db_provider}")
    
    async def init_database():
        """Initialize database connection"""
        global engine, SessionLocal
        
        if engine is None:
            database_url = get_database_url()
            logger.info(f"Connecting to database: {database_url.split('://')[0]}://...")
            
            engine = create_async_engine(
                database_url,
                echo=os.getenv("DB_ECHO", "false").lower() == "true",
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            SessionLocal = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database initialized successfully")
    
    async def close_database():
        """Close database connection"""
        global engine
        if engine:
            await engine.dispose()
            logger.info("Database connection closed")
    
    @asynccontextmanager
    async def get_session() -> AsyncSession:
        """Get database session"""
        if SessionLocal is None:
            await init_database()
        
        async with SessionLocal() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    ### src/infrastructure/persistence/models/commit_model.py
    ```python
    from sqlalchemy import Column, String, DateTime, Text, Integer, Float, JSON
    from sqlalchemy.sql import func
    from ..database import Base
    
    class CommitModel(Base):
        """SQLAlchemy model for commits"""
        
        __tablename__ = "commits"
        
        # Primary fields
        id = Column(String(36), primary_key=True)
        commit_hash = Column(String(40), unique=True, nullable=False, index=True)
        author_email = Column(String(255), nullable=False, index=True)
        author_name = Column(String(255), nullable=False)
        message = Column(Text, nullable=False)
        timestamp = Column(DateTime, nullable=False, index=True)
        branch = Column(String(255), nullable=False, index=True)
        project = Column(String(255), nullable=False, index=True)
        
        # File changes (stored as JSON)
        files_changed = Column(JSON, nullable=False, default=list)
        issue_numbers = Column(JSON, nullable=False, default=list)
        
        # Metrics
        total_lines_changed = Column(Integer, default=0)
        files_count = Column(Integer, default=0)
        complexity_score = Column(Float)
        impact_score = Column(Float)
        
        # Analysis results
        summary = Column(Text)
        tags = Column(JSON, default=list)
        sentiment_score = Column(Float)
        embedding_id = Column(String(255))
        analyzed_at = Column(DateTime)
        
        # Timestamps
        created_at = Column(DateTime, default=func.now(), nullable=False)
        updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
        
        def __repr__(self):
            return f"<CommitModel(hash={self.commit_hash}, project={self.project})>"
    
    class AnalysisModel(Base):
        """SQLAlchemy model for analysis results"""
        
        __tablename__ = "analyses"
        
        id = Column(String(36), primary_key=True)
        commit_id = Column(String(36), nullable=False, index=True)
        
        # Model information
        model_name = Column(String(100), nullable=False)
        model_version = Column(String(50), nullable=False)
        prompt_version = Column(String(50), nullable=False)
        
        # Analysis results
        summary = Column(Text, nullable=False)
        tags = Column(JSON, default=list)
        sentiment_score = Column(Float, nullable=False)
        confidence_score = Column(Float, nullable=False)
        extracted_entities = Column(JSON, default=list)
        
        # Performance metrics
        processing_time_ms = Column(Integer, nullable=False)
        tokens_used = Column(Integer, default=0)
        status = Column(String(20), default="completed")
        
        # Timestamps
        created_at = Column(DateTime, default=func.now(), nullable=False)
        
        def __repr__(self):
            return f"<AnalysisModel(commit_id={self.commit_id}, model={self.model_name})>"

Git AI Analytics - Complete Implementation

Code ∙ Version 5 

prompt = self.prompts.get\_summary\_prompt(commits) try: response = await self.\_generate\_response(prompt, temperature=0.5) return response.strip() except Exception as e: logger.error(

    ### src/infrastructure/messaging/tasks/analysis_tasks.py
    ```python
    from celery import shared_task
    from celery.utils.log import get_task_logger
    from typing import Dict, Any
    import traceback
    import asyncio
    from ...persistence.database import get_session
    from ...persistence.repositories.sqlite_commit_repository import SqliteCommitRepository
    from ...ai.ollama_service import OllamaService
    from ....domain.events.commit_events import (
        CommitAnalysisStartedEvent,
        CommitAnalysisCompletedEvent,
        CommitAnalysisFailedEvent
    )
    
    logger = get_task_logger(__name__)
    
    @shared_task(bind=True, name='analyze_commit')
    def analyze_commit_task(self, commit_id: str) -> Dict[str, Any]:
        """Analyze commit with AI"""
        try:
            logger.info(f"Starting analysis for commit: {commit_id}")
            
            # Run async analysis
            result = asyncio.run(_analyze_commit_async(commit_id))
            
            logger.info(f"Successfully analyzed commit: {commit_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze commit {commit_id}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fire failure event
            asyncio.run(_fire_analysis_failed_event(commit_id, str(e), self.request.retries))
            
            # Re-raise for Celery retry mechanism
            raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))
    
    async def _analyze_commit_async(commit_id: str) -> Dict[str, Any]:
        """Async function to analyze commit"""
        
        # Get dependencies
        async with get_session() as session:
            commit_repo = SqliteCommitRepository(session)
            ai_service = OllamaService()
            
            # Load commit
            commit = await commit_repo.find_by_id(commit_id)
            if not commit:
                raise ValueError(f"Commit {commit_id} not found")
            
            # Fire analysis started event
            started_event = CommitAnalysisStartedEvent(
                commit_id=commit_id,
                model_name=ai_service.get_model_info()['name']
            )
            # Note: In a full implementation, we'd dispatch this event
            
            # Generate AI analysis
            analysis_result = await ai_service.analyze_commit(commit)
            
            # Update commit with analysis
            commit.mark_as_analyzed(
                summary=analysis_result.summary,
                tags=analysis_result.tags,
                sentiment=analysis_result.sentiment_score
            )
            
            updated_commit = await commit_repo.update(commit)
            
            # Fire analysis completed event
            completed_event = CommitAnalysisCompletedEvent(
                commit_id=commit_id,
                analysis_id=f"analysis_{commit_id}",
                summary=analysis_result.summary,
                tags=analysis_result.tags,
                processing_time_ms=1000  # Placeholder
            )
            # Note: In a full implementation, we'd dispatch this event
            
            return {
                'status': 'success',
                'commit_id': commit_id,
                'summary': analysis_result.summary,
                'tags': analysis_result.tags,
                'sentiment_score': analysis_result.sentiment_score,
                'confidence_score': analysis_result.confidence_score
            }
    
    async def _fire_analysis_failed_event(commit_id: str, error_message: str, retry_count: int):
        """Fire analysis failed event"""
        failed_event = CommitAnalysisFailedEvent(
            commit_id=commit_id,
            error_message=error_message,
            retry_count=retry_count
        )
        # Note: In a full implementation, we'd dispatch this event
    
    @shared_task(name='generate_summary')
    def generate_summary_task(project: str, days: int = 7) -> Dict[str, Any]:
        """Generate project summary for recent commits"""
        try:
            logger.info(f"Generating summary for project: {project}")
            
            result = asyncio.run(_generate_summary_async(project, days))
            
            logger.info(f"Successfully generated summary for project: {project}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate summary for project {project}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def _generate_summary_async(project: str, days: int) -> Dict[str, Any]:
        """Async function to generate summary"""
        from datetime import datetime, timedelta
        
        async with get_session() as session:
            commit_repo = SqliteCommitRepository(session)
            ai_service = OllamaService()
            
            # Get recent commits
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            commits = await commit_repo.find_by_project(
                project=project,
                start_date=start_date,
                end_date=end_date,
                limit=50
            )
            
            if not commits:
                return {
                    'status': 'success',
                    'project': project,
                    'summary': f"No commits found for {project} in the last {days} days.",
                    'commits_count': 0
                }
            
            # Generate summary
            summary = await ai_service.generate_summary(commits)
            
            return {
                'status': 'success',
                'project': project,
                'summary': summary,
                'commits_count': len(commits),
                'period_days': days
            }
    
    ### src/infrastructure/events/event_bus.py
    ```python
    import asyncio
    import json
    import logging
    from typing import Callable, Dict, List
    from redis.asyncio import Redis
    from ...domain.events.base import DomainEvent
    from ...application.services.event_dispatcher import IEventDispatcher
    
    logger = logging.getLogger(__name__)
    
    class RedisEventBus(IEventDispatcher):
        """Redis-based event bus for distributed event handling"""
        
        def __init__(self, redis_client: Redis, channel_prefix: str = "events"):
            self.redis = redis_client
            self.channel_prefix = channel_prefix
            self.handlers: Dict[str, List[Callable]] = {}
            self.subscriber_task = None
            self._running = False
        
        async def start(self):
            """Start event bus subscriber"""
            if not self._running:
                self._running = True
                self.subscriber_task = asyncio.create_task(self._subscriber_loop())
                logger.info("### src/infrastructure/persistence/repositories/sqlite_commit_repository.py
    ```python
    from typing import List, Optional, Dict, Any
    from datetime import datetime
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import select, and_, or_, desc, func
    from sqlalchemy.orm import selectinload
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.entities.commit import Commit, CommitHash, FileChange
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
                    id=commit.id,
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
                logger.error(f"Error saving commit {commit.commit_hash.value}: {str(e)}")
                raise
        
        async def find_by_id(self, commit_id: str) -> Optional[Commit]:
            """Find commit by ID"""
            try:
                stmt = select(CommitModel).where(CommitModel.id == commit_id)
                result = await self.session.execute(stmt)
                db_commit = result.scalar_one_or_none()
                
                return self._to_domain(db_commit) if db_commit else None
            except Exception as e:
                logger.error(f"Error finding commit by ID {commit_id}: {str(e)}")
                return None
        
        async def find_by_hash(self, commit_hash: str) -> Optional[Commit]:
            """Find commit by hash"""
            try:
                stmt = select(CommitModel).where(CommitModel.commit_hash == commit_hash)
                result = await self.session.execute(stmt)
                db_commit = result.scalar_one_or_none()
                
                return self._to_domain(db_commit) if db_commit else None
            except Exception as e:
                logger.error(f"Error finding commit by hash {commit_hash}: {str(e)}")
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
                logger.error(f"Error finding commits by author {author_email}: {str(e)}")
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
                logger.error(f"Error finding commits by project {project}: {str(e)}")
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
                logger.error(f"Error searching commits with query '{query}': {str(e)}")
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
                logger.error(f"Error updating commit {commit.id}: {str(e)}")
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
                logger.error(f"Error getting statistics: {str(e)}")
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
    
    ### src/infrastructure/ai/ollama_service.py
    ```python
    import httpx
    import json
    import logging
    from typing import Dict, Any, List, Optional
    from ...domain.services.ai_analyzer import IAIAnalyzer
    from ...domain.entities.commit import Commit
    from ...domain.entities.analysis import AnalysisResult
    from .prompts.commit_prompts import CommitPrompts
    import time
    import os
    
    logger = logging.getLogger(__name__)
    
    class OllamaService(IAIAnalyzer):
        """Ollama AI service implementation"""
        
        def __init__(
            self,
            base_url: str = None,
            model: str = None,
            timeout: int = 30
        ):
            self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
            self.timeout = timeout
            self.prompts = CommitPrompts()
            
            # Remove trailing slash
            self.base_url = self.base_url.rstrip('/')
        
        async def analyze_commit(self, commit: Commit) -> AnalysisResult:
            """Analyze a commit using Ollama"""
            prompt = self.prompts.get_commit_analysis_prompt(commit)
            
            try:
                start_time = time.time()
                response = await self._generate_response(prompt, temperature=0.3)
                processing_time = int((time.time() - start_time) * 1000)
                
                # Parse structured response
                analysis_data = self._parse_analysis_response(response)
                
                return AnalysisResult(
                    summary=analysis_data.get('summary', 'No summary available'),
                    tags=analysis_data.get('tags', []),
                    sentiment_score=analysis_data.get('sentiment', 0.0),
                    confidence_score=analysis_data.get('confidence', 0.5),
                    extracted_entities=analysis_data.get('entities', [])
                )
                
            except Exception as e:
                logger.error(f"Error analyzing commit {commit.commit_hash.value}: {str(e)}")
                # Return default analysis on error
                return AnalysisResult(
                    summary=f"Analysis failed: {str(e)}",
                    tags=['error'],
                    sentiment_score=0.0,
                    confidence_score=0.0,
                    extracted_entities=[]
                )
        
        async def generate_summary(self, commits: List[Commit]) -> str:
            """Generate summary for multiple commits"""
            if not commits:
                return "No commits to summarize."
            
            # Git AI Analytics - Full Implementation
    
    ## Project Structure
    ```
    git-ai-analytics/
    ├── src/
    │   ├── domain/                    # Domain Layer
    │   │   ├── entities/
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py
    │   │   │   ├── commit.py
    │   │   │   ├── developer.py
    │   │   │   └── analysis.py
    │   │   ├── events/
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py
    │   │   │   └── commit_events.py
    │   │   ├── repositories/
    │   │   │   ├── __init__.py
    │   │   │   ├── commit_repository.py
    │   │   │   └── analysis_repository.py
    │   │   └── services/
    │   │       ├── __init__.py
    │   │       └── ai_analyzer.py
    │   ├── application/               # Application Layer
    │   │   ├── use_cases/
    │   │   │   ├── __init__.py
    │   │   │   ├── process_commit.py
    │   │   │   ├── analyze_commit.py
    │   │   │   └── search_commits.py
    │   │   ├── dto/
    │   │   │   ├── __init__.py
    │   │   │   ├── commit_dto.py
    │   │   │   └── webhook_dto.py
    │   │   ├── services/
    │   │   │   ├── __init__.py
    │   │   │   └── event_dispatcher.py
    │   │   └── interfaces/
    │   │       ├── __init__.py
    │   │       ├── cache_interface.py
    │   │       ├── ai_interface.py
    │   │       └── messaging_interface.py
    │   ├── infrastructure/           # Infrastructure Layer
    │   │   ├── persistence/
    │   │   │   ├── __init__.py
    │   │   │   ├── models/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── commit_model.py
    │   │   │   ├── repositories/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── sqlite_commit_repository.py
    │   │   │   └── database.py
    │   │   ├── ai/
    │   │   │   ├── __init__.py
    │   │   │   ├── ollama_service.py
    │   │   │   └── prompts/
    │   │   │       └── commit_prompts.py
    │   │   ├── cache/
    │   │   │   ├── __init__.py
    │   │   │   ├── memory_cache.py
    │   │   │   └── redis_cache.py
    │   │   ├── messaging/
    │   │   │   ├── __init__.py
    │   │   │   ├── celery_config.py
    │   │   │   └── tasks/
    │   │   │       ├── __init__.py
    │   │   │       └── analysis_tasks.py
    │   │   └── events/
    │   │       ├── __init__.py
    │   │       └── event_bus.py
    │   ├── presentation/             # Presentation Layer
    │   │   ├── api/
    │   │   │   ├── __init__.py
    │   │   │   ├── v1/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── webhooks.py
    │   │   │   │   ├── commits.py
    │   │   │   │   └── chat.py
    │   │   │   └── dependencies.py
    │   │   └── websocket/
    │   │       ├── __init__.py
    │   │       └── handlers.py
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── config.py
    │   │   └── logging.py
    │   └── main.py
    ├── configs/
    │   ├── default.yaml
    │   ├── development.yaml
    │   └── production.yaml
    ├── tests/
    ├── migrations/
    ├── docker/
    │   ├── Dockerfile
    │   └── docker-compose.yml
    ├── requirements/
    │   ├── base.txt
    │   ├── development.txt
    │   └── production.txt
    ├── .env.example
    └── README.md
    ```
    
    ## 1. Domain Layer
    
    ### src/domain/entities/base.py
    ```python
    from abc import ABC
    from datetime import datetime
    from typing import Any
    from uuid import uuid4, UUID
    from dataclasses import dataclass, field
    
    @dataclass
    class ValueObject(ABC):
        """Base class for value objects"""
        
        def __post_init__(self):
            self._validate()
        
        def _validate(self):
            """Override to add validation logic"""
            pass
    
    class Entity(ABC):
        """Base class for entities"""
        
        def __init__(self):
            self.id: str = str(uuid4())
            self.created_at: datetime = datetime.utcnow()
            self.updated_at: datetime = datetime.utcnow()
        
        def __eq__(self, other):
            if not isinstance(other, self.__class__):
                return False
            return self.id == other.id
        
        def __hash__(self):
            return hash(self.id)
    ```
    
    ### src/domain/entities/commit.py
    ```python
    from dataclasses import dataclass
    from datetime import datetime
    from typing import List, Optional
    from .base import Entity, ValueObject
    
    @dataclass
    class CommitHash(ValueObject):
        """Value object for commit hash"""
        value: str
        
        def _validate(self):
            if not self.value or len(self.value) < 6:
                raise ValueError("Invalid commit hash")
    
    @dataclass
    class FileChange(ValueObject):
        """Value object for file changes"""
        filename: str
        additions: int
        deletions: int
        status: str  # added, modified, deleted, renamed
        
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
        """Commit domain entity"""
        
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
            self.commit_hash = commit_hash
            self.author_email = author_email
            self.author_name = author_name
            self.message = message
            self.timestamp = timestamp
            self.branch = branch
            self.project = project
            self.files_changed = files_changed or []
            self.issue_numbers = issue_numbers or []
            
            # Analysis results (populated later)
            self.summary: Optional[str] = None
            self.tags: List[str] = []
            self.sentiment_score: Optional[float] = None
            self.embedding_id: Optional[str] = None
            self.analyzed_at: Optional[datetime] = None
            
            # Computed properties
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
    ```
    
    ### src/domain/entities/analysis.py
    ```python
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, List, Optional
    from .base import Entity, ValueObject
    
    @dataclass
    class AnalysisResult(ValueObject):
        """Value object for AI analysis results"""
        summary: str
        tags: List[str]
        sentiment_score: float
        confidence_score: float
        extracted_entities: List[str]
        
        def _validate(self):
            if not (-1 <= self.sentiment_score <= 1):
                raise ValueError("Sentiment score must be between -1 and 1")
            if not (0 <= self.confidence_score <= 1):
                raise ValueError("Confidence score must be between 0 and 1")
    
    class Analysis(Entity):
        """Analysis domain entity"""
        
        def __init__(
            self,
            commit_id: str,
            model_name: str,
            model_version: str,
            prompt_version: str,
            result: AnalysisResult,
            processing_time_ms: int,
            tokens_used: int = 0
        ):
            super().__init__()
            self.commit_id = commit_id
            self.model_name = model_name
            self.model_version = model_version
            self.prompt_version = prompt_version
            self.result = result
            self.processing_time_ms = processing_time_ms
            self.tokens_used = tokens_used
            self.status = "completed"
        
        def is_high_confidence(self) -> bool:
            """Check if analysis has high confidence"""
            return self.result.confidence_score >= 0.8
    ```
    
    ### src/domain/events/base.py
    ```python
    from abc import ABC
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, Any
    import json
    from uuid import uuid4
    
    @dataclass
    class DomainEvent(ABC):
        """Base class for domain events"""
        event_id: str = None
        occurred_at: datetime = None
        
        def __post_init__(self):
            if self.event_id is None:
                self.event_id = str(uuid4())
            if self.occurred_at is None:
                self.occurred_at = datetime.utcnow()
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert event to dictionary"""
            return {
                'event_type': self.__class__.__name__,
                'event_id': self.event_id,
                'occurred_at': self.occurred_at.isoformat(),
                'data': self._get_event_data()
            }
        
        def to_json(self) -> str:
            """Convert event to JSON string"""
            return json.dumps(self.to_dict(), default=str)
        
        def _get_event_data(self) -> Dict[str, Any]:
            """Override to provide event-specific data"""
            return {}
    ```
    
    ### src/domain/events/commit_events.py
    ```python
    from dataclasses import dataclass
    from typing import Dict, Any, List
    from .base import DomainEvent
    
    @dataclass
    class CommitReceivedEvent(DomainEvent):
        """Event fired when a new commit is received"""
        commit_id: str
        commit_hash: str
        project: str
        author: str
        branch: str
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'commit_hash': self.commit_hash,
                'project': self.project,
                'author': self.author,
                'branch': self.branch
            }
    
    @dataclass 
    class CommitAnalysisStartedEvent(DomainEvent):
        """Event fired when commit analysis starts"""
        commit_id: str
        model_name: str
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'model_name': self.model_name
            }
    
    @dataclass
    class CommitAnalysisCompletedEvent(DomainEvent):
        """Event fired when commit analysis completes"""
        commit_id: str
        analysis_id: str
        summary: str
        tags: List[str]
        processing_time_ms: int
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'analysis_id': self.analysis_id,
                'summary': self.summary,
                'tags': self.tags,
                'processing_time_ms': self.processing_time_ms
            }
    
    @dataclass
    class CommitAnalysisFailedEvent(DomainEvent):
        """Event fired when commit analysis fails"""
        commit_id: str
        error_message: str
        retry_count: int
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'error_message': self.error_message,
                'retry_count': self.retry_count
            }
    ```
    
    ### src/domain/repositories/commit_repository.py
    ```python
    from abc import ABC, abstractmethod
    from typing import List, Optional, Dict, Any
    from datetime import datetime
    from ..entities.commit import Commit
    
    class ICommitRepository(ABC):
        """Repository interface for Commit entity"""
        
        @abstractmethod
        async def save(self, commit: Commit) -> Commit:
            """Save a commit"""
            pass
        
        @abstractmethod
        async def find_by_id(self, commit_id: str) -> Optional[Commit]:
            """Find commit by ID"""
            pass
        
        @abstractmethod
        async def find_by_hash(self, commit_hash: str) -> Optional[Commit]:
            """Find commit by hash"""
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
    ```
    
    ### src/domain/services/ai_analyzer.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, List
    from ..entities.commit import Commit
    from ..entities.analysis import AnalysisResult
    
    class IAIAnalyzer(ABC):
        """Domain service interface for AI analysis"""
        
        @abstractmethod
        async def analyze_commit(self, commit: Commit) -> AnalysisResult:
            """Analyze a commit using AI"""
            pass
        
        @abstractmethod
        async def generate_summary(self, commits: List[Commit]) -> str:
            """Generate summary for multiple commits"""
            pass
        
        @abstractmethod
        async def answer_question(
            self, 
            question: str, 
            context_commits: List[Commit]
        ) -> str:
            """Answer question about commits"""
            pass
        
        @abstractmethod
        async def extract_entities(self, text: str) -> List[str]:
            """Extract named entities from text"""
            pass
        
        @abstractmethod
        def get_model_info(self) -> Dict[str, str]:
            """Get current model information"""
            pass
    ```
    
    ## 2. Application Layer
    
    ### src/application/dto/commit_dto.py
    ```python
    from pydantic import BaseModel, Field, validator
    from datetime import datetime
    from typing import List, Optional
    
    class FileChangeDto(BaseModel):
        """DTO for file changes"""
        filename: str = Field(..., min_length=1)
        additions: int = Field(..., ge=0)
        deletions: int = Field(..., ge=0)
        status: str = Field(..., regex="^(added|modified|deleted|renamed)$")
    
    class CommitDto(BaseModel):
        """DTO for commit data"""
        commit_hash: str = Field(..., min_length=6)
        author_email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
        author_name: str = Field(..., min_length=1)
        message: str = Field(..., min_length=1)
        timestamp: datetime
        branch: str = Field(..., min_length=1)
        project: str = Field(..., min_length=1)
        files_changed: List[FileChangeDto] = []
        issue_numbers: List[str] = []
    
    class CommitResponseDto(BaseModel):
        """DTO for commit response"""
        id: str
        commit_hash: str
        author_email: str
        author_name: str
        message: str
        timestamp: datetime
        branch: str
        project: str
        files_count: int
        total_lines_changed: int
        complexity_score: Optional[float] = None
        summary: Optional[str] = None
        tags: List[str] = []
        sentiment_score: Optional[float] = None
        analyzed_at: Optional[datetime] = None
        created_at: datetime
    
    class CommitSearchDto(BaseModel):
        """DTO for commit search"""
        query: Optional[str] = None
        project: Optional[str] = None
        author: Optional[str] = None
        branch: Optional[str] = None
        start_date: Optional[datetime] = None
        end_date: Optional[datetime] = None
        tags: List[str] = []
        limit: int = Field(50, ge=1, le=200)
        offset: int = Field(0, ge=0)
    
    class ChatRequestDto(BaseModel):
        """DTO for chat requests"""
        question: str = Field(..., min_length=1, max_length=1000)
        context_project: Optional[str] = None
        context_author: Optional[str] = None
        context_days: int = Field(30, ge=1, le=365)
    
    class ChatResponseDto(BaseModel):
        """DTO for chat responses"""
        answer: str
        context_commits_count: int
        processing_time_ms: int
        model_used: str
    ```
    
    ### src/application/dto/webhook_dto.py
    ```python
    from pydantic import BaseModel
    from typing import List, Optional
    from datetime import datetime
    
    class WebhookAuthorDto(BaseModel):
        """DTO for webhook author data"""
        name: str
        email: str
        username: Optional[str] = None
    
    class WebhookCommitDto(BaseModel):
        """DTO for webhook commit data"""
        id: str
        message: str
        timestamp: datetime
        author: WebhookAuthorDto
        added: List[str] = []
        modified: List[str] = []
        removed: List[str] = []
        url: Optional[str] = None
    
    class WebhookRepositoryDto(BaseModel):
        """DTO for webhook repository data"""
        name: str
        full_name: str
        url: str
        default_branch: str
    
    class WebhookPayloadDto(BaseModel):
        """DTO for webhook payload"""
        ref: str  # refs/heads/main
        repository: WebhookRepositoryDto
        commits: List[WebhookCommitDto]
        pusher: WebhookAuthorDto
        head_commit: Optional[WebhookCommitDto] = None
    ```
    
    ### src/application/use_cases/process_commit.py
    ```python
    from typing import Optional
    from datetime import datetime
    from ..dto.commit_dto import CommitDto, CommitResponseDto
    from ..services.event_dispatcher import IEventDispatcher
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.entities.commit import Commit, CommitHash, FileChange
    from ...domain.events.commit_events import CommitReceivedEvent
    from ...application.interfaces.cache_interface import ICacheService
    from ...application.interfaces.messaging_interface import IMessageQueue
    
    class ProcessCommitUseCase:
        """Use case for processing incoming commits"""
        
        def __init__(
            self,
            commit_repository: ICommitRepository,
            event_dispatcher: IEventDispatcher,
            message_queue: IMessageQueue,
            cache_service: ICacheService
        ):
            self.commit_repository = commit_repository
            self.event_dispatcher = event_dispatcher
            self.message_queue = message_queue
            self.cache_service = cache_service
        
        async def execute(self, commit_dto: CommitDto) -> CommitResponseDto:
            """Process a new commit"""
            
            # Check if commit already exists (idempotency)
            existing = await self.commit_repository.find_by_hash(commit_dto.commit_hash)
            if existing:
                return self._to_response_dto(existing)
            
            # Create domain entity
            commit = self._to_domain(commit_dto)
            
            # Save to repository
            saved_commit = await self.commit_repository.save(commit)
            
            # Dispatch domain event
            event = CommitReceivedEvent(
                commit_id=saved_commit.id,
                commit_hash=saved_commit.commit_hash.value,
                project=saved_commit.project,
                author=saved_commit.author_email,
                branch=saved_commit.branch
            )
            await self.event_dispatcher.dispatch(event)
            
            # Queue for async analysis
            await self.message_queue.enqueue(
                "analyze_commit",
                {"commit_id": saved_commit.id}
            )
            
            # Invalidate relevant caches
            await self.cache_service.invalidate_pattern(f"commits:project:{commit.project}:*")
            await self.cache_service.invalidate_pattern(f"commits:author:{commit.author_email}:*")
            
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
    ```
    
    ### src/application/use_cases/search_commits.py
    ```python
    from typing import List
    from ..dto.commit_dto import CommitSearchDto, CommitResponseDto
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...application.interfaces.cache_interface import ICacheService
    import hashlib
    import json
    
    class SearchCommitsUseCase:
        """Use case for searching commits"""
        
        def __init__(
            self,
            commit_repository: ICommitRepository,
            cache_service: ICacheService
        ):
            self.commit_repository = commit_repository
            self.cache_service = cache_service
        
        async def execute(self, search_dto: CommitSearchDto) -> List[CommitResponseDto]:
            """Search commits with caching"""
            
            # Generate cache key
            cache_key = self._generate_cache_key(search_dto)
            
            # Try cache first
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                return cached_result
            
            # Search in repository
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
                limit=search_dto.limit
            )
            
            # Convert to DTOs
            result = [self._to_response_dto(commit) for commit in commits]
            
            # Cache result for 5 minutes
            await self.cache_service.set(cache_key, result, ttl=300)
            
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
    ```
    
    ### src/application/use_cases/chat_with_ai.py
    ```python
    from typing import List
    from datetime import datetime, timedelta
    import time
    from ..dto.commit_dto import ChatRequestDto, ChatResponseDto
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.services.ai_analyzer import IAIAnalyzer
    from ...application.interfaces.cache_interface import ICacheService
    import hashlib
    
    class ChatWithAIUseCase:
        """Use case for chatting with AI about commits"""
        
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
            """Answer question about commits using AI"""
            
            start_time = time.time()
            
            # Generate cache key
            cache_key = self._generate_cache_key(chat_dto)
            
            # Try cache first
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                return cached_result
            
            # Get context commits
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=chat_dto.context_days)
            
            context_commits = await self.commit_repository.find_by_project(
                project=chat_dto.context_project,
                start_date=start_date,
                end_date=end_date,
                limit=100
            )
            
            if chat_dto.context_author:
                author_commits = await self.commit_repository.find_by_author(
                    author_email=chat_dto.context_author,
                    project=chat_dto.context_project,
                    start_date=start_date,
                    end_date=end_date,
                    limit=50
                )
                context_commits.extend(author_commits)
            
            # Remove duplicates
            unique_commits = {commit.id: commit for commit in context_commits}
            context_commits = list(unique_commits.values())
            
            # Ask AI
            answer = await self.ai_analyzer.answer_question(
                question=chat_dto.question,
                context_commits=context_commits
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            result = ChatResponseDto(
                answer=answer,
                context_commits_count=len(context_commits),
                processing_time_ms=processing_time_ms,
                model_used=self.ai_analyzer.get_model_info()['name']
            )
            
            # Cache result for 10 minutes
            await self.cache_service.set(cache_key, result, ttl=600)
            
            return result
        
        def _generate_cache_key(self, chat_dto: ChatRequestDto) -> str:
            """Generate cache key for chat request"""
            chat_hash = hashlib.md5(
                f"{chat_dto.question}:{chat_dto.context_project}:{chat_dto.context_author}:{chat_dto.context_days}".encode()
            ).hexdigest()
    ### src/application/services/event_dispatcher.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, List, Callable, Any
    from ...domain.events.base import DomainEvent
    import asyncio
    import logging
    
    logger = logging.getLogger(__name__)
    
    class IEventDispatcher(ABC):
        """Event dispatcher interface"""
        
        @abstractmethod
        async def dispatch(self, event: DomainEvent):
            """Dispatch event to all registered handlers"""
            pass
        
        @abstractmethod
        def register_handler(self, event_type: str, handler: Callable):
            """Register event handler"""
            pass
    
    class EventDispatcher(IEventDispatcher):
        """Implementation of event dispatcher"""
        
        def __init__(self):
            self.handlers: Dict[str, List[Callable]] = {}
        
        def register_handler(self, event_type: str, handler: Callable):
            """Register event handler"""
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
            logger.info(f"Registered handler for event type: {event_type}")
        
        async def dispatch(self, event: DomainEvent):
            """Dispatch event to all registered handlers"""
            event_type = event.__class__.__name__
            
            if event_type not in self.handlers:
                logger.debug(f"No handlers registered for event type: {event_type}")
                return
            
            logger.info(f"Dispatching event: {event_type} (ID: {event.event_id})")
            
            tasks = []
            for handler in self.handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        # Run sync handler in thread pool
                        loop = asyncio.get_event_loop()
                        tasks.append(loop.run_in_executor(None, handler, event))
                except Exception as e:
                    logger.error(f"Error preparing handler for {event_type}: {str(e)}")
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log any handler exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Handler {i} failed for {event_type}: {str(result)}")
    
    ### src/application/interfaces/cache_interface.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Any, Optional, List
    
    class ICacheService(ABC):
        """Interface for cache service"""
        
        @abstractmethod
        async def get(self, key: str) -> Optional[Any]:
            """Get value from cache"""
            pass
        
        @abstractmethod
        async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
            """Set value in cache with TTL in seconds"""
            pass
        
        @abstractmethod
        async def delete(self, key: str) -> bool:
            """Delete key from cache"""
            pass
        
        @abstractmethod
        async def exists(self, key: str) -> bool:
            """Check if key exists in cache"""
            pass
        
        @abstractmethod
        async def invalidate_pattern(self, pattern: str) -> int:
            """Invalidate all keys matching pattern (with * wildcard)"""
            pass
        
        @abstractmethod
        async def get_many(self, keys: List[str]) -> Dict[str, Any]:
            """Get multiple values from cache"""
            pass
        
        @abstractmethod
        async def set_many(self, data: Dict[str, Any], ttl: int = 3600) -> bool:
            """Set multiple values in cache"""
            pass
    
    ### src/application/interfaces/ai_interface.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, List, Optional
    from ...domain.entities.analysis import AnalysisResult
    
    class IAIService(ABC):
        """Interface for AI service"""
        
        @abstractmethod
        async def analyze_text(
            self,
            text: str,
            context: Optional[str] = None,
            temperature: float = 0.7
        ) -> AnalysisResult:
            """Analyze text and return structured results"""
            pass
        
        @abstractmethod
        async def generate_response(
            self,
            prompt: str,
            context: Optional[str] = None,
            max_tokens: int = 500
        ) -> str:
            """Generate text response"""
            pass
        
        @abstractmethod
        async def create_embedding(self, text: str) -> List[float]:
            """Create text embedding"""
            pass
        
        @abstractmethod
        async def health_check(self) -> bool:
            """Check if AI service is healthy"""
            pass
        
        @abstractmethod
        def get_model_info(self) -> Dict[str, str]:
            """Get model information"""
            pass
    
    ### src/application/interfaces/messaging_interface.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, Optional, Callable
    from enum import Enum
    
    class TaskStatus(Enum):
        PENDING = "PENDING"
        STARTED = "STARTED"
        SUCCESS = "SUCCESS"
        FAILURE = "FAILURE"
        RETRY = "RETRY"
    
    class IMessageQueue(ABC):
        """Interface for message queue"""
        
        @abstractmethod
        async def enqueue(
            self,
            task_name: str,
            payload: Dict[str, Any],
            delay_seconds: int = 0,
            priority: int = 5
        ) -> str:
            """Enqueue task for processing"""
            pass
        
        @abstractmethod
        async def get_task_status(self, task_id: str) -> TaskStatus:
            """Get task status"""
            pass
        
        @abstractmethod
        async def get_task_result(self, task_id: str) -> Optional[Any]:
            """Get task result"""
            pass
        
        @abstractmethod
        async def cancel_task(self, task_id: str) -> bool:
            """Cancel pending task"""
            pass
        
        @abstractmethod
        def register_task(self, task_name: str, handler: Callable):
            """Register task handler"""
            pass
    
    ## 3. Infrastructure Layer
    
    ### src/infrastructure/persistence/database.py
    ```python
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy import MetaData
    from contextlib import asynccontextmanager
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    # SQLAlchemy base
    Base = declarative_base()
    metadata = MetaData()
    
    # Global engine and session maker
    engine = None
    SessionLocal = None
    
    def get_database_url() -> str:
        """Get database URL from environment"""
        db_provider = os.getenv("DB_PROVIDER", "sqlite")
        
        if db_provider == "sqlite":
            db_path = os.getenv("DB_PATH", "data/git_analytics.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            return f"sqlite+aiosqlite:///{db_path}"
        
        elif db_provider == "postgresql":
            host = os.getenv("DB_HOST", "localhost")
            port = os.getenv("DB_PORT", "5432")
            name = os.getenv("DB_NAME", "git_analytics")
            user = os.getenv("DB_USER", "postgres")
            password = os.getenv("DB_PASSWORD", "")
            return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"
        
        else:
            raise ValueError(f"Unsupported database provider: {db_provider}")
    
    async def init_database():
        """Initialize database connection"""
        global engine, SessionLocal
        
        if engine is None:
            database_url = get_database_url()
            logger.info(f"Connecting to database: {database_url.split('://')[0]}://...")
            
            engine = create_async_engine(
                database_url,
                echo=os.getenv("DB_ECHO", "false").lower() == "true",
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            SessionLocal = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database initialized successfully")
    
    async def close_database():
        """Close database connection"""
        global engine
        if engine:
            await engine.dispose()
            logger.info("Database connection closed")
    
    @asynccontextmanager
    async def get_session() -> AsyncSession:
        """Get database session"""
        if SessionLocal is None:
            await init_database()
        
        async with SessionLocal() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    ### src/infrastructure/persistence/models/commit_model.py
    ```python
    from sqlalchemy import Column, String, DateTime, Text, Integer, Float, JSON
    from sqlalchemy.sql import func
    from ..database import Base
    
    class CommitModel(Base):
        """SQLAlchemy model for commits"""
        
        __tablename__ = "commits"
        
        # Primary fields
        id = Column(String(36), primary_key=True)
        commit_hash = Column(String(40), unique=True, nullable=False, index=True)
        author_email = Column(String(255), nullable=False, index=True)
        author_name = Column(String(255), nullable=False)
        message = Column(Text, nullable=False)
        timestamp = Column(DateTime, nullable=False, index=True)
        branch = Column(String(255), nullable=False, index=True)
        project = Column(String(255), nullable=False, index=True)
        
        # File changes (stored as JSON)
        files_changed = Column(JSON, nullable=False, default=list)
        issue_numbers = Column(JSON, nullable=False, default=list)
        
        # Metrics
        total_lines_changed = Column(Integer, default=0)
        files_count = Column(Integer, default=0)
        complexity_score = Column(Float)
        impact_score = Column(Float)
        
        # Analysis results
        summary = Column(Text)
        tags = Column(JSON, default=list)
        sentiment_score = Column(Float)
        embedding_id = Column(String(255))
        analyzed_at = Column(DateTime)
        
        # Timestamps
        created_at = Column(DateTime, default=func.now(), nullable=False)
        updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
        
        def __repr__(self):
            return f"<CommitModel(hash={self.commit_hash}, project={self.project})>"
    
    class AnalysisModel(Base):
        """SQLAlchemy model for analysis results"""
        
        __tablename__ = "analyses"
        
        id = Column(String(36), primary_key=True)
        commit_id = Column(String(36), nullable=False, index=True)
        
        # Model information
        model_name = Column(String(100), nullable=False)
        model_version = Column(String(50), nullable=False)
        prompt_version = Column(String(50), nullable=False)
        
        # Analysis results
        summary = Column(Text, nullable=False)
        tags = Column(JSON, default=list)
        sentiment_score = Column(Float, nullable=False)
        confidence_score = Column(Float, nullable=False)
        extracted_entities = Column(JSON, default=list)
        
        # Performance metrics
        processing_time_ms = Column(Integer, nullable=False)
        tokens_used = Column(Integer, default=0)
        status = Column(String(20), default="completed")
        
        # Timestamps
        created_at = Column(DateTime, default=func.now(), nullable=False)
        
        def __repr__(self):
            return f"<AnalysisModel(commit_id={self.commit_id}, model={self.model_name})>"

Git AI Analytics - Complete Implementation

Code ∙ Version 6 

\### src/infrastructure/messaging/tasks/analysis\_tasks.py \`\`\`python from celery import shared\_task from celery.utils.log import get\_task\_logger from typing import Dict, Any import traceback import asyncio from ...persistence.database import get\_sessio

Retry

---

You asked:
----------

LL

Continue

Edit

---

Claude Replied:
---------------

    Redis event bus started")
        
        async def stop(self):
            """Stop event bus subscriber"""
            self._running = False
            if self.subscriber_task:
                self.subscriber_task.cancel()
                try:
                    await self.subscriber_task
                except asyncio.CancelledError:
                    pass
            logger.info("Redis event bus stopped")
        
        def register_handler(self, event_type: str, handler: Callable):
            """Register event handler"""
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
            logger.info(f"Registered Redis handler for event type: {event_type}")
        
        async def dispatch(self, event: DomainEvent):
            """Dispatch event to Redis"""
            try:
                event_type = event.__class__.__name__
                channel = f"{self.channel_prefix}:{event_type}"
                
                message = event.to_json()
                await self.redis.publish(channel, message)
                
                logger.debug(f"Published event {event_type} to Redis channel: {channel}")
                
            except Exception as e:
                logger.error(f"Error publishing event to Redis: {str(e)}")
        
        async def _subscriber_loop(self):
            """Main subscriber loop"""
            try:
                pubsub = self.redis.pubsub()
                
                # Subscribe to all event channels
                pattern = f"{self.channel_prefix}:*"
                await pubsub.psubscribe(pattern)
                
                logger.info(f"Subscribed to Redis pattern: {pattern}")
                
                async for message in pubsub.listen():
                    if not self._running:
                        break
                        
                    if message['type'] == 'pmessage':
                        await self._handle_message(message)
                        
            except asyncio.CancelledError:
                logger.info("Redis subscriber loop cancelled")
            except Exception as e:
                logger.error(f"Error in Redis subscriber loop: {str(e)}")
            finally:
                try:
                    await pubsub.unsubscribe()
                    await pubsub.close()
                except Exception:
                    pass
        
        async def _handle_message(self, message):
            """Handle incoming Redis message"""
            try:
                channel = message['channel'].decode('utf-8')
                event_type = channel.split(':')[-1]
                
                if event_type not in self.handlers:
                    return
                
                # Parse event data
                event_data = json.loads(message['data'].decode('utf-8'))
                
                # Execute handlers
                tasks = []
                for handler in self.handlers[event_type]:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event_data))
                    else:
                        loop = asyncio.get_event_loop()
                        tasks.append(loop.run_in_executor(None, handler, event_data))
                
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Log handler exceptions
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            logger.error(f"Redis handler {i} failed for {event_type}: {str(result)}")
                            
            except Exception as e:
                logger.error(f"Error handling Redis message: {str(e)}")
    
    ## 4. Presentation Layer
    
    ### src/presentation/api/dependencies.py
    ```python
    from typing import AsyncGenerator
    from fastapi import Depends, HTTPException, status
    from sqlalchemy.ext.asyncio import AsyncSession
    from redis.asyncio import Redis
    
    # Import domain interfaces
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.services.ai_analyzer import IAIAnalyzer
    
    # Import application interfaces  
    from ...application.interfaces.cache_interface import ICacheService
    from ...application.interfaces.messaging_interface import IMessageQueue
    from ...application.services.event_dispatcher import IEventDispatcher
    
    # Import infrastructure implementations
    from ...infrastructure.persistence.database import get_session
    from ...infrastructure.persistence.repositories.sqlite_commit_repository import SqliteCommitRepository
    from ...infrastructure.ai.ollama_service import OllamaService
    from ...infrastructure.cache.memory_cache import MemoryCacheService
    from ...infrastructure.cache.redis_cache import RedisCacheService
    from ...infrastructure.messaging.celery_config import celery_app, CeleryMessageQueue
    from ...infrastructure.events.event_bus import RedisEventBus
    
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Global singletons
    _redis_client = None
    _cache_service = None
    _event_dispatcher = None
    _message_queue = None
    _ai_service = None
    
    async def get_redis_client() -> Redis:
        """Get Redis client singleton"""
        global _redis_client
        if _redis_client is None:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            try:
                _redis_client = Redis.from_url(redis_url, decode_responses=False)
                # Test connection
                await _redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {str(e)}, falling back to memory cache")
                _redis_client = None
        return _redis_client
    
    async def get_cache_service() -> ICacheService:
        """Get cache service singleton"""
        global _cache_service
        if _cache_service is None:
            redis_client = await get_redis_client()
            if redis_client:
                _cache_service = RedisCacheService(redis_client)
            else:
                _cache_service = MemoryCacheService()
            logger.info(f"Cache service initialized: {type(_cache_service).__name__}")
        return _cache_service
    
    async def get_message_queue() -> IMessageQueue:
        """Get message queue singleton"""
        global _message_queue
        if _message_queue is None:
            _message_queue = CeleryMessageQueue(celery_app)
            logger.info("Message queue initialized")
        return _message_queue
    
    async def get_event_dispatcher() -> IEventDispatcher:
        """Get event dispatcher singleton"""
        global _event_dispatcher
        if _event_dispatcher is None:
            redis_client = await get_redis_client()
            if redis_client:
                _event_dispatcher = RedisEventBus(redis_client)
                await _event_dispatcher.start()
            else:
                from ...application.services.event_dispatcher import EventDispatcher
                _event_dispatcher = EventDispatcher()
            logger.info(f"Event dispatcher initialized: {type(_event_dispatcher).__name__}")
        return _event_dispatcher
    
    async def get_ai_service() -> IAIAnalyzer:
        """Get AI service singleton"""
        global _ai_service
        if _ai_service is None:
            _ai_service = OllamaService()
            # Test AI service health
            try:
                health = await _ai_service.health_check()
                if not health:
                    logger.warning("AI service health check failed")
            except Exception as e:
                logger.warning(f"AI service health check error: {str(e)}")
            logger.info("AI service initialized")
        return _ai_service
    
    async def get_commit_repository(
        session: AsyncSession = Depends(get_session)
    ) -> ICommitRepository:
        """Get commit repository with injected session"""
        return SqliteCommitRepository(session)
    
    # Use case dependencies
    async def get_process_commit_use_case(
        repo: ICommitRepository = Depends(get_commit_repository),
        dispatcher: IEventDispatcher = Depends(get_event_dispatcher),
        message_queue: IMessageQueue = Depends(get_message_queue),
        cache: ICacheService = Depends(get_cache_service)
    ):
        """Get process commit use case with all dependencies"""
        from ...application.use_cases.process_commit import ProcessCommitUseCase
        return ProcessCommitUseCase(repo, dispatcher, message_queue, cache)
    
    async def get_search_commits_use_case(
        repo: ICommitRepository = Depends(get_commit_repository),
        cache: ICacheService = Depends(get_cache_service)
    ):
        """Get search commits use case"""
        from ...application.use_cases.search_commits import SearchCommitsUseCase
        return SearchCommitsUseCase(repo, cache)
    
    async def get_chat_use_case(
        repo: ICommitRepository = Depends(get_commit_repository),
        ai_service: IAIAnalyzer = Depends(get_ai_service),
        cache: ICacheService = Depends(get_cache_service)
    ):
        """Get chat use case"""
        from ...application.use_cases.chat_with_ai import ChatWithAIUseCase
        return ChatWithAIUseCase(repo, ai_service, cache)
    
    def verify_webhook_signature(x_hub_signature: str = None) -> bool:
        """Verify webhook signature (simplified for demo)"""
        webhook_secret = os.getenv("WEBHOOK_SECRET")
        if not webhook_secret:
            return True  # Skip verification in development
        
        # In production, implement proper HMAC verification
        # For demo purposes, just check if signature exists
        return x_hub_signature is not None
    
    ### src/presentation/api/v1/webhooks.py
    ```python
    from fastapi import APIRouter, Depends, BackgroundTasks, Header, HTTPException
    from typing import List
    import logging
    
    from ....application.use_cases.process_commit import ProcessCommitUseCase
    from ....application.dto.webhook_dto import WebhookPayloadDto
    from ....application.dto.commit_dto import CommitDto, FileChangeDto
    from ..dependencies import get_process_commit_use_case, verify_webhook_signature
    
    logger = logging.getLogger(__name__)
    router = APIRouter(prefix="/webhooks", tags=["webhooks"])
    
    @router.post("/github")
    async def handle_github_webhook(
        payload: WebhookPayloadDto,
        background_tasks: BackgroundTasks,
        use_case: ProcessCommitUseCase = Depends(get_process_commit_use_case),
        signature_verified: bool = Depends(verify_webhook_signature)
    ):
        """Handle GitHub webhook"""
        
        if not signature_verified:
            raise HTTPException(
                status_code=401, 
                detail="Invalid webhook signature"
            )
        
        logger.info(f"Received webhook for {len(payload.commits)} commits in {payload.repository.name}")
        
        processed_commits = []
        
        # Process each commit
        for commit_data in payload.commits:
            try:
                # Convert webhook data to commit DTO
                commit_dto = CommitDto(
                    commit_hash=commit_data.id,
                    author_email=commit_data.author.email,
                    author_name=commit_data.author.name,
                    message=commit_data.message,
                    timestamp=commit_data.timestamp,
                    branch=payload.ref.split('/')[-1],  # Extract branch from refs/heads/main
                    project=payload.repository.name,
                    files_changed=_convert_file_changes(commit_data),
                    issue_numbers=_extract_issue_numbers(commit_data.message)
                )
                
                # Process commit (this will queue analysis automatically)
                result = await use_case.execute(commit_dto)
                processed_commits.append(result.commit_hash)
                
                logger.info(f"Processed commit: {commit_data.id[:8]}")
                
            except Exception as e:
                logger.error(f"Error processing commit {commit_data.id}: {str(e)}")
                # Continue processing other commits
        
        return {
            "status": "success",
            "message": f"Processed {len(processed_commits)} commits",
            "commits": processed_commits,
            "repository": payload.repository.name
        }
    
    @router.post("/gitlab")
    async def handle_gitlab_webhook(
        payload: dict,
        background_tasks: BackgroundTasks,
        use_case: ProcessCommitUseCase = Depends(get_process_commit_use_case),
        signature_verified: bool = Depends(verify_webhook_signature)
    ):
        """Handle GitLab webhook"""
        
        if not signature_verified:
            raise HTTPException(
                status_code=401,
                detail="Invalid webhook signature"
            )
        
        # Convert GitLab payload to our standard format
        # This is a simplified implementation
        logger.info("Received GitLab webhook (simplified handling)")
        
        return {
            "status": "success",
            "message": "GitLab webhook received (basic handling)",
            "commits": []
        }
    
    def _convert_file_changes(commit_data) -> List[FileChangeDto]:
        """Convert webhook file changes to our DTO format"""
        file_changes = []
        
        # Added files
        for filename in commit_data.added:
            file_changes.append(FileChangeDto(
                filename=filename,
                additions=10,  # Placeholder - GitHub doesn't provide this in webhook
                deletions=0,
                status="added"
            ))
        
        # Modified files
        for filename in commit_data.modified:
            file_changes.append(FileChangeDto(
                filename=filename,
                additions=5,   # Placeholder
                deletions=2,   # Placeholder
                status="modified"
            ))
        
        # Removed files
        for filename in commit_data.removed:
            file_changes.append(FileChangeDto(
                filename=filename,
                additions=0,
                deletions=20,  # Placeholder
                status="deleted"
            ))
        
        return file_changes
    
    def _extract_issue_numbers(message: str) -> List[str]:
        """Extract issue numbers from commit message"""
        import re
        
        # Look for patterns like #123, fixes #456, closes #789
        patterns = [
            r'#(\d+)',
            r'(?:fix|fixes|close|closes|resolve|resolves)\s+#(\d+)',
            r'(?:fix|fixes|close|closes|resolve|resolves)\s+(\d+)'
        ]
        
        issue_numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            issue_numbers.extend(matches)
        
        return list(set(issue_numbers))  # Remove duplicates
    
    ### src/presentation/api/v1/commits.py
    ```python
    from fastapi import APIRouter, Depends, Query, HTTPException
    from typing import List, Optional
    from datetime import datetime
    
    from ....application.use_cases.search_commits import SearchCommitsUseCase
    from ....application.dto.commit_dto import CommitSearchDto, CommitResponseDto
    from ..dependencies import get_search_commits_use_case, get_commit_repository
    from ....domain.repositories.commit_repository import ICommitRepository
    
    router = APIRouter(prefix="/commits", tags=["commits"])
    
    @router.get("/", response_model=List[CommitResponseDto])
    async def search_commits(
        q: Optional[str] = Query(None, description="Search query"),
        project: Optional[str] = Query(None, description="Filter by project"),
        author: Optional[str] = Query(None, description="Filter by author email"),
        branch: Optional[str] = Query(None, description="Filter by branch"),
        start_date: Optional[datetime] = Query(None, description="Start date filter"),
        end_date: Optional[datetime] = Query(None, description="End date filter"),
        tags: List[str] = Query([], description="Filter by tags"),
        limit: int = Query(50, ge=1, le=200, description="Result limit"),
        offset: int = Query(0, ge=0, description="Result offset"),
        use_case: SearchCommitsUseCase = Depends(get_search_commits_use_case)
    ):
        """Search commits with various filters"""
        
        search_dto = CommitSearchDto(
            query=q,
            project=project,
            author=author,
            branch=branch,
            start_date=start_date,
            end_date=end_date,
            tags=tags,
            limit=limit,
            offset=offset
        )
        
        results = await use_case.execute(search_dto)
        return results
    
    @router.get("/{commit_id}", response_model=CommitResponseDto)
    async def get_commit(
        commit_id: str,
        repo: ICommitRepository = Depends(get_commit_repository)
    ):
        """Get specific commit by ID"""
        
        commit = await repo.find_by_id(commit_id)
        if not commit:
            raise HTTPException(status_code=404, detail="Commit not found")
        
        # Convert to response DTO
        from ....application.use_cases.search_commits import SearchCommitsUseCase
        search_use_case = SearchCommitsUseCase(repo, None)
        return search_use_case._to_response_dto(commit)
    
    @router.get("/stats/overview")
    async def get_commit_statistics(
        project: Optional[str] = Query(None),
        author: Optional[str] = Query(None),
        start_date: Optional[datetime] = Query(None),
        end_date: Optional[datetime] = Query(None),
        repo: ICommitRepository = Depends(get_commit_repository)
    ):
        """Get commit statistics"""
        
        stats = await repo.get_statistics(
            project=project,
            author=author,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "status": "success",
            "data": stats
        }
    
    ### src/presentation/api/v1/chat.py
    ```python
    from fastapi import APIRouter, Depends, HTTPException
    from ....application.use_cases.chat_with_ai import ChatWithAIUseCase
    from ....application.dto.commit_dto import ChatRequestDto, ChatResponseDto
    from ..dependencies import get_chat_use_case
    
    router = APIRouter(prefix="/chat", tags=["chat"])
    
    @router.post("/", response_model=ChatResponseDto)
    async def chat_with_ai(
        request: ChatRequestDto,
        use_case: ChatWithAIUseCase = Depends(get_chat_use_case)
    ):
        """Chat with AI about commits"""
        
        try:
            response = await use_case.execute(request)
            return response
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing chat request: {str(e)}"
            )
    
    @router.get("/health")
    async def chat_health_check(
        use_case: ChatWithAIUseCase = Depends(get_chat_use_case)
    ):
        """Check AI service health"""
        
        try:
            health = await use_case.ai_analyzer.health_check()
            model_info = use_case.ai_analyzer.get_model_info()
            
            return {
                "status": "healthy" if health else "unhealthy",
                "model": model_info
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    ### src/presentation/websocket/handlers.py
    ```python
    from fastapi import WebSocket, WebSocketDisconnect
    from typing import Dict, Set
    import json
    import logging
    import asyncio
    
    logger = logging.getLogger(__name__)
    
    class WebSocketManager:
        """WebSocket connection manager"""
        
        def __init__(self):
            self.active_connections: Set[WebSocket] = set()
            self.project_subscribers: Dict[str, Set[WebSocket]] = {}
        
        async def connect(self, websocket: WebSocket):
            """Accept websocket connection"""
            await websocket.accept()
            self.active_connections.add(websocket)
            logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
        def disconnect(self, websocket: WebSocket):
            """Remove websocket connection"""
            self.active_connections.discard(websocket)
            
            # Remove from project subscriptions
            for project, subscribers in self.project_subscribers.items():
                subscribers.discard(websocket)
            
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
        
        async def subscribe_to_project(self, websocket: WebSocket, project: str):
            """Subscribe websocket to project updates"""
            if project not in self.project_subscribers:
                self.project_subscribers[project] = set()
            
            self.project_subscribers[project].add(websocket)
            
            await self.send_personal_message(websocket, {
                "type": "subscription_confirmed",
                "project": project
            })
            
            logger.info(f"WebSocket subscribed to project: {project}")
        
        async def send_personal_message(self, websocket: WebSocket, message: dict):
            """Send message to specific websocket"""
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending personal message: {str(e)}")
                self.disconnect(websocket)
        
        async def broadcast_to_project(self, project: str, message: dict):
            """Broadcast message to all subscribers of a project"""
            if project not in self.project_subscribers:
                return
            
            disconnected = set()
            
            for websocket in self.project_subscribers[project]:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error broadcasting to project {project}: {str(e)}")
                    disconnected.add(websocket)
            
            # Clean up disconnected websockets
            for websocket in disconnected:
                self.disconnect(websocket)
        
        async def broadcast_to_all(self, message: dict):
            """Broadcast message to all connected websockets"""
            disconnected = set()
            
            for websocket in self.active_connections.copy():
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error broadcasting to all: {str(e)}")
                    disconnected.add(websocket)
            
            # Clean up disconnected websockets
            for websocket in disconnected:
                self.disconnect(websocket)
    
    # Global WebSocket manager
    websocket_manager = WebSocketManager()
    
    async def websocket_endpoint(websocket: WebSocket):
        """Main WebSocket endpoint"""
        await websocket_manager.connect(websocket)
        
        try:
            while True:
                # Wait for message from client
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    await handle_websocket_message(websocket, message)
                except json.JSONDecodeError:
                    await websocket_manager.send_personal_message(websocket, {
                        "type": "error",
                        "message": "Invalid JSON format"
                    })
                
        except WebSocketDisconnect:
            websocket_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            websocket_manager.disconnect(websocket)
    
    async def handle_websocket_message(websocket: WebSocket, message: dict):
        """Handle incoming WebSocket message"""
        message_type = message.get("type")
        
        if message_type == "subscribe_project":
            project = message.get("project")
            if project:
                await websocket_manager.subscribe_to_project(websocket, project)
            else:
                await websocket_manager.send_personal_message(websocket, {
                    "type": "error",
                    "message": "Project name required for subscription"
                })
        
        elif message_type == "ping":
            await websocket_manager.send_personal_message(websocket, {
                "type": "pong",
                "timestamp": message.get("timestamp")
            })
        
        else:
            await websocket_manager.send_personal_message(websocket, {
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            })
    
    # Event handlers for broadcasting
    async def broadcast_commit_received(event_data: dict):
        """Broadcast commit received event"""
        project = event_data.get("project")
        if project:
            await websocket_manager.broadcast_to_project(project, {
                "type": "commit_received",
                "data": event_data
            })
    
    async def broadcast_commit_analyzed(event_data: dict):
        """Broadcast commit analysis completed event"""
        # In a real implementation, you'd get the project from the commit
        await websocket_manager.broadcast_to_all({
            "type": "commit_analyzed",
            "data": event_data
        })
    
    ## 5. Core Configuration
    
    ### src/core/config.py
    ```python
    from pydantic_settings import BaseSettings
    from typing import Optional
    import os
    
    class Settings(BaseSettings):
        """Application settings"""
        
        # Application
        app_name: str = "Git AI Analytics"
        app_version: str = "1.0.0"
        environment: str = "development"
        debug: bool = True
        
        # Server
        host: str = "0.0.0.0"
        port: int = 8000
        workers: int = 1
        reload: bool = True
        
        # Database
        db_provider: str = "sqlite"
        db_path: str = "data/git_analytics.db"
        db_host: Optional[str] = None
        db_port: Optional[int] = None
        db_name: Optional[str] = None
        db_user: Optional[str] = None
        db_password: Optional[str] = None
        db_echo: bool = False
        
        # Redis
        redis_url: str = "redis://localhost:6379"
        redis_password: Optional[str] = None
        
        # Celery
        celery_broker_url: str = "redis://localhost:6379/0"
        celery_result_backend: str = "redis://localhost:6379/0"
        
        # AI/Ollama
        ollama_base_url: str = "http://localhost:11434"
        ollama_model: str = "llama3.2:3b"
        ollama_timeout: int = 30
        
        # Security
        webhook_secret: Optional[str] = None
        api_key: Optional[str] = None
        
        # Cache
        cache_ttl_default: int = 3600
        cache_ttl_search: int = 300
        cache_ttl_chat: int = 600
        memory_cache_size: int = 1000
        
        # Logging
        log_level: str = "INFO"
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        class Config:
            env_file = ".env"
            case_sensitive = False
    
    # Global settings instance
    settings = Settings()
    
    ### src/core/logging.py
    ```python
    import logging
    import logging.config
    from .config import settings
    
    def setup_logging():
        """Setup logging configuration"""
        
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": settings.log_format,
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": settings.log_level,
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.FileHandler",
                    "level": "INFO",
                    "formatter": "detailed",
                    "filename": "logs/app.log",
                    "mode": "a",
                },
            },
            "loggers": {
                "": {
                    "level": settings.log_level,
                    "handlers": ["console"],
                    "propagate": False,
                },
                "src": {
                    "level": settings.log_level,
                    "handlers": ["console", "file"] if settings.environment == "production" else ["console"],
                    "propagate": False,
                },
                "uvicorn": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "celery": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
        }
        
        # Create logs directory
        import os
        os.makedirs("logs", exist_ok=True)
        
        logging.config.dictConfig(logging_config)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured for environment: {settings.environment}")
    
    ### src/main.py
    ```python
    import asyncio
    import logging
    from contextlib import asynccontextmanager
    
    from fastapi import FastAPI, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    
    from src.core.config import settings
    from src.core.logging import setup_logging
    from src.infrastructure.persistence.database import init_database, close_database
    from src.presentation.api.v1 import webhooks, commits, chat
    from src.presentation.websocket.handlers import websocket_endpoint
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager"""
        # Startup
        logger.info(f"Starting {settings.app_name} v{settings.app_version}")
        logger.info(f"Environment: {settings.environment}")
        
        # Initialize database
        await init_database()
        
        yield
        
        # Shutdown
        logger.info("Shutting down application")
        await close_database()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Git commit analytics with AI-powered insights",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routers
    app.include_router(webhooks.router, prefix="/api/v1")
    app.include_router(commits.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    
    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_route(websocket: WebSocket):
        await websocket_endpoint(websocket)
    
    # Health check
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "app": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment
        }
    
    # Root endpoint with simple UI
    @app.get("/", response_class=HTMLResponse)
    async def root():
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Git AI Analytics</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .method { font-weight: bold; color: #2196F3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 Git AI Analytics API</h1>
                <p>AI-powered Git commit analysis and search system</p>
                
                <h2>Available Endpoints:</h2>
                
                ### src/infrastructure/messaging/tasks/analysis_tasks.py
    ```python
    from celery import shared_task
    from celery.utils.log import get_task_logger
    from typing import Dict, Any
    import traceback
    import asyncio
    from ...persistence.database import get_session
    from ...persistence.repositories.sqlite_commit_repository import SqliteCommitRepository
    from ...ai.ollama_service import OllamaService
    from ....domain.events.commit_events import (
        CommitAnalysisStartedEvent,
        CommitAnalysisCompletedEvent,
        CommitAnalysisFailedEvent
    )
    
    logger = get_task_logger(__name__)
    
    @shared_task(bind=True, name='analyze_commit')
    def analyze_commit_task(self, commit_id: str) -> Dict[str, Any]:
        """Analyze commit with AI"""
        try:
            logger.info(f"Starting analysis for commit: {commit_id}")
            
            # Run async analysis
            result = asyncio.run(_analyze_commit_async(commit_id))
            
            logger.info(f"Successfully analyzed commit: {commit_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze commit {commit_id}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fire failure event
            asyncio.run(_fire_analysis_failed_event(commit_id, str(e), self.request.retries))
            
            # Re-raise for Celery retry mechanism
            raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))
    
    async def _analyze_commit_async(commit_id: str) -> Dict[str, Any]:
        """Async function to analyze commit"""
        
        # Get dependencies
        async with get_session() as session:
            commit_repo = SqliteCommitRepository(session)
            ai_service = OllamaService()
            
            # Load commit
            commit = await commit_repo.find_by_id(commit_id)
            if not commit:
                raise ValueError(f"Commit {commit_id} not found")
            
            # Fire analysis started event
            started_event = CommitAnalysisStartedEvent(
                commit_id=commit_id,
                model_name=ai_service.get_model_info()['name']
            )
            # Note: In a full implementation, we'd dispatch this event
            
            # Generate AI analysis
            analysis_result = await ai_service.analyze_commit(commit)
            
            # Update commit with analysis
            commit.mark_as_analyzed(
                summary=analysis_result.summary,
                tags=analysis_result.tags,
                sentiment=analysis_result.sentiment_score
            )
            
            updated_commit = await commit_repo.update(commit)
            
            # Fire analysis completed event
            completed_event = CommitAnalysisCompletedEvent(
                commit_id=commit_id,
                analysis_id=f"analysis_{commit_id}",
                summary=analysis_result.summary,
                tags=analysis_result.tags,
                processing_time_ms=1000  # Placeholder
            )
            # Note: In a full implementation, we'd dispatch this event
            
            return {
                'status': 'success',
                'commit_id': commit_id,
                'summary': analysis_result.summary,
                'tags': analysis_result.tags,
                'sentiment_score': analysis_result.sentiment_score,
                'confidence_score': analysis_result.confidence_score
            }
    
    async def _fire_analysis_failed_event(commit_id: str, error_message: str, retry_count: int):
        """Fire analysis failed event"""
        failed_event = CommitAnalysisFailedEvent(
            commit_id=commit_id,
            error_message=error_message,
            retry_count=retry_count
        )
        # Note: In a full implementation, we'd dispatch this event
    
    @shared_task(name='generate_summary')
    def generate_summary_task(project: str, days: int = 7) -> Dict[str, Any]:
        """Generate project summary for recent commits"""
        try:
            logger.info(f"Generating summary for project: {project}")
            
            result = asyncio.run(_generate_summary_async(project, days))
            
            logger.info(f"Successfully generated summary for project: {project}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate summary for project {project}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def _generate_summary_async(project: str, days: int) -> Dict[str, Any]:
        """Async function to generate summary"""
        from datetime import datetime, timedelta
        
        async with get_session() as session:
            commit_repo = SqliteCommitRepository(session)
            ai_service = OllamaService()
            
            # Get recent commits
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            commits = await commit_repo.find_by_project(
                project=project,
                start_date=start_date,
                end_date=end_date,
                limit=50
            )
            
            if not commits:
                return {
                    'status': 'success',
                    'project': project,
                    'summary': f"No commits found for {project} in the last {days} days.",
                    'commits_count': 0
                }
            
            # Generate summary
            summary = await ai_service.generate_summary(commits)
            
            return {
                'status': 'success',
                'project': project,
                'summary': summary,
                'commits_count': len(commits),
                'period_days': days
            }
    
    ### src/infrastructure/events/event_bus.py
    ```python
    import asyncio
    import json
    import logging
    from typing import Callable, Dict, List
    from redis.asyncio import Redis
    from ...domain.events.base import DomainEvent
    from ...application.services.event_dispatcher import IEventDispatcher
    
    logger = logging.getLogger(__name__)
    
    class RedisEventBus(IEventDispatcher):
        """Redis-based event bus for distributed event handling"""
        
        def __init__(self, redis_client: Redis, channel_prefix: str = "events"):
            self.redis = redis_client
            self.channel_prefix = channel_prefix
            self.handlers: Dict[str, List[Callable]] = {}
            self.subscriber_task = None
            self._running = False
        
        async def start(self):
            """Start event bus subscriber"""
            if not self._running:
                self._running = True
                self.subscriber_task = asyncio.create_task(self._subscriber_loop())
                logger.info("### src/infrastructure/persistence/repositories/sqlite_commit_repository.py
    ```python
    from typing import List, Optional, Dict, Any
    from datetime import datetime
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import select, and_, or_, desc, func
    from sqlalchemy.orm import selectinload
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.entities.commit import Commit, CommitHash, FileChange
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
                    id=commit.id,
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
                logger.error(f"Error saving commit {commit.commit_hash.value}: {str(e)}")
                raise
        
        async def find_by_id(self, commit_id: str) -> Optional[Commit]:
            """Find commit by ID"""
            try:
                stmt = select(CommitModel).where(CommitModel.id == commit_id)
                result = await self.session.execute(stmt)
                db_commit = result.scalar_one_or_none()
                
                return self._to_domain(db_commit) if db_commit else None
            except Exception as e:
                logger.error(f"Error finding commit by ID {commit_id}: {str(e)}")
                return None
        
        async def find_by_hash(self, commit_hash: str) -> Optional[Commit]:
            """Find commit by hash"""
            try:
                stmt = select(CommitModel).where(CommitModel.commit_hash == commit_hash)
                result = await self.session.execute(stmt)
                db_commit = result.scalar_one_or_none()
                
                return self._to_domain(db_commit) if db_commit else None
            except Exception as e:
                logger.error(f"Error finding commit by hash {commit_hash}: {str(e)}")
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
                logger.error(f"Error finding commits by author {author_email}: {str(e)}")
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
                logger.error(f"Error finding commits by project {project}: {str(e)}")
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
                logger.error(f"Error searching commits with query '{query}': {str(e)}")
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
                logger.error(f"Error updating commit {commit.id}: {str(e)}")
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
                logger.error(f"Error getting statistics: {str(e)}")
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
    
    ### src/infrastructure/ai/ollama_service.py
    ```python
    import httpx
    import json
    import logging
    from typing import Dict, Any, List, Optional
    from ...domain.services.ai_analyzer import IAIAnalyzer
    from ...domain.entities.commit import Commit
    from ...domain.entities.analysis import AnalysisResult
    from .prompts.commit_prompts import CommitPrompts
    import time
    import os
    
    logger = logging.getLogger(__name__)
    
    class OllamaService(IAIAnalyzer):
        """Ollama AI service implementation"""
        
        def __init__(
            self,
            base_url: str = None,
            model: str = None,
            timeout: int = 30
        ):
            self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
            self.timeout = timeout
            self.prompts = CommitPrompts()
            
            # Remove trailing slash
            self.base_url = self.base_url.rstrip('/')
        
        async def analyze_commit(self, commit: Commit) -> AnalysisResult:
            """Analyze a commit using Ollama"""
            prompt = self.prompts.get_commit_analysis_prompt(commit)
            
            try:
                start_time = time.time()
                response = await self._generate_response(prompt, temperature=0.3)
                processing_time = int((time.time() - start_time) * 1000)
                
                # Parse structured response
                analysis_data = self._parse_analysis_response(response)
                
                return AnalysisResult(
                    summary=analysis_data.get('summary', 'No summary available'),
                    tags=analysis_data.get('tags', []),
                    sentiment_score=analysis_data.get('sentiment', 0.0),
                    confidence_score=analysis_data.get('confidence', 0.5),
                    extracted_entities=analysis_data.get('entities', [])
                )
                
            except Exception as e:
                logger.error(f"Error analyzing commit {commit.commit_hash.value}: {str(e)}")
                # Return default analysis on error
                return AnalysisResult(
                    summary=f"Analysis failed: {str(e)}",
                    tags=['error'],
                    sentiment_score=0.0,
                    confidence_score=0.0,
                    extracted_entities=[]
                )
        
        async def generate_summary(self, commits: List[Commit]) -> str:
            """Generate summary for multiple commits"""
            if not commits:
                return "No commits to summarize."
            
            # Git AI Analytics - Full Implementation
    
    ## Project Structure
    ```
    git-ai-analytics/
    ├── src/
    │   ├── domain/                    # Domain Layer
    │   │   ├── entities/
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py
    │   │   │   ├── commit.py
    │   │   │   ├── developer.py
    │   │   │   └── analysis.py
    │   │   ├── events/
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py
    │   │   │   └── commit_events.py
    │   │   ├── repositories/
    │   │   │   ├── __init__.py
    │   │   │   ├── commit_repository.py
    │   │   │   └── analysis_repository.py
    │   │   └── services/
    │   │       ├── __init__.py
    │   │       └── ai_analyzer.py
    │   ├── application/               # Application Layer
    │   │   ├── use_cases/
    │   │   │   ├── __init__.py
    │   │   │   ├── process_commit.py
    │   │   │   ├── analyze_commit.py
    │   │   │   └── search_commits.py
    │   │   ├── dto/
    │   │   │   ├── __init__.py
    │   │   │   ├── commit_dto.py
    │   │   │   └── webhook_dto.py
    │   │   ├── services/
    │   │   │   ├── __init__.py
    │   │   │   └── event_dispatcher.py
    │   │   └── interfaces/
    │   │       ├── __init__.py
    │   │       ├── cache_interface.py
    │   │       ├── ai_interface.py
    │   │       └── messaging_interface.py
    │   ├── infrastructure/           # Infrastructure Layer
    │   │   ├── persistence/
    │   │   │   ├── __init__.py
    │   │   │   ├── models/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── commit_model.py
    │   │   │   ├── repositories/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── sqlite_commit_repository.py
    │   │   │   └── database.py
    │   │   ├── ai/
    │   │   │   ├── __init__.py
    │   │   │   ├── ollama_service.py
    │   │   │   └── prompts/
    │   │   │       └── commit_prompts.py
    │   │   ├── cache/
    │   │   │   ├── __init__.py
    │   │   │   ├── memory_cache.py
    │   │   │   └── redis_cache.py
    │   │   ├── messaging/
    │   │   │   ├── __init__.py
    │   │   │   ├── celery_config.py
    │   │   │   └── tasks/
    │   │   │       ├── __init__.py
    │   │   │       └── analysis_tasks.py
    │   │   └── events/
    │   │       ├── __init__.py
    │   │       └── event_bus.py
    │   ├── presentation/             # Presentation Layer
    │   │   ├── api/
    │   │   │   ├── __init__.py
    │   │   │   ├── v1/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── webhooks.py
    │   │   │   │   ├── commits.py
    │   │   │   │   └── chat.py
    │   │   │   └── dependencies.py
    │   │   └── websocket/
    │   │       ├── __init__.py
    │   │       └── handlers.py
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── config.py
    │   │   └── logging.py
    │   └── main.py
    ├── configs/
    │   ├── default.yaml
    │   ├── development.yaml
    │   └── production.yaml
    ├── tests/
    ├── migrations/
    ├── docker/
    │   ├── Dockerfile
    │   └── docker-compose.yml
    ├── requirements/
    │   ├── base.txt
    │   ├── development.txt
    │   └── production.txt
    ├── .env.example
    └── README.md
    ```
    
    ## 1. Domain Layer
    
    ### src/domain/entities/base.py
    ```python
    from abc import ABC
    from datetime import datetime
    from typing import Any
    from uuid import uuid4, UUID
    from dataclasses import dataclass, field
    
    @dataclass
    class ValueObject(ABC):
        """Base class for value objects"""
        
        def __post_init__(self):
            self._validate()
        
        def _validate(self):
            """Override to add validation logic"""
            pass
    
    class Entity(ABC):
        """Base class for entities"""
        
        def __init__(self):
            self.id: str = str(uuid4())
            self.created_at: datetime = datetime.utcnow()
            self.updated_at: datetime = datetime.utcnow()
        
        def __eq__(self, other):
            if not isinstance(other, self.__class__):
                return False
            return self.id == other.id
        
        def __hash__(self):
            return hash(self.id)
    ```
    
    ### src/domain/entities/commit.py
    ```python
    from dataclasses import dataclass
    from datetime import datetime
    from typing import List, Optional
    from .base import Entity, ValueObject
    
    @dataclass
    class CommitHash(ValueObject):
        """Value object for commit hash"""
        value: str
        
        def _validate(self):
            if not self.value or len(self.value) < 6:
                raise ValueError("Invalid commit hash")
    
    @dataclass
    class FileChange(ValueObject):
        """Value object for file changes"""
        filename: str
        additions: int
        deletions: int
        status: str  # added, modified, deleted, renamed
        
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
        """Commit domain entity"""
        
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
            self.commit_hash = commit_hash
            self.author_email = author_email
            self.author_name = author_name
            self.message = message
            self.timestamp = timestamp
            self.branch = branch
            self.project = project
            self.files_changed = files_changed or []
            self.issue_numbers = issue_numbers or []
            
            # Analysis results (populated later)
            self.summary: Optional[str] = None
            self.tags: List[str] = []
            self.sentiment_score: Optional[float] = None
            self.embedding_id: Optional[str] = None
            self.analyzed_at: Optional[datetime] = None
            
            # Computed properties
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
    ```
    
    ### src/domain/entities/analysis.py
    ```python
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, List, Optional
    from .base import Entity, ValueObject
    
    @dataclass
    class AnalysisResult(ValueObject):
        """Value object for AI analysis results"""
        summary: str
        tags: List[str]
        sentiment_score: float
        confidence_score: float
        extracted_entities: List[str]
        
        def _validate(self):
            if not (-1 <= self.sentiment_score <= 1):
                raise ValueError("Sentiment score must be between -1 and 1")
            if not (0 <= self.confidence_score <= 1):
                raise ValueError("Confidence score must be between 0 and 1")
    
    class Analysis(Entity):
        """Analysis domain entity"""
        
        def __init__(
            self,
            commit_id: str,
            model_name: str,
            model_version: str,
            prompt_version: str,
            result: AnalysisResult,
            processing_time_ms: int,
            tokens_used: int = 0
        ):
            super().__init__()
            self.commit_id = commit_id
            self.model_name = model_name
            self.model_version = model_version
            self.prompt_version = prompt_version
            self.result = result
            self.processing_time_ms = processing_time_ms
            self.tokens_used = tokens_used
            self.status = "completed"
        
        def is_high_confidence(self) -> bool:
            """Check if analysis has high confidence"""
            return self.result.confidence_score >= 0.8
    ```
    
    ### src/domain/events/base.py
    ```python
    from abc import ABC
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, Any
    import json
    from uuid import uuid4
    
    @dataclass
    class DomainEvent(ABC):
        """Base class for domain events"""
        event_id: str = None
        occurred_at: datetime = None
        
        def __post_init__(self):
            if self.event_id is None:
                self.event_id = str(uuid4())
            if self.occurred_at is None:
                self.occurred_at = datetime.utcnow()
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert event to dictionary"""
            return {
                'event_type': self.__class__.__name__,
                'event_id': self.event_id,
                'occurred_at': self.occurred_at.isoformat(),
                'data': self._get_event_data()
            }
        
        def to_json(self) -> str:
            """Convert event to JSON string"""
            return json.dumps(self.to_dict(), default=str)
        
        def _get_event_data(self) -> Dict[str, Any]:
            """Override to provide event-specific data"""
            return {}
    ```
    
    ### src/domain/events/commit_events.py
    ```python
    from dataclasses import dataclass
    from typing import Dict, Any, List
    from .base import DomainEvent
    
    @dataclass
    class CommitReceivedEvent(DomainEvent):
        """Event fired when a new commit is received"""
        commit_id: str
        commit_hash: str
        project: str
        author: str
        branch: str
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'commit_hash': self.commit_hash,
                'project': self.project,
                'author': self.author,
                'branch': self.branch
            }
    
    @dataclass 
    class CommitAnalysisStartedEvent(DomainEvent):
        """Event fired when commit analysis starts"""
        commit_id: str
        model_name: str
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'model_name': self.model_name
            }
    
    @dataclass
    class CommitAnalysisCompletedEvent(DomainEvent):
        """Event fired when commit analysis completes"""
        commit_id: str
        analysis_id: str
        summary: str
        tags: List[str]
        processing_time_ms: int
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'analysis_id': self.analysis_id,
                'summary': self.summary,
                'tags': self.tags,
                'processing_time_ms': self.processing_time_ms
            }
    
    @dataclass
    class CommitAnalysisFailedEvent(DomainEvent):
        """Event fired when commit analysis fails"""
        commit_id: str
        error_message: str
        retry_count: int
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'error_message': self.error_message,
                'retry_count': self.retry_count
            }
    ```
    
    ### src/domain/repositories/commit_repository.py
    ```python
    from abc import ABC, abstractmethod
    from typing import List, Optional, Dict, Any
    from datetime import datetime
    from ..entities.commit import Commit
    
    class ICommitRepository(ABC):
        """Repository interface for Commit entity"""
        
        @abstractmethod
        async def save(self, commit: Commit) -> Commit:
            """Save a commit"""
            pass
        
        @abstractmethod
        async def find_by_id(self, commit_id: str) -> Optional[Commit]:
            """Find commit by ID"""
            pass
        
        @abstractmethod
        async def find_by_hash(self, commit_hash: str) -> Optional[Commit]:
            """Find commit by hash"""
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
    ```
    
    ### src/domain/services/ai_analyzer.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, List
    from ..entities.commit import Commit
    from ..entities.analysis import AnalysisResult
    
    class IAIAnalyzer(ABC):
        """Domain service interface for AI analysis"""
        
        @abstractmethod
        async def analyze_commit(self, commit: Commit) -> AnalysisResult:
            """Analyze a commit using AI"""
            pass
        
        @abstractmethod
        async def generate_summary(self, commits: List[Commit]) -> str:
            """Generate summary for multiple commits"""
            pass
        
        @abstractmethod
        async def answer_question(
            self, 
            question: str, 
            context_commits: List[Commit]
        ) -> str:
            """Answer question about commits"""
            pass
        
        @abstractmethod
        async def extract_entities(self, text: str) -> List[str]:
            """Extract named entities from text"""
            pass
        
        @abstractmethod
        def get_model_info(self) -> Dict[str, str]:
            """Get current model information"""
            pass
    ```
    
    ## 2. Application Layer
    
    ### src/application/dto/commit_dto.py
    ```python
    from pydantic import BaseModel, Field, validator
    from datetime import datetime
    from typing import List, Optional
    
    class FileChangeDto(BaseModel):
        """DTO for file changes"""
        filename: str = Field(..., min_length=1)
        additions: int = Field(..., ge=0)
        deletions: int = Field(..., ge=0)
        status: str = Field(..., regex="^(added|modified|deleted|renamed)$")
    
    class CommitDto(BaseModel):
        """DTO for commit data"""
        commit_hash: str = Field(..., min_length=6)
        author_email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
        author_name: str = Field(..., min_length=1)
        message: str = Field(..., min_length=1)
        timestamp: datetime
        branch: str = Field(..., min_length=1)
        project: str = Field(..., min_length=1)
        files_changed: List[FileChangeDto] = []
        issue_numbers: List[str] = []
    
    class CommitResponseDto(BaseModel):
        """DTO for commit response"""
        id: str
        commit_hash: str
        author_email: str
        author_name: str
        message: str
        timestamp: datetime
        branch: str
        project: str
        files_count: int
        total_lines_changed: int
        complexity_score: Optional[float] = None
        summary: Optional[str] = None
        tags: List[str] = []
        sentiment_score: Optional[float] = None
        analyzed_at: Optional[datetime] = None
        created_at: datetime
    
    class CommitSearchDto(BaseModel):
        """DTO for commit search"""
        query: Optional[str] = None
        project: Optional[str] = None
        author: Optional[str] = None
        branch: Optional[str] = None
        start_date: Optional[datetime] = None
        end_date: Optional[datetime] = None
        tags: List[str] = []
        limit: int = Field(50, ge=1, le=200)
        offset: int = Field(0, ge=0)
    
    class ChatRequestDto(BaseModel):
        """DTO for chat requests"""
        question: str = Field(..., min_length=1, max_length=1000)
        context_project: Optional[str] = None
        context_author: Optional[str] = None
        context_days: int = Field(30, ge=1, le=365)
    
    class ChatResponseDto(BaseModel):
        """DTO for chat responses"""
        answer: str
        context_commits_count: int
        processing_time_ms: int
        model_used: str
    ```
    
    ### src/application/dto/webhook_dto.py
    ```python
    from pydantic import BaseModel
    from typing import List, Optional
    from datetime import datetime
    
    class WebhookAuthorDto(BaseModel):
        """DTO for webhook author data"""
        name: str
        email: str
        username: Optional[str] = None
    
    class WebhookCommitDto(BaseModel):
        """DTO for webhook commit data"""
        id: str
        message: str
        timestamp: datetime
        author: WebhookAuthorDto
        added: List[str] = []
        modified: List[str] = []
        removed: List[str] = []
        url: Optional[str] = None
    
    class WebhookRepositoryDto(BaseModel):
        """DTO for webhook repository data"""
        name: str
        full_name: str
        url: str
        default_branch: str
    
    class WebhookPayloadDto(BaseModel):
        """DTO for webhook payload"""
        ref: str  # refs/heads/main
        repository: WebhookRepositoryDto
        commits: List[WebhookCommitDto]
        pusher: WebhookAuthorDto
        head_commit: Optional[WebhookCommitDto] = None
    ```
    
    ### src/application/use_cases/process_commit.py
    ```python
    from typing import Optional
    from datetime import datetime
    from ..dto.commit_dto import CommitDto, CommitResponseDto
    from ..services.event_dispatcher import IEventDispatcher
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.entities.commit import Commit, CommitHash, FileChange
    from ...domain.events.commit_events import CommitReceivedEvent
    from ...application.interfaces.cache_interface import ICacheService
    from ...application.interfaces.messaging_interface import IMessageQueue
    
    class ProcessCommitUseCase:
        """Use case for processing incoming commits"""
        
        def __init__(
            self,
            commit_repository: ICommitRepository,
            event_dispatcher: IEventDispatcher,
            message_queue: IMessageQueue,
            cache_service: ICacheService
        ):
            self.commit_repository = commit_repository
            self.event_dispatcher = event_dispatcher
            self.message_queue = message_queue
            self.cache_service = cache_service
        
        async def execute(self, commit_dto: CommitDto) -> CommitResponseDto:
            """Process a new commit"""
            
            # Check if commit already exists (idempotency)
            existing = await self.commit_repository.find_by_hash(commit_dto.commit_hash)
            if existing:
                return self._to_response_dto(existing)
            
            # Create domain entity
            commit = self._to_domain(commit_dto)
            
            # Save to repository
            saved_commit = await self.commit_repository.save(commit)
            
            # Dispatch domain event
            event = CommitReceivedEvent(
                commit_id=saved_commit.id,
                commit_hash=saved_commit.commit_hash.value,
                project=saved_commit.project,
                author=saved_commit.author_email,
                branch=saved_commit.branch
            )
            await self.event_dispatcher.dispatch(event)
            
            # Queue for async analysis
            await self.message_queue.enqueue(
                "analyze_commit",
                {"commit_id": saved_commit.id}
            )
            
            # Invalidate relevant caches
            await self.cache_service.invalidate_pattern(f"commits:project:{commit.project}:*")
            await self.cache_service.invalidate_pattern(f"commits:author:{commit.author_email}:*")
            
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
    ```
    
    ### src/application/use_cases/search_commits.py
    ```python
    from typing import List
    from ..dto.commit_dto import CommitSearchDto, CommitResponseDto
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...application.interfaces.cache_interface import ICacheService
    import hashlib
    import json
    
    class SearchCommitsUseCase:
        """Use case for searching commits"""
        
        def __init__(
            self,
            commit_repository: ICommitRepository,
            cache_service: ICacheService
        ):
            self.commit_repository = commit_repository
            self.cache_service = cache_service
        
        async def execute(self, search_dto: CommitSearchDto) -> List[CommitResponseDto]:
            """Search commits with caching"""
            
            # Generate cache key
            cache_key = self._generate_cache_key(search_dto)
            
            # Try cache first
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                return cached_result
            
            # Search in repository
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
                limit=search_dto.limit
            )
            
            # Convert to DTOs
            result = [self._to_response_dto(commit) for commit in commits]
            
            # Cache result for 5 minutes
            await self.cache_service.set(cache_key, result, ttl=300)
            
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
    ```
    
    ### src/application/use_cases/chat_with_ai.py
    ```python
    from typing import List
    from datetime import datetime, timedelta
    import time
    from ..dto.commit_dto import ChatRequestDto, ChatResponseDto
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.services.ai_analyzer import IAIAnalyzer
    from ...application.interfaces.cache_interface import ICacheService
    import hashlib
    
    class ChatWithAIUseCase:
        """Use case for chatting with AI about commits"""
        
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
            """Answer question about commits using AI"""
            
            start_time = time.time()
            
            # Generate cache key
            cache_key = self._generate_cache_key(chat_dto)
            
            # Try cache first
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                return cached_result
            
            # Get context commits
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=chat_dto.context_days)
            
            context_commits = await self.commit_repository.find_by_project(
                project=chat_dto.context_project,
                start_date=start_date,
                end_date=end_date,
                limit=100
            )
            
            if chat_dto.context_author:
                author_commits = await self.commit_repository.find_by_author(
                    author_email=chat_dto.context_author,
                    project=chat_dto.context_project,
                    start_date=start_date,
                    end_date=end_date,
                    limit=50
                )
                context_commits.extend(author_commits)
            
            # Remove duplicates
            unique_commits = {commit.id: commit for commit in context_commits}
            context_commits = list(unique_commits.values())
            
            # Ask AI
            answer = await self.ai_analyzer.answer_question(
                question=chat_dto.question,
                context_commits=context_commits
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            result = ChatResponseDto(
                answer=answer,
                context_commits_count=len(context_commits),
                processing_time_ms=processing_time_ms,
                model_used=self.ai_analyzer.get_model_info()['name']
            )
            
            # Cache result for 10 minutes
            await self.cache_service.set(cache_key, result, ttl=600)
            
            return result
        
        def _generate_cache_key(self, chat_dto: ChatRequestDto) -> str:
            """Generate cache key for chat request"""
            chat_hash = hashlib.md5(
                f"{chat_dto.question}:{chat_dto.context_project}:{chat_dto.context_author}:{chat_dto.context_days}".encode()
            ).hexdigest()
    ### src/application/services/event_dispatcher.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, List, Callable, Any
    from ...domain.events.base import DomainEvent
    import asyncio
    import logging
    
    logger = logging.getLogger(__name__)
    
    class IEventDispatcher(ABC):
        """Event dispatcher interface"""
        
        @abstractmethod
        async def dispatch(self, event: DomainEvent):
            """Dispatch event to all registered handlers"""
            pass
        
        @abstractmethod
        def register_handler(self, event_type: str, handler: Callable):
            """Register event handler"""
            pass
    
    class EventDispatcher(IEventDispatcher):
        """Implementation of event dispatcher"""
        
        def __init__(self):
            self.handlers: Dict[str, List[Callable]] = {}
        
        def register_handler(self, event_type: str, handler: Callable):
            """Register event handler"""
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
            logger.info(f"Registered handler for event type: {event_type}")
        
        async def dispatch(self, event: DomainEvent):
            """Dispatch event to all registered handlers"""
            event_type = event.__class__.__name__
            
            if event_type not in self.handlers:
                logger.debug(f"No handlers registered for event type: {event_type}")
                return
            
            logger.info(f"Dispatching event: {event_type} (ID: {event.event_id})")
            
            tasks = []
            for handler in self.handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        # Run sync handler in thread pool
                        loop = asyncio.get_event_loop()
                        tasks.append(loop.run_in_executor(None, handler, event))
                except Exception as e:
                    logger.error(f"Error preparing handler for {event_type}: {str(e)}")
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log any handler exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Handler {i} failed for {event_type}: {str(result)}")
    
    ### src/application/interfaces/cache_interface.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Any, Optional, List
    
    class ICacheService(ABC):
        """Interface for cache service"""
        
        @abstractmethod
        async def get(self, key: str) -> Optional[Any]:
            """Get value from cache"""
            pass
        
        @abstractmethod
        async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
            """Set value in cache with TTL in seconds"""
            pass
        
        @abstractmethod
        async def delete(self, key: str) -> bool:
            """Delete key from cache"""
            pass
        
        @abstractmethod
        async def exists(self, key: str) -> bool:
            """Check if key exists in cache"""
            pass
        
        @abstractmethod
        async def invalidate_pattern(self, pattern: str) -> int:
            """Invalidate all keys matching pattern (with * wildcard)"""
            pass
        
        @abstractmethod
        async def get_many(self, keys: List[str]) -> Dict[str, Any]:
            """Get multiple values from cache"""
            pass
        
        @abstractmethod
        async def set_many(self, data: Dict[str, Any], ttl: int = 3600) -> bool:
            """Set multiple values in cache"""
            pass
    
    ### src/application/interfaces/ai_interface.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, List, Optional
    from ...domain.entities.analysis import AnalysisResult
    
    class IAIService(ABC):
        """Interface for AI service"""
        
        @abstractmethod
        async def analyze_text(
            self,
            text: str,
            context: Optional[str] = None,
            temperature: float = 0.7
        ) -> AnalysisResult:
            """Analyze text and return structured results"""
            pass
        
        @abstractmethod
        async def generate_response(
            self,
            prompt: str,
            context: Optional[str] = None,
            max_tokens: int = 500
        ) -> str:
            """Generate text response"""
            pass
        
        @abstractmethod
        async def create_embedding(self, text: str) -> List[float]:
            """Create text embedding"""
            pass
        
        @abstractmethod
        async def health_check(self) -> bool:
            """Check if AI service is healthy"""
            pass
        
        @abstractmethod
        def get_model_info(self) -> Dict[str, str]:
            """Get model information"""
            pass
    
    ### src/application/interfaces/messaging_interface.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, Optional, Callable
    from enum import Enum
    
    class TaskStatus(Enum):
        PENDING = "PENDING"
        STARTED = "STARTED"
        SUCCESS = "SUCCESS"
        FAILURE = "FAILURE"
        RETRY = "RETRY"
    
    class IMessageQueue(ABC):
        """Interface for message queue"""
        
        @abstractmethod
        async def enqueue(
            self,
            task_name: str,
            payload: Dict[str, Any],
            delay_seconds: int = 0,
            priority: int = 5
        ) -> str:
            """Enqueue task for processing"""
            pass
        
        @abstractmethod
        async def get_task_status(self, task_id: str) -> TaskStatus:
            """Get task status"""
            pass
        
        @abstractmethod
        async def get_task_result(self, task_id: str) -> Optional[Any]:
            """Get task result"""
            pass
        
        @abstractmethod
        async def cancel_task(self, task_id: str) -> bool:
            """Cancel pending task"""
            pass
        
        @abstractmethod
        def register_task(self, task_name: str, handler: Callable):
            """Register task handler"""
            pass
    
    ## 3. Infrastructure Layer
    
    ### src/infrastructure/persistence/database.py
    ```python
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy import MetaData
    from contextlib import asynccontextmanager
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    # SQLAlchemy base
    Base = declarative_base()
    metadata = MetaData()
    
    # Global engine and session maker
    engine = None
    SessionLocal = None
    
    def get_database_url() -> str:
        """Get database URL from environment"""
        db_provider = os.getenv("DB_PROVIDER", "sqlite")
        
        if db_provider == "sqlite":
            db_path = os.getenv("DB_PATH", "data/git_analytics.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            return f"sqlite+aiosqlite:///{db_path}"
        
        elif db_provider == "postgresql":
            host = os.getenv("DB_HOST", "localhost")
            port = os.getenv("DB_PORT", "5432")
            name = os.getenv("DB_NAME", "git_analytics")
            user = os.getenv("DB_USER", "postgres")
            password = os.getenv("DB_PASSWORD", "")
            return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"
        
        else:
            raise ValueError(f"Unsupported database provider: {db_provider}")
    
    async def init_database():
        """Initialize database connection"""
        global engine, SessionLocal
        
        if engine is None:
            database_url = get_database_url()
            logger.info(f"Connecting to database: {database_url.split('://')[0]}://...")
            
            engine = create_async_engine(
                database_url,
                echo=os.getenv("DB_ECHO", "false").lower() == "true",
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            SessionLocal = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database initialized successfully")
    
    async def close_database():
        """Close database connection"""
        global engine
        if engine:
            await engine.dispose()
            logger.info("Database connection closed")
    
    @asynccontextmanager
    async def get_session() -> AsyncSession:
        """Get database session"""
        if SessionLocal is None:
            await init_database()
        
        async with SessionLocal() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    ### src/infrastructure/persistence/models/commit_model.py
    ```python
    from sqlalchemy import Column, String, DateTime, Text, Integer, Float, JSON
    from sqlalchemy.sql import func
    from ..database import Base
    
    class CommitModel(Base):
        """SQLAlchemy model for commits"""
        
        __tablename__ = "commits"
        
        # Primary fields
        id = Column(String(36), primary_key=True)
        commit_hash = Column(String(40), unique=True, nullable=False, index=True)
        author_email = Column(String(255), nullable=False, index=True)
        author_name = Column(String(255), nullable=False)
        message = Column(Text, nullable=False)
        timestamp = Column(DateTime, nullable=False, index=True)
        branch = Column(String(255), nullable=False, index=True)
        project = Column(String(255), nullable=False, index=True)
        
        # File changes (stored as JSON)
        files_changed = Column(JSON, nullable=False, default=list)
        issue_numbers = Column(JSON, nullable=False, default=list)
        
        # Metrics
        total_lines_changed = Column(Integer, default=0)
        files_count = Column(Integer, default=0)
        complexity_score = Column(Float)
        impact_score = Column(Float)
        
        # Analysis results
        summary = Column(Text)
        tags = Column(JSON, default=list)
        sentiment_score = Column(Float)
        embedding_id = Column(String(255))
        analyzed_at = Column(DateTime)
        
        # Timestamps
        created_at = Column(DateTime, default=func.now(), nullable=False)
        updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
        
        def __repr__(self):
            return f"<CommitModel(hash={self.commit_hash}, project={self.project})>"
    
    class AnalysisModel(Base):
        """SQLAlchemy model for analysis results"""
        
        __tablename__ = "analyses"
        
        id = Column(String(36), primary_key=True)
        commit_id = Column(String(36), nullable=False, index=True)
        
        # Model information
        model_name = Column(String(100), nullable=False)
        model_version = Column(String(50), nullable=False)
        prompt_version = Column(String(50), nullable=False)
        
        # Analysis results
        summary = Column(Text, nullable=False)
        tags = Column(JSON, default=list)
        sentiment_score = Column(Float, nullable=False)
        confidence_score = Column(Float, nullable=False)
        extracted_entities = Column(JSON, default=list)
        
        # Performance metrics
        processing_time_ms = Column(Integer, nullable=False)
        tokens_used = Column(Integer, default=0)
        status = Column(String(20), default="completed")
        
        # Timestamps
        created_at = Column(DateTime, default=func.now(), nullable=False)
        
        def __repr__(self):
            return f"<AnalysisModel(commit_id={self.commit_id}, model={self.model_name})>"

Git AI Analytics - Complete Implementation

Code ∙ Version 7 

Redis event bus started") async def stop(self): """Stop event bus subscriber""" self.\_running = False if self.subscriber\_task: self.subscriber\_task.cancel() try: await self.subs

Retry

---

You asked:
----------

LL

Continue

Edit

---

Claude Replied:
---------------

    Redis event bus started")
        
        async def stop(self):
            """Stop event bus subscriber"""
            self._running = False
            if self.subscriber_task:
                self.subscriber_task.cancel()
                try:
                    await self.subscriber_task
                except asyncio.CancelledError:
                    pass
            logger.info("Redis event bus stopped")
        
        def register_handler(self, event_type: str, handler: Callable):
            """Register event handler"""
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
            logger.info(f"Registered Redis handler for event type: {event_type}")
        
        async def dispatch(self, event: DomainEvent):
            """Dispatch event to Redis"""
            try:
                event_type = event.__class__.__name__
                channel = f"{self.channel_prefix}:{event_type}"
                
                message = event.to_json()
                await self.redis.publish(channel, message)
                
                logger.debug(f"Published event {event_type} to Redis channel: {channel}")
                
            except Exception as e:
                logger.error(f"Error publishing event to Redis: {str(e)}")
        
        async def _subscriber_loop(self):
            """Main subscriber loop"""
            try:
                pubsub = self.redis.pubsub()
                
                # Subscribe to all event channels
                pattern = f"{self.channel_prefix}:*"
                await pubsub.psubscribe(pattern)
                
                logger.info(f"Subscribed to Redis pattern: {pattern}")
                
                async for message in pubsub.listen():
                    if not self._running:
                        break
                        
                    if message['type'] == 'pmessage':
                        await self._handle_message(message)
                        
            except asyncio.CancelledError:
                logger.info("Redis subscriber loop cancelled")
            except Exception as e:
                logger.error(f"Error in Redis subscriber loop: {str(e)}")
            finally:
                try:
                    await pubsub.unsubscribe()
                    await pubsub.close()
                except Exception:
                    pass
        
        async def _handle_message(self, message):
            """Handle incoming Redis message"""
            try:
                channel = message['channel'].decode('utf-8')
                event_type = channel.split(':')[-1]
                
                if event_type not in self.handlers:
                    return
                
                # Parse event data
                event_data = json.loads(message['data'].decode('utf-8'))
                
                # Execute handlers
                tasks = []
                for handler in self.handlers[event_type]:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event_data))
                    else:
                        loop = asyncio.get_event_loop()
                        tasks.append(loop.run_in_executor(None, handler, event_data))
                
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Log handler exceptions
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            logger.error(f"Redis handler {i} failed for {event_type}: {str(result)}")
                            
            except Exception as e:
                logger.error(f"Error handling Redis message: {str(e)}")
    
    ## 4. Presentation Layer
    
    ### src/presentation/api/dependencies.py
    ```python
    from typing import AsyncGenerator
    from fastapi import Depends, HTTPException, status
    from sqlalchemy.ext.asyncio import AsyncSession
    from redis.asyncio import Redis
    
    # Import domain interfaces
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.services.ai_analyzer import IAIAnalyzer
    
    # Import application interfaces  
    from ...application.interfaces.cache_interface import ICacheService
    from ...application.interfaces.messaging_interface import IMessageQueue
    from ...application.services.event_dispatcher import IEventDispatcher
    
    # Import infrastructure implementations
    from ...infrastructure.persistence.database import get_session
    from ...infrastructure.persistence.repositories.sqlite_commit_repository import SqliteCommitRepository
    from ...infrastructure.ai.ollama_service import OllamaService
    from ...infrastructure.cache.memory_cache import MemoryCacheService
    from ...infrastructure.cache.redis_cache import RedisCacheService
    from ...infrastructure.messaging.celery_config import celery_app, CeleryMessageQueue
    from ...infrastructure.events.event_bus import RedisEventBus
    
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Global singletons
    _redis_client = None
    _cache_service = None
    _event_dispatcher = None
    _message_queue = None
    _ai_service = None
    
    async def get_redis_client() -> Redis:
        """Get Redis client singleton"""
        global _redis_client
        if _redis_client is None:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            try:
                _redis_client = Redis.from_url(redis_url, decode_responses=False)
                # Test connection
                await _redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {str(e)}, falling back to memory cache")
                _redis_client = None
        return _redis_client
    
    async def get_cache_service() -> ICacheService:
        """Get cache service singleton"""
        global _cache_service
        if _cache_service is None:
            redis_client = await get_redis_client()
            if redis_client:
                _cache_service = RedisCacheService(redis_client)
            else:
                _cache_service = MemoryCacheService()
            logger.info(f"Cache service initialized: {type(_cache_service).__name__}")
        return _cache_service
    
    async def get_message_queue() -> IMessageQueue:
        """Get message queue singleton"""
        global _message_queue
        if _message_queue is None:
            _message_queue = CeleryMessageQueue(celery_app)
            logger.info("Message queue initialized")
        return _message_queue
    
    async def get_event_dispatcher() -> IEventDispatcher:
        """Get event dispatcher singleton"""
        global _event_dispatcher
        if _event_dispatcher is None:
            redis_client = await get_redis_client()
            if redis_client:
                _event_dispatcher = RedisEventBus(redis_client)
                await _event_dispatcher.start()
            else:
                from ...application.services.event_dispatcher import EventDispatcher
                _event_dispatcher = EventDispatcher()
            logger.info(f"Event dispatcher initialized: {type(_event_dispatcher).__name__}")
        return _event_dispatcher
    
    async def get_ai_service() -> IAIAnalyzer:
        """Get AI service singleton"""
        global _ai_service
        if _ai_service is None:
            _ai_service = OllamaService()
            # Test AI service health
            try:
                health = await _ai_service.health_check()
                if not health:
                    logger.warning("AI service health check failed")
            except Exception as e:
                logger.warning(f"AI service health check error: {str(e)}")
            logger.info("AI service initialized")
        return _ai_service
    
    async def get_commit_repository(
        session: AsyncSession = Depends(get_session)
    ) -> ICommitRepository:
        """Get commit repository with injected session"""
        return SqliteCommitRepository(session)
    
    # Use case dependencies
    async def get_process_commit_use_case(
        repo: ICommitRepository = Depends(get_commit_repository),
        dispatcher: IEventDispatcher = Depends(get_event_dispatcher),
        message_queue: IMessageQueue = Depends(get_message_queue),
        cache: ICacheService = Depends(get_cache_service)
    ):
        """Get process commit use case with all dependencies"""
        from ...application.use_cases.process_commit import ProcessCommitUseCase
        return ProcessCommitUseCase(repo, dispatcher, message_queue, cache)
    
    async def get_search_commits_use_case(
        repo: ICommitRepository = Depends(get_commit_repository),
        cache: ICacheService = Depends(get_cache_service)
    ):
        """Get search commits use case"""
        from ...application.use_cases.search_commits import SearchCommitsUseCase
        return SearchCommitsUseCase(repo, cache)
    
    async def get_chat_use_case(
        repo: ICommitRepository = Depends(get_commit_repository),
        ai_service: IAIAnalyzer = Depends(get_ai_service),
        cache: ICacheService = Depends(get_cache_service)
    ):
        """Get chat use case"""
        from ...application.use_cases.chat_with_ai import ChatWithAIUseCase
        return ChatWithAIUseCase(repo, ai_service, cache)
    
    def verify_webhook_signature(x_hub_signature: str = None) -> bool:
        """Verify webhook signature (simplified for demo)"""
        webhook_secret = os.getenv("WEBHOOK_SECRET")
        if not webhook_secret:
            return True  # Skip verification in development
        
        # In production, implement proper HMAC verification
        # For demo purposes, just check if signature exists
        return x_hub_signature is not None
    
    ### src/presentation/api/v1/webhooks.py
    ```python
    from fastapi import APIRouter, Depends, BackgroundTasks, Header, HTTPException
    from typing import List
    import logging
    
    from ....application.use_cases.process_commit import ProcessCommitUseCase
    from ....application.dto.webhook_dto import WebhookPayloadDto
    from ....application.dto.commit_dto import CommitDto, FileChangeDto
    from ..dependencies import get_process_commit_use_case, verify_webhook_signature
    
    logger = logging.getLogger(__name__)
    router = APIRouter(prefix="/webhooks", tags=["webhooks"])
    
    @router.post("/github")
    async def handle_github_webhook(
        payload: WebhookPayloadDto,
        background_tasks: BackgroundTasks,
        use_case: ProcessCommitUseCase = Depends(get_process_commit_use_case),
        signature_verified: bool = Depends(verify_webhook_signature)
    ):
        """Handle GitHub webhook"""
        
        if not signature_verified:
            raise HTTPException(
                status_code=401, 
                detail="Invalid webhook signature"
            )
        
        logger.info(f"Received webhook for {len(payload.commits)} commits in {payload.repository.name}")
        
        processed_commits = []
        
        # Process each commit
        for commit_data in payload.commits:
            try:
                # Convert webhook data to commit DTO
                commit_dto = CommitDto(
                    commit_hash=commit_data.id,
                    author_email=commit_data.author.email,
                    author_name=commit_data.author.name,
                    message=commit_data.message,
                    timestamp=commit_data.timestamp,
                    branch=payload.ref.split('/')[-1],  # Extract branch from refs/heads/main
                    project=payload.repository.name,
                    files_changed=_convert_file_changes(commit_data),
                    issue_numbers=_extract_issue_numbers(commit_data.message)
                )
                
                # Process commit (this will queue analysis automatically)
                result = await use_case.execute(commit_dto)
                processed_commits.append(result.commit_hash)
                
                logger.info(f"Processed commit: {commit_data.id[:8]}")
                
            except Exception as e:
                logger.error(f"Error processing commit {commit_data.id}: {str(e)}")
                # Continue processing other commits
        
        return {
            "status": "success",
            "message": f"Processed {len(processed_commits)} commits",
            "commits": processed_commits,
            "repository": payload.repository.name
        }
    
    @router.post("/gitlab")
    async def handle_gitlab_webhook(
        payload: dict,
        background_tasks: BackgroundTasks,
        use_case: ProcessCommitUseCase = Depends(get_process_commit_use_case),
        signature_verified: bool = Depends(verify_webhook_signature)
    ):
        """Handle GitLab webhook"""
        
        if not signature_verified:
            raise HTTPException(
                status_code=401,
                detail="Invalid webhook signature"
            )
        
        # Convert GitLab payload to our standard format
        # This is a simplified implementation
        logger.info("Received GitLab webhook (simplified handling)")
        
        return {
            "status": "success",
            "message": "GitLab webhook received (basic handling)",
            "commits": []
        }
    
    def _convert_file_changes(commit_data) -> List[FileChangeDto]:
        """Convert webhook file changes to our DTO format"""
        file_changes = []
        
        # Added files
        for filename in commit_data.added:
            file_changes.append(FileChangeDto(
                filename=filename,
                additions=10,  # Placeholder - GitHub doesn't provide this in webhook
                deletions=0,
                status="added"
            ))
        
        # Modified files
        for filename in commit_data.modified:
            file_changes.append(FileChangeDto(
                filename=filename,
                additions=5,   # Placeholder
                deletions=2,   # Placeholder
                status="modified"
            ))
        
        # Removed files
        for filename in commit_data.removed:
            file_changes.append(FileChangeDto(
                filename=filename,
                additions=0,
                deletions=20,  # Placeholder
                status="deleted"
            ))
        
        return file_changes
    
    def _extract_issue_numbers(message: str) -> List[str]:
        """Extract issue numbers from commit message"""
        import re
        
        # Look for patterns like #123, fixes #456, closes #789
        patterns = [
            r'#(\d+)',
            r'(?:fix|fixes|close|closes|resolve|resolves)\s+#(\d+)',
            r'(?:fix|fixes|close|closes|resolve|resolves)\s+(\d+)'
        ]
        
        issue_numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            issue_numbers.extend(matches)
        
        return list(set(issue_numbers))  # Remove duplicates
    
    ### src/presentation/api/v1/commits.py
    ```python
    from fastapi import APIRouter, Depends, Query, HTTPException
    from typing import List, Optional
    from datetime import datetime
    
    from ....application.use_cases.search_commits import SearchCommitsUseCase
    from ....application.dto.commit_dto import CommitSearchDto, CommitResponseDto
    from ..dependencies import get_search_commits_use_case, get_commit_repository
    from ....domain.repositories.commit_repository import ICommitRepository
    
    router = APIRouter(prefix="/commits", tags=["commits"])
    
    @router.get("/", response_model=List[CommitResponseDto])
    async def search_commits(
        q: Optional[str] = Query(None, description="Search query"),
        project: Optional[str] = Query(None, description="Filter by project"),
        author: Optional[str] = Query(None, description="Filter by author email"),
        branch: Optional[str] = Query(None, description="Filter by branch"),
        start_date: Optional[datetime] = Query(None, description="Start date filter"),
        end_date: Optional[datetime] = Query(None, description="End date filter"),
        tags: List[str] = Query([], description="Filter by tags"),
        limit: int = Query(50, ge=1, le=200, description="Result limit"),
        offset: int = Query(0, ge=0, description="Result offset"),
        use_case: SearchCommitsUseCase = Depends(get_search_commits_use_case)
    ):
        """Search commits with various filters"""
        
        search_dto = CommitSearchDto(
            query=q,
            project=project,
            author=author,
            branch=branch,
            start_date=start_date,
            end_date=end_date,
            tags=tags,
            limit=limit,
            offset=offset
        )
        
        results = await use_case.execute(search_dto)
        return results
    
    @router.get("/{commit_id}", response_model=CommitResponseDto)
    async def get_commit(
        commit_id: str,
        repo: ICommitRepository = Depends(get_commit_repository)
    ):
        """Get specific commit by ID"""
        
        commit = await repo.find_by_id(commit_id)
        if not commit:
            raise HTTPException(status_code=404, detail="Commit not found")
        
        # Convert to response DTO
        from ....application.use_cases.search_commits import SearchCommitsUseCase
        search_use_case = SearchCommitsUseCase(repo, None)
        return search_use_case._to_response_dto(commit)
    
    @router.get("/stats/overview")
    async def get_commit_statistics(
        project: Optional[str] = Query(None),
        author: Optional[str] = Query(None),
        start_date: Optional[datetime] = Query(None),
        end_date: Optional[datetime] = Query(None),
        repo: ICommitRepository = Depends(get_commit_repository)
    ):
        """Get commit statistics"""
        
        stats = await repo.get_statistics(
            project=project,
            author=author,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "status": "success",
            "data": stats
        }
    
    ### src/presentation/api/v1/chat.py
    ```python
    from fastapi import APIRouter, Depends, HTTPException
    from ....application.use_cases.chat_with_ai import ChatWithAIUseCase
    from ....application.dto.commit_dto import ChatRequestDto, ChatResponseDto
    from ..dependencies import get_chat_use_case
    
    router = APIRouter(prefix="/chat", tags=["chat"])
    
    @router.post("/", response_model=ChatResponseDto)
    async def chat_with_ai(
        request: ChatRequestDto,
        use_case: ChatWithAIUseCase = Depends(get_chat_use_case)
    ):
        """Chat with AI about commits"""
        
        try:
            response = await use_case.execute(request)
            return response
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing chat request: {str(e)}"
            )
    
    @router.get("/health")
    async def chat_health_check(
        use_case: ChatWithAIUseCase = Depends(get_chat_use_case)
    ):
        """Check AI service health"""
        
        try:
            health = await use_case.ai_analyzer.health_check()
            model_info = use_case.ai_analyzer.get_model_info()
            
            return {
                "status": "healthy" if health else "unhealthy",
                "model": model_info
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    ### src/presentation/websocket/handlers.py
    ```python
    from fastapi import WebSocket, WebSocketDisconnect
    from typing import Dict, Set
    import json
    import logging
    import asyncio
    
    logger = logging.getLogger(__name__)
    
    class WebSocketManager:
        """WebSocket connection manager"""
        
        def __init__(self):
            self.active_connections: Set[WebSocket] = set()
            self.project_subscribers: Dict[str, Set[WebSocket]] = {}
        
        async def connect(self, websocket: WebSocket):
            """Accept websocket connection"""
            await websocket.accept()
            self.active_connections.add(websocket)
            logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
        def disconnect(self, websocket: WebSocket):
            """Remove websocket connection"""
            self.active_connections.discard(websocket)
            
            # Remove from project subscriptions
            for project, subscribers in self.project_subscribers.items():
                subscribers.discard(websocket)
            
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
        
        async def subscribe_to_project(self, websocket: WebSocket, project: str):
            """Subscribe websocket to project updates"""
            if project not in self.project_subscribers:
                self.project_subscribers[project] = set()
            
            self.project_subscribers[project].add(websocket)
            
            await self.send_personal_message(websocket, {
                "type": "subscription_confirmed",
                "project": project
            })
            
            logger.info(f"WebSocket subscribed to project: {project}")
        
        async def send_personal_message(self, websocket: WebSocket, message: dict):
            """Send message to specific websocket"""
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending personal message: {str(e)}")
                self.disconnect(websocket)
        
        async def broadcast_to_project(self, project: str, message: dict):
            """Broadcast message to all subscribers of a project"""
            if project not in self.project_subscribers:
                return
            
            disconnected = set()
            
            for websocket in self.project_subscribers[project]:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error broadcasting to project {project}: {str(e)}")
                    disconnected.add(websocket)
            
            # Clean up disconnected websockets
            for websocket in disconnected:
                self.disconnect(websocket)
        
        async def broadcast_to_all(self, message: dict):
            """Broadcast message to all connected websockets"""
            disconnected = set()
            
            for websocket in self.active_connections.copy():
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error broadcasting to all: {str(e)}")
                    disconnected.add(websocket)
            
            # Clean up disconnected websockets
            for websocket in disconnected:
                self.disconnect(websocket)
    
    # Global WebSocket manager
    websocket_manager = WebSocketManager()
    
    async def websocket_endpoint(websocket: WebSocket):
        """Main WebSocket endpoint"""
        await websocket_manager.connect(websocket)
        
        try:
            while True:
                # Wait for message from client
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    await handle_websocket_message(websocket, message)
                except json.JSONDecodeError:
                    await websocket_manager.send_personal_message(websocket, {
                        "type": "error",
                        "message": "Invalid JSON format"
                    })
                
        except WebSocketDisconnect:
            websocket_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            websocket_manager.disconnect(websocket)
    
    async def handle_websocket_message(websocket: WebSocket, message: dict):
        """Handle incoming WebSocket message"""
        message_type = message.get("type")
        
        if message_type == "subscribe_project":
            project = message.get("project")
            if project:
                await websocket_manager.subscribe_to_project(websocket, project)
            else:
                await websocket_manager.send_personal_message(websocket, {
                    "type": "error",
                    "message": "Project name required for subscription"
                })
        
        elif message_type == "ping":
            await websocket_manager.send_personal_message(websocket, {
                "type": "pong",
                "timestamp": message.get("timestamp")
            })
        
        else:
            await websocket_manager.send_personal_message(websocket, {
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            })
    
    # Event handlers for broadcasting
    async def broadcast_commit_received(event_data: dict):
        """Broadcast commit received event"""
        project = event_data.get("project")
        if project:
            await websocket_manager.broadcast_to_project(project, {
                "type": "commit_received",
                "data": event_data
            })
    
    async def broadcast_commit_analyzed(event_data: dict):
        """Broadcast commit analysis completed event"""
        # In a real implementation, you'd get the project from the commit
        await websocket_manager.broadcast_to_all({
            "type": "commit_analyzed",
            "data": event_data
        })
    
    ## 5. Core Configuration
    
    ### src/core/config.py
    ```python
    from pydantic_settings import BaseSettings
    from typing import Optional
    import os
    
    class Settings(BaseSettings):
        """Application settings"""
        
        # Application
        app_name: str = "Git AI Analytics"
        app_version: str = "1.0.0"
        environment: str = "development"
        debug: bool = True
        
        # Server
        host: str = "0.0.0.0"
        port: int = 8000
        workers: int = 1
        reload: bool = True
        
        # Database
        db_provider: str = "sqlite"
        db_path: str = "data/git_analytics.db"
        db_host: Optional[str] = None
        db_port: Optional[int] = None
        db_name: Optional[str] = None
        db_user: Optional[str] = None
        db_password: Optional[str] = None
        db_echo: bool = False
        
        # Redis
        redis_url: str = "redis://localhost:6379"
        redis_password: Optional[str] = None
        
        # Celery
        celery_broker_url: str = "redis://localhost:6379/0"
        celery_result_backend: str = "redis://localhost:6379/0"
        
        # AI/Ollama
        ollama_base_url: str = "http://localhost:11434"
        ollama_model: str = "llama3.2:3b"
        ollama_timeout: int = 30
        
        # Security
        webhook_secret: Optional[str] = None
        api_key: Optional[str] = None
        
        # Cache
        cache_ttl_default: int = 3600
        cache_ttl_search: int = 300
        cache_ttl_chat: int = 600
        memory_cache_size: int = 1000
        
        # Logging
        log_level: str = "INFO"
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        class Config:
            env_file = ".env"
            case_sensitive = False
    
    # Global settings instance
    settings = Settings()
    
    ### src/core/logging.py
    ```python
    import logging
    import logging.config
    from .config import settings
    
    def setup_logging():
        """Setup logging configuration"""
        
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": settings.log_format,
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": settings.log_level,
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.FileHandler",
                    "level": "INFO",
                    "formatter": "detailed",
                    "filename": "logs/app.log",
                    "mode": "a",
                },
            },
            "loggers": {
                "": {
                    "level": settings.log_level,
                    "handlers": ["console"],
                    "propagate": False,
                },
                "src": {
                    "level": settings.log_level,
                    "handlers": ["console", "file"] if settings.environment == "production" else ["console"],
                    "propagate": False,
                },
                "uvicorn": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "celery": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
        }
        
        # Create logs directory
        import os
        os.makedirs("logs", exist_ok=True)
        
        logging.config.dictConfig(logging_config)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured for environment: {settings.environment}")
    
    ### src/main.py
    ```python
    import asyncio
    import logging
    from contextlib import asynccontextmanager
    
    from fastapi import FastAPI, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    
    from src.core.config import settings
    from src.core.logging import setup_logging
    from src.infrastructure.persistence.database import init_database, close_database
    from src.presentation.api.v1 import webhooks, commits, chat
    from src.presentation.websocket.handlers import websocket_endpoint
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager"""
        # Startup
        logger.info(f"Starting {settings.app_name} v{settings.app_version}")
        logger.info(f"Environment: {settings.environment}")
        
        # Initialize database
        await init_database()
        
        yield
        
        # Shutdown
        logger.info("Shutting down application")
        await close_database()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Git commit analytics with AI-powered insights",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routers
    app.include_router(webhooks.router, prefix="/api/v1")
    app.include_router(commits.router, prefix="/api/v1")
    app.include_router(chat.router, prefix="/api/v1")
    
    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_route(websocket: WebSocket):
        await websocket_endpoint(websocket)
    
    # Health check
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "app": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment
        }
    
    # Root endpoint with simple UI
    @app.get("/", response_class=HTMLResponse)
    async def root():
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Git AI Analytics</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .method { font-weight: bold; color: #2196F3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 Git AI Analytics API</h1>
                <p>AI-powered Git commit analysis and search system</p>
                
                <div class="endpoint">
                    <span class="method">POST</span> /api/v1/webhooks/github - GitHub webhook handler
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> /api/v1/commits/ - Search commits
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> /api/v1/commits/{id} - Get specific commit
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> /api/v1/chat/ - Chat with AI about commits
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> /health - Health check
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> /docs - Interactive API documentation
                </div>
                
                <h2>WebSocket:</h2>
                <div class="endpoint">
                    <span class="method">WS</span> /ws - Real-time updates
                </div>
                
                <h2>Quick Start:</h2>
                <ol>
                    <li>Setup webhook: POST to /api/v1/webhooks/github</li>
                    <li>Search commits: GET /api/v1/commits/?q=bugfix</li>
                    <li>Chat with AI: POST /api/v1/chat/ with your question</li>
                </ol>
            </div>
        </body>
        </html>
        """
    
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(
            "src.main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.reload,
            workers=settings.workers if not settings.reload else 1,
            log_level=settings.log_level.lower()
        )
    
    ## 6. Configuration Files
    
    ### configs/default.yaml
    ```yaml
    app:
      name: "Git AI Analytics"
      version: "1.0.0"
      environment: "development"
      debug: true
    
    server:
      host: "0.0.0.0"
      port: 8000
      workers: 1
      reload: true
    
    database:
      provider: "sqlite"  # sqlite, postgresql
      sqlite:
        path: "data/git_analytics.db"
      postgresql:
        host: "localhost"
        port: 5432
        database: "git_analytics"
        username: "postgres"
        password: ""
      echo: false
    
    cache:
      provider: "memory"  # memory, redis
      redis:
        url: "redis://localhost:6379"
        password: null
      memory:
        max_size: 1000
        default_ttl: 3600
      ttl:
        default: 3600
        search: 300
        chat: 600
    
    ai:
      provider: "ollama"
      ollama:
        base_url: "http://localhost:11434"
        model: "llama3.2:3b"
        timeout: 30
      
    messaging:
      provider: "celery"
      celery:
        broker_url: "redis://localhost:6379/0"
        result_backend: "redis://localhost:6379/0"
      
    security:
      webhook_secret: null
      api_key: null
    
    logging:
      level: "INFO"
      format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ```
    
    ### configs/development.yaml
    ```yaml
    app:
      debug: true
      environment: "development"
    
    server:
      reload: true
      workers: 1
    
    database:
      provider: "sqlite"
      echo: true
    
    cache:
      provider: "memory"
    
    logging:
      level: "DEBUG"
    ```
    
    ### configs/production.yaml
    ```yaml
    app:
      debug: false
      environment: "production"
    
    server:
      reload: false
      workers: 4
    
    database:
      provider: "postgresql"
      echo: false
    
    cache:
      provider: "redis"
    
    logging:
      level: "INFO"
    ```
    
    ## 7. Requirements
    
    ### requirements/base.txt
    ```
    # FastAPI and server
    fastapi==0.104.1
    uvicorn[standard]==0.24.0
    python-multipart==0.0.6
    
    # Database
    sqlalchemy[asyncio]==2.0.23
    alembic==1.12.1
    aiosqlite==0.19.0
    asyncpg==0.29.0
    
    # Redis and Celery
    redis[hiredis]==5.0.1
    celery[redis]==5.3.4
    
    # AI and HTTP
    httpx==0.25.2
    ollama==0.1.7
    
    # Data validation
    pydantic==2.5.0
    pydantic-settings==2.1.0
    
    # Utilities
    python-dotenv==1.0.0
    PyYAML==6.0.1
    ```
    
    ### requirements/development.txt
    ```
    -r base.txt
    
    # Development tools
    pytest==7.4.3
    pytest-asyncio==0.21.1
    pytest-cov==4.1.0
    black==23.11.0
    isort==5.12.0
    flake8==6.1.0
    mypy==1.7.1
    
    # Testing
    httpx==0.25.2
    pytest-mock==3.12.0
    ```
    
    ### requirements/production.txt
    ```
    -r base.txt
    
    # Production extras
    gunicorn==21.2.0
    prometheus-client==0.19.0
    sentry-sdk[fastapi]==1.39.1
    ```
    
    ## 8. Docker Configuration
    
    ### docker/Dockerfile
    ```dockerfile
    FROM python:3.11-slim
    
    # Set environment variables
    ENV PYTHONDONTWRITEBYTECODE=1
    ENV PYTHONUNBUFFERED=1
    ENV PYTHONPATH=/app
    
    # Set work directory
    WORKDIR /app
    
    # Install system dependencies
    RUN apt-get update \
        && apt-get install -y --no-install-recommends \
            curl \
            build-essential \
        && rm -rf /var/lib/apt/lists/*
    
    # Install Python dependencies
    COPY requirements/ requirements/
    RUN pip install --no-cache-dir -r requirements/production.txt
    
    # Copy project
    COPY . .
    
    # Create directories
    RUN mkdir -p data logs
    
    # Create non-root user
    RUN adduser --disabled-password --gecos '' appuser \
        && chown -R appuser:appuser /app
    USER appuser
    
    # Health check
    HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
        CMD curl -f http://localhost:8000/health || exit 1
    
    # Default command
    CMD ["python", "-m", "src.main"]
    ```
    
    ### docker/docker-compose.yml
    ```yaml
    version: '3.8'
    
    services:
      app:
        build:
          context: ..
          dockerfile: docker/Dockerfile
        ports:
          - "8000:8000"
        environment:
          - DB_PROVIDER=postgresql
          - DB_HOST=postgres
          - DB_NAME=git_analytics
          - DB_USER=postgres
          - DB_PASSWORD=postgres123
          - REDIS_URL=redis://redis:6379
          - OLLAMA_BASE_URL=http://ollama:11434
        depends_on:
          - postgres
          - redis
          - ollama
        volumes:
          - app_data:/app/data
          - app_logs:/app/logs
    
      postgres:
        image: postgres:15-alpine
        environment:
          POSTGRES_DB: git_analytics
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres123
        volumes:
          - postgres_data:/var/lib/postgresql/data
        ports:
          - "5432:5432"
    
      redis:
        image: redis:7-alpine
        command: redis-server --appendonly yes
        volumes:
          - redis_data:/data
        ports:
          - "6379:6379"
    
      ollama:
        image: ollama/ollama:latest
        volumes:
          - ollama_data:/root/.ollama
        ports:
          - "11434:11434"
        environment:
          - OLLAMA_KEEP_ALIVE=24h
    
      celery:
        build:
          context: ..
          dockerfile: docker/Dockerfile
        command: celery -A src.infrastructure.messaging.celery_config worker --loglevel=info
        environment:
          - DB_PROVIDER=postgresql
          - DB_HOST=postgres
          - DB_NAME=git_analytics
          - DB_USER=postgres
          - DB_PASSWORD=postgres123
          - REDIS_URL=redis://redis:6379
          - OLLAMA_BASE_URL=http://ollama:11434
        depends_on:
          - postgres
          - redis
          - ollama
        volumes:
          - app_data:/app/data
    
    volumes:
      postgres_data:
      redis_data:
      ollama_data:
      app_data:
      app_logs:
    ```
    
    ## 9. Environment Configuration
    
    ### .env.example
    ```bash
    # Application
    APP_NAME="Git AI Analytics"
    APP_VERSION="1.0.0"
    ENVIRONMENT="development"
    DEBUG="true"
    
    # Server
    HOST="0.0.0.0"
    PORT="8000"
    WORKERS="1"
    RELOAD="true"
    
    # Database
    DB_PROVIDER="sqlite"  # sqlite, postgresql
    DB_PATH="data/git_analytics.db"
    DB_HOST="localhost"
    DB_PORT="5432"
    DB_NAME="git_analytics"
    DB_USER="postgres"
    DB_PASSWORD=""
    DB_ECHO="false"
    
    # Redis
    REDIS_URL="redis://localhost:6379"
    REDIS_PASSWORD=""
    
    # Celery
    CELERY_BROKER_URL="redis://localhost:6379/0"
    CELERY_RESULT_BACKEND="redis://localhost:6379/0"
    
    # AI/Ollama
    OLLAMA_BASE_URL="http://localhost:11434"
    OLLAMA_MODEL="llama3.2:3b"
    OLLAMA_TIMEOUT="30"
    
    # Security
    WEBHOOK_SECRET=""
    API_KEY=""
    
    # Cache
    CACHE_TTL_DEFAULT="3600"
    CACHE_TTL_SEARCH="300"
    CACHE_TTL_CHAT="600"
    MEMORY_CACHE_SIZE="1000"
    
    # Logging
    LOG_LEVEL="INFO"
    ```
    
    ## 10. Testing
    
    ### tests/test_commit_entity.py
    ```python
    import pytest
    from datetime import datetime
    from src.domain.entities.commit import Commit, CommitHash, FileChange
    
    def test_commit_creation():
        """Test basic commit creation"""
        commit = Commit(
            commit_hash=CommitHash("abc123def456"),
            author_email="dev@example.com",
            author_name="Developer",
            message="Add new feature",
            timestamp=datetime.utcnow(),
            branch="main",
            project="test-project",
            files_changed=[
                FileChange("src/main.py", 10, 2, "modified")
            ]
        )
        
        assert commit.commit_hash.value == "abc123def456"
        assert commit.author_email == "dev@example.com"
        assert commit.metrics.files_count == 1
        assert commit.metrics.total_lines_changed == 12
    
    def test_commit_complexity_calculation():
        """Test commit complexity calculation"""
        commit = Commit(
            commit_hash=CommitHash("abc123def456"),
            author_email="dev@example.com",
            author_name="Developer",
            message="Major refactor",
            timestamp=datetime.utcnow(),
            branch="main",
            project="test-project",
            files_changed=[
                FileChange("src/config.yaml", 50, 30, "modified"),
                FileChange("src/main.py", 100, 20, "modified"),
                FileChange("src/models.py", 200, 50, "modified"),
            ]
        )
        
        complexity = commit.metrics.complexity_score
        assert 0 <= complexity <= 1
        assert complexity > 0.5  # Should be complex due to many changes
    
    def test_commit_business_rules():
        """Test commit business rules"""
        hotfix_commit = Commit(
            commit_hash=CommitHash("hotfix123"),
            author_email="dev@example.com",
            author_name="Developer",
            message="HOTFIX: Critical bug in payment",
            timestamp=datetime.utcnow(),
            branch="main",
            project="test-project",
            files_changed=[]
        )
        
        feature_commit = Commit(
            commit_hash=CommitHash("feature123"),
            author_email="dev@example.com",
            author_name="Developer",
            message="feat: Add user authentication",
            timestamp=datetime.utcnow(),
            branch="feature/auth",
            project="test-project",
            files_changed=[]
        )
        
        assert hotfix_commit.is_hotfix()
        assert not hotfix_commit.is_feature()
        assert feature_commit.is_feature()
        assert not feature_commit.is_hotfix()
    
    ## Usage Examples
    
    ### Basic Setup and Usage
    ```bash
    # 1. Clone and setup
    git clone <repository>
    cd git-ai-analytics
    
    # 2. Install dependencies
    pip install -r requirements/development.txt
    
    # 3. Setup environment
    cp .env.example .env
    # Edit .env with your settings
    
    # 4. Initialize database
    python -c "
    import asyncio
    from src.infrastructure.persistence.database import init_database
    asyncio.run(init_database())
    "
    
    # 5. Start Ollama (in separate terminal)
    ollama serve
    ollama pull llama3.2:3b
    
    # 6. Start Redis (in separate terminal)
    redis-server
    
    # 7. Start Celery worker (in separate terminal)
    celery -A src.infrastructure.messaging.celery_config worker --loglevel=info
    
    # 8. Start the application
    python -m src.main
    ```
    
    ### Docker Setup
    ```bash
    # 1. Build and start all services
    docker-compose -f docker/docker-compose.yml up --build
    
    # 2. Pull Ollama model
    docker-compose exec ollama ollama pull llama3.2:3b
    
    # 3. Check health
    curl http://localhost:8000/health
    ```
    
    ### GitHub Webhook Setup
    ```bash
    # 1. Setup webhook in GitHub repository settings:
    # URL: http://your-server.com/api/v1/webhooks/github
    # Content type: application/json
    # Events: Push events
    
    # 2. Test webhook locally with ngrok:
    ngrok http 8000
    # Use the ngrok URL for webhook
    ```
    
    ### API Usage Examples
    ```bash
    # Search commits
    curl "http://localhost:8000/api/v1/commits/?q=bugfix&limit=10"
    
    # Get commit statistics
    curl "http://localhost:8000/api/v1/commits/stats/overview?project=my-project"
    
    # Chat with AI
    curl -X POST "http://localhost:8000/api/v1/chat/" \
      -H "Content-Type: application/json" \
      -d '{
        "question": "What did John work on this week?",
        "context_project": "my-project",
        "context_author": "john@company.com",
        "context_days": 7
      }'
    ```
    
    This implementation provides:
    
    ✅ **Clean Architecture** with clear separation of concerns
    ✅ **Event-Driven Design** with domain events and handlers  
    ✅ **Dependency Injection** using FastAPI's DI system
    ✅ **Repository Pattern** with SQLAlchemy/SQLite → PostgreSQL migration path
    ✅ **Multi-layer Caching** (Memory + Redis)
    ✅ **Async Message Queue** with Celery for heavy AI processing
    ✅ **WebSocket Support** for real-time updates
    ✅ **AI Integration** with Ollama (free, lightweight)
    ✅ **Configuration Management** with environment-based configs
    ✅ **Docker Support** for easy deployment
    ✅ **Comprehensive Error Handling** and logging
    ✅ **Scalable Design** ready for 20+ developers
    
    The system can handle webhook → processing → analysis → search/chat flow efficiently and is designed to scale horizontally.### src/infrastructure/messaging/tasks/analysis_tasks.py
    ```python
    from celery import shared_task
    from celery.utils.log import get_task_logger
    from typing import Dict, Any
    import traceback
    import asyncio
    from ...persistence.database import get_session
    from ...persistence.repositories.sqlite_commit_repository import SqliteCommitRepository
    from ...ai.ollama_service import OllamaService
    from ....domain.events.commit_events import (
        CommitAnalysisStartedEvent,
        CommitAnalysisCompletedEvent,
        CommitAnalysisFailedEvent
    )
    
    logger = get_task_logger(__name__)
    
    @shared_task(bind=True, name='analyze_commit')
    def analyze_commit_task(self, commit_id: str) -> Dict[str, Any]:
        """Analyze commit with AI"""
        try:
            logger.info(f"Starting analysis for commit: {commit_id}")
            
            # Run async analysis
            result = asyncio.run(_analyze_commit_async(commit_id))
            
            logger.info(f"Successfully analyzed commit: {commit_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze commit {commit_id}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fire failure event
            asyncio.run(_fire_analysis_failed_event(commit_id, str(e), self.request.retries))
            
            # Re-raise for Celery retry mechanism
            raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))
    
    async def _analyze_commit_async(commit_id: str) -> Dict[str, Any]:
        """Async function to analyze commit"""
        
        # Get dependencies
        async with get_session() as session:
            commit_repo = SqliteCommitRepository(session)
            ai_service = OllamaService()
            
            # Load commit
            commit = await commit_repo.find_by_id(commit_id)
            if not commit:
                raise ValueError(f"Commit {commit_id} not found")
            
            # Fire analysis started event
            started_event = CommitAnalysisStartedEvent(
                commit_id=commit_id,
                model_name=ai_service.get_model_info()['name']
            )
            # Note: In a full implementation, we'd dispatch this event
            
            # Generate AI analysis
            analysis_result = await ai_service.analyze_commit(commit)
            
            # Update commit with analysis
            commit.mark_as_analyzed(
                summary=analysis_result.summary,
                tags=analysis_result.tags,
                sentiment=analysis_result.sentiment_score
            )
            
            updated_commit = await commit_repo.update(commit)
            
            # Fire analysis completed event
            completed_event = CommitAnalysisCompletedEvent(
                commit_id=commit_id,
                analysis_id=f"analysis_{commit_id}",
                summary=analysis_result.summary,
                tags=analysis_result.tags,
                processing_time_ms=1000  # Placeholder
            )
            # Note: In a full implementation, we'd dispatch this event
            
            return {
                'status': 'success',
                'commit_id': commit_id,
                'summary': analysis_result.summary,
                'tags': analysis_result.tags,
                'sentiment_score': analysis_result.sentiment_score,
                'confidence_score': analysis_result.confidence_score
            }
    
    async def _fire_analysis_failed_event(commit_id: str, error_message: str, retry_count: int):
        """Fire analysis failed event"""
        failed_event = CommitAnalysisFailedEvent(
            commit_id=commit_id,
            error_message=error_message,
            retry_count=retry_count
        )
        # Note: In a full implementation, we'd dispatch this event
    
    @shared_task(name='generate_summary')
    def generate_summary_task(project: str, days: int = 7) -> Dict[str, Any]:
        """Generate project summary for recent commits"""
        try:
            logger.info(f"Generating summary for project: {project}")
            
            result = asyncio.run(_generate_summary_async(project, days))
            
            logger.info(f"Successfully generated summary for project: {project}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate summary for project {project}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def _generate_summary_async(project: str, days: int) -> Dict[str, Any]:
        """Async function to generate summary"""
        from datetime import datetime, timedelta
        
        async with get_session() as session:
            commit_repo = SqliteCommitRepository(session)
            ai_service = OllamaService()
            
            # Get recent commits
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            commits = await commit_repo.find_by_project(
                project=project,
                start_date=start_date,
                end_date=end_date,
                limit=50
            )
            
            if not commits:
                return {
                    'status': 'success',
                    'project': project,
                    'summary': f"No commits found for {project} in the last {days} days.",
                    'commits_count': 0
                }
            
            # Generate summary
            summary = await ai_service.generate_summary(commits)
            
            return {
                'status': 'success',
                'project': project,
                'summary': summary,
                'commits_count': len(commits),
                'period_days': days
            }
    
    ### src/infrastructure/events/event_bus.py
    ```python
    import asyncio
    import json
    import logging
    from typing import Callable, Dict, List
    from redis.asyncio import Redis
    from ...domain.events.base import DomainEvent
    from ...application.services.event_dispatcher import IEventDispatcher
    
    logger = logging.getLogger(__name__)
    
    class RedisEventBus(IEventDispatcher):
        """Redis-based event bus for distributed event handling"""
        
        def __init__(self, redis_client: Redis, channel_prefix: str = "events"):
            self.redis = redis_client
            self.channel_prefix = channel_prefix
            self.handlers: Dict[str, List[Callable]] = {}
            self.subscriber_task = None
            self._running = False
        
        async def start(self):
            """Start event bus subscriber"""
            if not self._running:
                self._running = True
                self.subscriber_task = asyncio.create_task(self._subscriber_loop())
                logger.info("### src/infrastructure/persistence/repositories/sqlite_commit_repository.py
    ```python
    from typing import List, Optional, Dict, Any
    from datetime import datetime
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import select, and_, or_, desc, func
    from sqlalchemy.orm import selectinload
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.entities.commit import Commit, CommitHash, FileChange
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
                    id=commit.id,
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
                logger.error(f"Error saving commit {commit.commit_hash.value}: {str(e)}")
                raise
        
        async def find_by_id(self, commit_id: str) -> Optional[Commit]:
            """Find commit by ID"""
            try:
                stmt = select(CommitModel).where(CommitModel.id == commit_id)
                result = await self.session.execute(stmt)
                db_commit = result.scalar_one_or_none()
                
                return self._to_domain(db_commit) if db_commit else None
            except Exception as e:
                logger.error(f"Error finding commit by ID {commit_id}: {str(e)}")
                return None
        
        async def find_by_hash(self, commit_hash: str) -> Optional[Commit]:
            """Find commit by hash"""
            try:
                stmt = select(CommitModel).where(CommitModel.commit_hash == commit_hash)
                result = await self.session.execute(stmt)
                db_commit = result.scalar_one_or_none()
                
                return self._to_domain(db_commit) if db_commit else None
            except Exception as e:
                logger.error(f"Error finding commit by hash {commit_hash}: {str(e)}")
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
                logger.error(f"Error finding commits by author {author_email}: {str(e)}")
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
                logger.error(f"Error finding commits by project {project}: {str(e)}")
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
                logger.error(f"Error searching commits with query '{query}': {str(e)}")
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
                logger.error(f"Error updating commit {commit.id}: {str(e)}")
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
                logger.error(f"Error getting statistics: {str(e)}")
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
    
    ### src/infrastructure/ai/ollama_service.py
    ```python
    import httpx
    import json
    import logging
    from typing import Dict, Any, List, Optional
    from ...domain.services.ai_analyzer import IAIAnalyzer
    from ...domain.entities.commit import Commit
    from ...domain.entities.analysis import AnalysisResult
    from .prompts.commit_prompts import CommitPrompts
    import time
    import os
    
    logger = logging.getLogger(__name__)
    
    class OllamaService(IAIAnalyzer):
        """Ollama AI service implementation"""
        
        def __init__(
            self,
            base_url: str = None,
            model: str = None,
            timeout: int = 30
        ):
            self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
            self.timeout = timeout
            self.prompts = CommitPrompts()
            
            # Remove trailing slash
            self.base_url = self.base_url.rstrip('/')
        
        async def analyze_commit(self, commit: Commit) -> AnalysisResult:
            """Analyze a commit using Ollama"""
            prompt = self.prompts.get_commit_analysis_prompt(commit)
            
            try:
                start_time = time.time()
                response = await self._generate_response(prompt, temperature=0.3)
                processing_time = int((time.time() - start_time) * 1000)
                
                # Parse structured response
                analysis_data = self._parse_analysis_response(response)
                
                return AnalysisResult(
                    summary=analysis_data.get('summary', 'No summary available'),
                    tags=analysis_data.get('tags', []),
                    sentiment_score=analysis_data.get('sentiment', 0.0),
                    confidence_score=analysis_data.get('confidence', 0.5),
                    extracted_entities=analysis_data.get('entities', [])
                )
                
            except Exception as e:
                logger.error(f"Error analyzing commit {commit.commit_hash.value}: {str(e)}")
                # Return default analysis on error
                return AnalysisResult(
                    summary=f"Analysis failed: {str(e)}",
                    tags=['error'],
                    sentiment_score=0.0,
                    confidence_score=0.0,
                    extracted_entities=[]
                )
        
        async def generate_summary(self, commits: List[Commit]) -> str:
            """Generate summary for multiple commits"""
            if not commits:
                return "No commits to summarize."
            
            # Git AI Analytics - Full Implementation
    
    ## Project Structure
    ```
    git-ai-analytics/
    ├── src/
    │   ├── domain/                    # Domain Layer
    │   │   ├── entities/
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py
    │   │   │   ├── commit.py
    │   │   │   ├── developer.py
    │   │   │   └── analysis.py
    │   │   ├── events/
    │   │   │   ├── __init__.py
    │   │   │   ├── base.py
    │   │   │   └── commit_events.py
    │   │   ├── repositories/
    │   │   │   ├── __init__.py
    │   │   │   ├── commit_repository.py
    │   │   │   └── analysis_repository.py
    │   │   └── services/
    │   │       ├── __init__.py
    │   │       └── ai_analyzer.py
    │   ├── application/               # Application Layer
    │   │   ├── use_cases/
    │   │   │   ├── __init__.py
    │   │   │   ├── process_commit.py
    │   │   │   ├── analyze_commit.py
    │   │   │   └── search_commits.py
    │   │   ├── dto/
    │   │   │   ├── __init__.py
    │   │   │   ├── commit_dto.py
    │   │   │   └── webhook_dto.py
    │   │   ├── services/
    │   │   │   ├── __init__.py
    │   │   │   └── event_dispatcher.py
    │   │   └── interfaces/
    │   │       ├── __init__.py
    │   │       ├── cache_interface.py
    │   │       ├── ai_interface.py
    │   │       └── messaging_interface.py
    │   ├── infrastructure/           # Infrastructure Layer
    │   │   ├── persistence/
    │   │   │   ├── __init__.py
    │   │   │   ├── models/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── commit_model.py
    │   │   │   ├── repositories/
    │   │   │   │   ├── __init__.py
    │   │   │   │   └── sqlite_commit_repository.py
    │   │   │   └── database.py
    │   │   ├── ai/
    │   │   │   ├── __init__.py
    │   │   │   ├── ollama_service.py
    │   │   │   └── prompts/
    │   │   │       └── commit_prompts.py
    │   │   ├── cache/
    │   │   │   ├── __init__.py
    │   │   │   ├── memory_cache.py
    │   │   │   └── redis_cache.py
    │   │   ├── messaging/
    │   │   │   ├── __init__.py
    │   │   │   ├── celery_config.py
    │   │   │   └── tasks/
    │   │   │       ├── __init__.py
    │   │   │       └── analysis_tasks.py
    │   │   └── events/
    │   │       ├── __init__.py
    │   │       └── event_bus.py
    │   ├── presentation/             # Presentation Layer
    │   │   ├── api/
    │   │   │   ├── __init__.py
    │   │   │   ├── v1/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── webhooks.py
    │   │   │   │   ├── commits.py
    │   │   │   │   └── chat.py
    │   │   │   └── dependencies.py
    │   │   └── websocket/
    │   │       ├── __init__.py
    │   │       └── handlers.py
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── config.py
    │   │   └── logging.py
    │   └── main.py
    ├── configs/
    │   ├── default.yaml
    │   ├── development.yaml
    │   └── production.yaml
    ├── tests/
    ├── migrations/
    ├── docker/
    │   ├── Dockerfile
    │   └── docker-compose.yml
    ├── requirements/
    │   ├── base.txt
    │   ├── development.txt
    │   └── production.txt
    ├── .env.example
    └── README.md
    ```
    
    ## 1. Domain Layer
    
    ### src/domain/entities/base.py
    ```python
    from abc import ABC
    from datetime import datetime
    from typing import Any
    from uuid import uuid4, UUID
    from dataclasses import dataclass, field
    
    @dataclass
    class ValueObject(ABC):
        """Base class for value objects"""
        
        def __post_init__(self):
            self._validate()
        
        def _validate(self):
            """Override to add validation logic"""
            pass
    
    class Entity(ABC):
        """Base class for entities"""
        
        def __init__(self):
            self.id: str = str(uuid4())
            self.created_at: datetime = datetime.utcnow()
            self.updated_at: datetime = datetime.utcnow()
        
        def __eq__(self, other):
            if not isinstance(other, self.__class__):
                return False
            return self.id == other.id
        
        def __hash__(self):
            return hash(self.id)
    ```
    
    ### src/domain/entities/commit.py
    ```python
    from dataclasses import dataclass
    from datetime import datetime
    from typing import List, Optional
    from .base import Entity, ValueObject
    
    @dataclass
    class CommitHash(ValueObject):
        """Value object for commit hash"""
        value: str
        
        def _validate(self):
            if not self.value or len(self.value) < 6:
                raise ValueError("Invalid commit hash")
    
    @dataclass
    class FileChange(ValueObject):
        """Value object for file changes"""
        filename: str
        additions: int
        deletions: int
        status: str  # added, modified, deleted, renamed
        
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
        """Commit domain entity"""
        
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
            self.commit_hash = commit_hash
            self.author_email = author_email
            self.author_name = author_name
            self.message = message
            self.timestamp = timestamp
            self.branch = branch
            self.project = project
            self.files_changed = files_changed or []
            self.issue_numbers = issue_numbers or []
            
            # Analysis results (populated later)
            self.summary: Optional[str] = None
            self.tags: List[str] = []
            self.sentiment_score: Optional[float] = None
            self.embedding_id: Optional[str] = None
            self.analyzed_at: Optional[datetime] = None
            
            # Computed properties
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
    ```
    
    ### src/domain/entities/analysis.py
    ```python
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, List, Optional
    from .base import Entity, ValueObject
    
    @dataclass
    class AnalysisResult(ValueObject):
        """Value object for AI analysis results"""
        summary: str
        tags: List[str]
        sentiment_score: float
        confidence_score: float
        extracted_entities: List[str]
        
        def _validate(self):
            if not (-1 <= self.sentiment_score <= 1):
                raise ValueError("Sentiment score must be between -1 and 1")
            if not (0 <= self.confidence_score <= 1):
                raise ValueError("Confidence score must be between 0 and 1")
    
    class Analysis(Entity):
        """Analysis domain entity"""
        
        def __init__(
            self,
            commit_id: str,
            model_name: str,
            model_version: str,
            prompt_version: str,
            result: AnalysisResult,
            processing_time_ms: int,
            tokens_used: int = 0
        ):
            super().__init__()
            self.commit_id = commit_id
            self.model_name = model_name
            self.model_version = model_version
            self.prompt_version = prompt_version
            self.result = result
            self.processing_time_ms = processing_time_ms
            self.tokens_used = tokens_used
            self.status = "completed"
        
        def is_high_confidence(self) -> bool:
            """Check if analysis has high confidence"""
            return self.result.confidence_score >= 0.8
    ```
    
    ### src/domain/events/base.py
    ```python
    from abc import ABC
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, Any
    import json
    from uuid import uuid4
    
    @dataclass
    class DomainEvent(ABC):
        """Base class for domain events"""
        event_id: str = None
        occurred_at: datetime = None
        
        def __post_init__(self):
            if self.event_id is None:
                self.event_id = str(uuid4())
            if self.occurred_at is None:
                self.occurred_at = datetime.utcnow()
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert event to dictionary"""
            return {
                'event_type': self.__class__.__name__,
                'event_id': self.event_id,
                'occurred_at': self.occurred_at.isoformat(),
                'data': self._get_event_data()
            }
        
        def to_json(self) -> str:
            """Convert event to JSON string"""
            return json.dumps(self.to_dict(), default=str)
        
        def _get_event_data(self) -> Dict[str, Any]:
            """Override to provide event-specific data"""
            return {}
    ```
    
    ### src/domain/events/commit_events.py
    ```python
    from dataclasses import dataclass
    from typing import Dict, Any, List
    from .base import DomainEvent
    
    @dataclass
    class CommitReceivedEvent(DomainEvent):
        """Event fired when a new commit is received"""
        commit_id: str
        commit_hash: str
        project: str
        author: str
        branch: str
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'commit_hash': self.commit_hash,
                'project': self.project,
                'author': self.author,
                'branch': self.branch
            }
    
    @dataclass 
    class CommitAnalysisStartedEvent(DomainEvent):
        """Event fired when commit analysis starts"""
        commit_id: str
        model_name: str
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'model_name': self.model_name
            }
    
    @dataclass
    class CommitAnalysisCompletedEvent(DomainEvent):
        """Event fired when commit analysis completes"""
        commit_id: str
        analysis_id: str
        summary: str
        tags: List[str]
        processing_time_ms: int
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'analysis_id': self.analysis_id,
                'summary': self.summary,
                'tags': self.tags,
                'processing_time_ms': self.processing_time_ms
            }
    
    @dataclass
    class CommitAnalysisFailedEvent(DomainEvent):
        """Event fired when commit analysis fails"""
        commit_id: str
        error_message: str
        retry_count: int
        
        def _get_event_data(self) -> Dict[str, Any]:
            return {
                'commit_id': self.commit_id,
                'error_message': self.error_message,
                'retry_count': self.retry_count
            }
    ```
    
    ### src/domain/repositories/commit_repository.py
    ```python
    from abc import ABC, abstractmethod
    from typing import List, Optional, Dict, Any
    from datetime import datetime
    from ..entities.commit import Commit
    
    class ICommitRepository(ABC):
        """Repository interface for Commit entity"""
        
        @abstractmethod
        async def save(self, commit: Commit) -> Commit:
            """Save a commit"""
            pass
        
        @abstractmethod
        async def find_by_id(self, commit_id: str) -> Optional[Commit]:
            """Find commit by ID"""
            pass
        
        @abstractmethod
        async def find_by_hash(self, commit_hash: str) -> Optional[Commit]:
            """Find commit by hash"""
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
    ```
    
    ### src/domain/services/ai_analyzer.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, List
    from ..entities.commit import Commit
    from ..entities.analysis import AnalysisResult
    
    class IAIAnalyzer(ABC):
        """Domain service interface for AI analysis"""
        
        @abstractmethod
        async def analyze_commit(self, commit: Commit) -> AnalysisResult:
            """Analyze a commit using AI"""
            pass
        
        @abstractmethod
        async def generate_summary(self, commits: List[Commit]) -> str:
            """Generate summary for multiple commits"""
            pass
        
        @abstractmethod
        async def answer_question(
            self, 
            question: str, 
            context_commits: List[Commit]
        ) -> str:
            """Answer question about commits"""
            pass
        
        @abstractmethod
        async def extract_entities(self, text: str) -> List[str]:
            """Extract named entities from text"""
            pass
        
        @abstractmethod
        def get_model_info(self) -> Dict[str, str]:
            """Get current model information"""
            pass
    ```
    
    ## 2. Application Layer
    
    ### src/application/dto/commit_dto.py
    ```python
    from pydantic import BaseModel, Field, validator
    from datetime import datetime
    from typing import List, Optional
    
    class FileChangeDto(BaseModel):
        """DTO for file changes"""
        filename: str = Field(..., min_length=1)
        additions: int = Field(..., ge=0)
        deletions: int = Field(..., ge=0)
        status: str = Field(..., regex="^(added|modified|deleted|renamed)$")
    
    class CommitDto(BaseModel):
        """DTO for commit data"""
        commit_hash: str = Field(..., min_length=6)
        author_email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
        author_name: str = Field(..., min_length=1)
        message: str = Field(..., min_length=1)
        timestamp: datetime
        branch: str = Field(..., min_length=1)
        project: str = Field(..., min_length=1)
        files_changed: List[FileChangeDto] = []
        issue_numbers: List[str] = []
    
    class CommitResponseDto(BaseModel):
        """DTO for commit response"""
        id: str
        commit_hash: str
        author_email: str
        author_name: str
        message: str
        timestamp: datetime
        branch: str
        project: str
        files_count: int
        total_lines_changed: int
        complexity_score: Optional[float] = None
        summary: Optional[str] = None
        tags: List[str] = []
        sentiment_score: Optional[float] = None
        analyzed_at: Optional[datetime] = None
        created_at: datetime
    
    class CommitSearchDto(BaseModel):
        """DTO for commit search"""
        query: Optional[str] = None
        project: Optional[str] = None
        author: Optional[str] = None
        branch: Optional[str] = None
        start_date: Optional[datetime] = None
        end_date: Optional[datetime] = None
        tags: List[str] = []
        limit: int = Field(50, ge=1, le=200)
        offset: int = Field(0, ge=0)
    
    class ChatRequestDto(BaseModel):
        """DTO for chat requests"""
        question: str = Field(..., min_length=1, max_length=1000)
        context_project: Optional[str] = None
        context_author: Optional[str] = None
        context_days: int = Field(30, ge=1, le=365)
    
    class ChatResponseDto(BaseModel):
        """DTO for chat responses"""
        answer: str
        context_commits_count: int
        processing_time_ms: int
        model_used: str
    ```
    
    ### src/application/dto/webhook_dto.py
    ```python
    from pydantic import BaseModel
    from typing import List, Optional
    from datetime import datetime
    
    class WebhookAuthorDto(BaseModel):
        """DTO for webhook author data"""
        name: str
        email: str
        username: Optional[str] = None
    
    class WebhookCommitDto(BaseModel):
        """DTO for webhook commit data"""
        id: str
        message: str
        timestamp: datetime
        author: WebhookAuthorDto
        added: List[str] = []
        modified: List[str] = []
        removed: List[str] = []
        url: Optional[str] = None
    
    class WebhookRepositoryDto(BaseModel):
        """DTO for webhook repository data"""
        name: str
        full_name: str
        url: str
        default_branch: str
    
    class WebhookPayloadDto(BaseModel):
        """DTO for webhook payload"""
        ref: str  # refs/heads/main
        repository: WebhookRepositoryDto
        commits: List[WebhookCommitDto]
        pusher: WebhookAuthorDto
        head_commit: Optional[WebhookCommitDto] = None
    ```
    
    ### src/application/use_cases/process_commit.py
    ```python
    from typing import Optional
    from datetime import datetime
    from ..dto.commit_dto import CommitDto, CommitResponseDto
    from ..services.event_dispatcher import IEventDispatcher
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.entities.commit import Commit, CommitHash, FileChange
    from ...domain.events.commit_events import CommitReceivedEvent
    from ...application.interfaces.cache_interface import ICacheService
    from ...application.interfaces.messaging_interface import IMessageQueue
    
    class ProcessCommitUseCase:
        """Use case for processing incoming commits"""
        
        def __init__(
            self,
            commit_repository: ICommitRepository,
            event_dispatcher: IEventDispatcher,
            message_queue: IMessageQueue,
            cache_service: ICacheService
        ):
            self.commit_repository = commit_repository
            self.event_dispatcher = event_dispatcher
            self.message_queue = message_queue
            self.cache_service = cache_service
        
        async def execute(self, commit_dto: CommitDto) -> CommitResponseDto:
            """Process a new commit"""
            
            # Check if commit already exists (idempotency)
            existing = await self.commit_repository.find_by_hash(commit_dto.commit_hash)
            if existing:
                return self._to_response_dto(existing)
            
            # Create domain entity
            commit = self._to_domain(commit_dto)
            
            # Save to repository
            saved_commit = await self.commit_repository.save(commit)
            
            # Dispatch domain event
            event = CommitReceivedEvent(
                commit_id=saved_commit.id,
                commit_hash=saved_commit.commit_hash.value,
                project=saved_commit.project,
                author=saved_commit.author_email,
                branch=saved_commit.branch
            )
            await self.event_dispatcher.dispatch(event)
            
            # Queue for async analysis
            await self.message_queue.enqueue(
                "analyze_commit",
                {"commit_id": saved_commit.id}
            )
            
            # Invalidate relevant caches
            await self.cache_service.invalidate_pattern(f"commits:project:{commit.project}:*")
            await self.cache_service.invalidate_pattern(f"commits:author:{commit.author_email}:*")
            
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
    ```
    
    ### src/application/use_cases/search_commits.py
    ```python
    from typing import List
    from ..dto.commit_dto import CommitSearchDto, CommitResponseDto
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...application.interfaces.cache_interface import ICacheService
    import hashlib
    import json
    
    class SearchCommitsUseCase:
        """Use case for searching commits"""
        
        def __init__(
            self,
            commit_repository: ICommitRepository,
            cache_service: ICacheService
        ):
            self.commit_repository = commit_repository
            self.cache_service = cache_service
        
        async def execute(self, search_dto: CommitSearchDto) -> List[CommitResponseDto]:
            """Search commits with caching"""
            
            # Generate cache key
            cache_key = self._generate_cache_key(search_dto)
            
            # Try cache first
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                return cached_result
            
            # Search in repository
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
                limit=search_dto.limit
            )
            
            # Convert to DTOs
            result = [self._to_response_dto(commit) for commit in commits]
            
            # Cache result for 5 minutes
            await self.cache_service.set(cache_key, result, ttl=300)
            
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
    ```
    
    ### src/application/use_cases/chat_with_ai.py
    ```python
    from typing import List
    from datetime import datetime, timedelta
    import time
    from ..dto.commit_dto import ChatRequestDto, ChatResponseDto
    from ...domain.repositories.commit_repository import ICommitRepository
    from ...domain.services.ai_analyzer import IAIAnalyzer
    from ...application.interfaces.cache_interface import ICacheService
    import hashlib
    
    class ChatWithAIUseCase:
        """Use case for chatting with AI about commits"""
        
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
            """Answer question about commits using AI"""
            
            start_time = time.time()
            
            # Generate cache key
            cache_key = self._generate_cache_key(chat_dto)
            
            # Try cache first
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                return cached_result
            
            # Get context commits
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=chat_dto.context_days)
            
            context_commits = await self.commit_repository.find_by_project(
                project=chat_dto.context_project,
                start_date=start_date,
                end_date=end_date,
                limit=100
            )
            
            if chat_dto.context_author:
                author_commits = await self.commit_repository.find_by_author(
                    author_email=chat_dto.context_author,
                    project=chat_dto.context_project,
                    start_date=start_date,
                    end_date=end_date,
                    limit=50
                )
                context_commits.extend(author_commits)
            
            # Remove duplicates
            unique_commits = {commit.id: commit for commit in context_commits}
            context_commits = list(unique_commits.values())
            
            # Ask AI
            answer = await self.ai_analyzer.answer_question(
                question=chat_dto.question,
                context_commits=context_commits
            )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            result = ChatResponseDto(
                answer=answer,
                context_commits_count=len(context_commits),
                processing_time_ms=processing_time_ms,
                model_used=self.ai_analyzer.get_model_info()['name']
            )
            
            # Cache result for 10 minutes
            await self.cache_service.set(cache_key, result, ttl=600)
            
            return result
        
        def _generate_cache_key(self, chat_dto: ChatRequestDto) -> str:
            """Generate cache key for chat request"""
            chat_hash = hashlib.md5(
                f"{chat_dto.question}:{chat_dto.context_project}:{chat_dto.context_author}:{chat_dto.context_days}".encode()
            ).hexdigest()
    ### src/application/services/event_dispatcher.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, List, Callable, Any
    from ...domain.events.base import DomainEvent
    import asyncio
    import logging
    
    logger = logging.getLogger(__name__)
    
    class IEventDispatcher(ABC):
        """Event dispatcher interface"""
        
        @abstractmethod
        async def dispatch(self, event: DomainEvent):
            """Dispatch event to all registered handlers"""
            pass
        
        @abstractmethod
        def register_handler(self, event_type: str, handler: Callable):
            """Register event handler"""
            pass
    
    class EventDispatcher(IEventDispatcher):
        """Implementation of event dispatcher"""
        
        def __init__(self):
            self.handlers: Dict[str, List[Callable]] = {}
        
        def register_handler(self, event_type: str, handler: Callable):
            """Register event handler"""
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            self.handlers[event_type].append(handler)
            logger.info(f"Registered handler for event type: {event_type}")
        
        async def dispatch(self, event: DomainEvent):
            """Dispatch event to all registered handlers"""
            event_type = event.__class__.__name__
            
            if event_type not in self.handlers:
                logger.debug(f"No handlers registered for event type: {event_type}")
                return
            
            logger.info(f"Dispatching event: {event_type} (ID: {event.event_id})")
            
            tasks = []
            for handler in self.handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        # Run sync handler in thread pool
                        loop = asyncio.get_event_loop()
                        tasks.append(loop.run_in_executor(None, handler, event))
                except Exception as e:
                    logger.error(f"Error preparing handler for {event_type}: {str(e)}")
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log any handler exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Handler {i} failed for {event_type}: {str(result)}")
    
    ### src/application/interfaces/cache_interface.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Any, Optional, List
    
    class ICacheService(ABC):
        """Interface for cache service"""
        
        @abstractmethod
        async def get(self, key: str) -> Optional[Any]:
            """Get value from cache"""
            pass
        
        @abstractmethod
        async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
            """Set value in cache with TTL in seconds"""
            pass
        
        @abstractmethod
        async def delete(self, key: str) -> bool:
            """Delete key from cache"""
            pass
        
        @abstractmethod
        async def exists(self, key: str) -> bool:
            """Check if key exists in cache"""
            pass
        
        @abstractmethod
        async def invalidate_pattern(self, pattern: str) -> int:
            """Invalidate all keys matching pattern (with * wildcard)"""
            pass
        
        @abstractmethod
        async def get_many(self, keys: List[str]) -> Dict[str, Any]:
            """Get multiple values from cache"""
            pass
        
        @abstractmethod
        async def set_many(self, data: Dict[str, Any], ttl: int = 3600) -> bool:
            """Set multiple values in cache"""
            pass
    
    ### src/application/interfaces/ai_interface.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, List, Optional
    from ...domain.entities.analysis import AnalysisResult
    
    class IAIService(ABC):
        """Interface for AI service"""
        
        @abstractmethod
        async def analyze_text(
            self,
            text: str,
            context: Optional[str] = None,
            temperature: float = 0.7
        ) -> AnalysisResult:
            """Analyze text and return structured results"""
            pass
        
        @abstractmethod
        async def generate_response(
            self,
            prompt: str,
            context: Optional[str] = None,
            max_tokens: int = 500
        ) -> str:
            """Generate text response"""
            pass
        
        @abstractmethod
        async def create_embedding(self, text: str) -> List[float]:
            """Create text embedding"""
            pass
        
        @abstractmethod
        async def health_check(self) -> bool:
            """Check if AI service is healthy"""
            pass
        
        @abstractmethod
        def get_model_info(self) -> Dict[str, str]:
            """Get model information"""
            pass
    
    ### src/application/interfaces/messaging_interface.py
    ```python
    from abc import ABC, abstractmethod
    from typing import Dict, Any, Optional, Callable
    from enum import Enum
    
    class TaskStatus(Enum):
        PENDING = "PENDING"
        STARTED = "STARTED"
        SUCCESS = "SUCCESS"
        FAILURE = "FAILURE"
        RETRY = "RETRY"
    
    class IMessageQueue(ABC):
        """Interface for message queue"""
        
        @abstractmethod
        async def enqueue(
            self,
            task_name: str,
            payload: Dict[str, Any],
            delay_seconds: int = 0,
            priority: int = 5
        ) -> str:
            """Enqueue task for processing"""
            pass
        
        @abstractmethod
        async def get_task_status(self, task_id: str) -> TaskStatus:
            """Get task status"""
            pass
        
        @abstractmethod
        async def get_task_result(self, task_id: str) -> Optional[Any]:
            """Get task result"""
            pass
        
        @abstractmethod
        async def cancel_task(self, task_id: str) -> bool:
            """Cancel pending task"""
            pass
        
        @abstractmethod
        def register_task(self, task_name: str, handler: Callable):
            """Register task handler"""
            pass
    
    ## 3. Infrastructure Layer
    
    ### src/infrastructure/persistence/database.py
    ```python
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy import MetaData
    from contextlib import asynccontextmanager
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    # SQLAlchemy base
    Base = declarative_base()
    metadata = MetaData()
    
    # Global engine and session maker
    engine = None
    SessionLocal = None
    
    def get_database_url() -> str:
        """Get database URL from environment"""
        db_provider = os.getenv("DB_PROVIDER", "sqlite")
        
        if db_provider == "sqlite":
            db_path = os.getenv("DB_PATH", "data/git_analytics.db")
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            return f"sqlite+aiosqlite:///{db_path}"
        
        elif db_provider == "postgresql":
            host = os.getenv("DB_HOST", "localhost")
            port = os.getenv("DB_PORT", "5432")
            name = os.getenv("DB_NAME", "git_analytics")
            user = os.getenv("DB_USER", "postgres")
            password = os.getenv("DB_PASSWORD", "")
            return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{name}"
        
        else:
            raise ValueError(f"Unsupported database provider: {db_provider}")
    
    async def init_database():
        """Initialize database connection"""
        global engine, SessionLocal
        
        if engine is None:
            database_url = get_database_url()
            logger.info(f"Connecting to database: {database_url.split('://')[0]}://...")
            
            engine = create_async_engine(
                database_url,
                echo=os.getenv("DB_ECHO", "false").lower() == "true",
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            SessionLocal = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database initialized successfully")
    
    async def close_database():
        """Close database connection"""
        global engine
        if engine:
            await engine.dispose()
            logger.info("Database connection closed")
    
    @asynccontextmanager
    async def get_session() -> AsyncSession:
        """Get database session"""
        if SessionLocal is None:
            await init_database()
        
        async with SessionLocal() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    ### src/infrastructure/persistence/models/commit_model.py
    ```python
    from sqlalchemy import Column, String, DateTime, Text, Integer, Float, JSON
    from sqlalchemy.sql import func
    from ..database import Base
    
    class CommitModel(Base):
        """SQLAlchemy model for commits"""
        
        __tablename__ = "commits"
        
        # Primary fields
        id = Column(String(36), primary_key=True)
        commit_hash = Column(String(40), unique=True, nullable=False, index=True)
        author_email = Column(String(255), nullable=False, index=True)
        author_name = Column(String(255), nullable=False)
        message = Column(Text, nullable=False)
        timestamp = Column(DateTime, nullable=False, index=True)
        branch = Column(String(255), nullable=False, index=True)
        project = Column(String(255), nullable=False, index=True)
        
        # File changes (stored as JSON)
        files_changed = Column(JSON, nullable=False, default=list)
        issue_numbers = Column(JSON, nullable=False, default=list)
        
        # Metrics
        total_lines_changed = Column(Integer, default=0)
        files_count = Column(Integer, default=0)
        complexity_score = Column(Float)
        impact_score = Column(Float)
        
        # Analysis results
        summary = Column(Text)
        tags = Column(JSON, default=list)
        sentiment_score = Column(Float)
        embedding_id = Column(String(255))
        analyzed_at = Column(DateTime)
        
        # Timestamps
        created_at = Column(DateTime, default=func.now(), nullable=False)
        updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
        
        def __repr__(self):
            return f"<CommitModel(hash={self.commit_hash}, project={self.project})>"
    
    class AnalysisModel(Base):
        """SQLAlchemy model for analysis results"""
        
        __tablename__ = "analyses"
        
        id = Column(String(36), primary_key=True)
        commit_id = Column(String(36), nullable=False, index=True)
        
        # Model information
        model_name = Column(String(100), nullable=False)
        model_version = Column(String(50), nullable=False)
        prompt_version = Column(String(50), nullable=False)
        
        # Analysis results
        summary = Column(Text, nullable=False)
        tags = Column(JSON, default=list)
        sentiment_score = Column(Float, nullable=False)
        confidence_score = Column(Float, nullable=False)
        extracted_entities = Column(JSON, default=list)
        
        # Performance metrics
        processing_time_ms = Column(Integer, nullable=False)
        tokens_used = Column(Integer, default=0)
        status = Column(String(20), default="completed")
        
        # Timestamps
        created_at = Column(DateTime, default=func.now(), nullable=False)
        
        def __repr__(self):
            return f"<AnalysisModel(commit_id={self.commit_id}, model={self.model_name})>"

Git AI Analytics - Complete Implementation

Code ∙ Version 8 

<div class="endpoint"> <span class="method">POST</span> /api/v1/webhooks/github - GitHub webhook handler </div> <div class="endpoint"> <span class="method">GET</span> /a

Tôi đã implement đầy đủ hệ thống Git AI Analytics theo clean architecture với tất cả các yêu cầu bạn đã đề ra. Đây là một hệ thống hoàn chỉnh bao gồm:

🏗️ **Kiến trúc Clean Architecture**
------------------------------------

*   **Domain Layer**: Entities, Value Objects, Repository Interfaces, Domain Events
*   **Application Layer**: Use Cases, DTOs, Application Services
*   **Infrastructure Layer**: Repository Implementations, AI Services, Cache, Messaging
*   **Presentation Layer**: FastAPI endpoints, WebSocket handlers

🚀 **Tính năng chính:**
-----------------------

### **Phase 1 - Learning Ready:**

*   ✅ SQLite database (dễ setup)
*   ✅ In-memory cache (không cần Redis)
*   ✅ Ollama AI integration (free, local)
*   ✅ Basic webhook handling
*   ✅ Simple commit analysis

### **Phase 2 - Production Ready:**

*   ✅ PostgreSQL migration path
*   ✅ Redis cache & messaging
*   ✅ Event-driven architecture
*   ✅ Celery async processing
*   ✅ WebSocket real-time updates
*   ✅ Multi-layer caching strategy

🛠️ **Tech Stack:**
-------------------

*   **Backend**: FastAPI (async, DI tốt)
*   **Database**: SQLite → PostgreSQL
*   **AI**: Ollama với Llama3.2:3b (free, 2GB RAM)
*   **Cache**: Memory → Redis
*   **Queue**: Celery + Redis
*   **Real-time**: WebSocket
*   **Config**: Environment-based với YAML

📊 **Flow hoạt động:**
----------------------

    Git Commit → Webhook → FastAPI → Event Dispatcher → Database
                                          ↓
                                      Celery Queue → AI Analysis → Update DB
                                          ↓
                                  WebSocket Broadcast → Real-time UI

🔧 **Khởi chạy nhanh:**
-----------------------

bash

    # 1. Setup cơ bản
    pip install -r requirements/development.txt
    cp .env.example .env
    
    # 2. Start services
    ollama serve && ollama pull llama3.2:3b
    redis-server  # optional
    python -m src.main
    
    # 3. Test
    curl http://localhost:8000/health

🎯 **Điểm mạnh thiết kế:**
--------------------------

### **Scalability:**

*   Interface-based design dễ thay đổi implementation
*   Event-driven cho loose coupling
*   Async processing với Celery
*   Multi-layer cache strategy
*   Horizontal scaling ready

### **Maintainability:**

*   Clean code với SOLID principles
*   Comprehensive logging
*   Error handling ở mọi layer
*   Type hints và validation
*   Comprehensive testing structure

### **Flexibility:**

*   Config-driven (dev/prod environments)
*   Provider-agnostic (SQLite↔PostgreSQL, Memory↔Redis)
*   Webhook support (GitHub/GitLab)
*   Multiple AI model support

Hệ thống này sẵn sàng cho phase 1 (học tập) và có thể scale lên phase 2 (20 devs) mà không cần refactor architecture cốt lõi. Bạn có muốn tôi giải thích chi tiết về phần nào không?

Retry

---

