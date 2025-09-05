# Git Analytics API - Hệ thống phân tích commit thông minh

## 🎯 Tổng quan dự án

Đây là một **hệ thống phân tích Git Analytics** được thiết kế để thu thập, phân tích và cung cấp thông tin chi tiết về các commit trong repository Git thông qua AI. Hệ thống nhận webhook từ GitHub/GitLab, phân tích commit bằng AI (Ollama), và cung cấp API để truy vấn dữ liệu.
Ý tưởng là xây dựng một hệ thống có thể xử lý các commit, phân tích chúng và cung cấp các thông tin hữu ích cho người dùng thông qua API, nhất là các nhà phát triển phần mềm.
Phase 1 sẽ tập trung vào việc thu thập và phân tích commit, tương tác thông qua API Chatbot với AI để trả lời các câu hỏi liên quan đến commit.

## 🏗️ Kiến trúc hệ thống

Dự án được xây dựng theo **Clean Architecture** với các layer rõ ràng:

### 1. Domain Layer (Lõi nghiệp vụ)
- **Entities**: `Commit`, `Analysis`, `CommitMetrics`, `FileChange`
- **Value Objects**: `CommitHash`, `AnalysisResult`
- **Repositories**: Interface cho `ICommitRepository`
- **Services**: Interface cho `IAIAnalyzer`, `ICacheService`, `IQueueService`
- **Events**: Hệ thống domain events với `DomainEvent` base class

### 2. Application Layer (Use Cases)
- **ProcessCommitUseCase**: Xử lý commit từ webhook
- **SearchCommitsUseCase**: Tìm kiếm và lọc commit
- **ChatWithAIUseCase**: Chat với AI về các commit

### 3. Infrastructure Layer (Chi tiết kỹ thuật)
- **Database**: SQLite với SQLAlchemy async
- **AI Service**: Tích hợp Ollama cho phân tích commit
- **Cache**: Redis hoặc Memory cache
- **Queue**: Celery cho xử lý bất đồng bộ
- **Events**: Redis event bus cho distributed events

### 4. Interface Layer (Giao diện)
- **REST API**: FastAPI với các endpoint
- **WebSocket**: Hỗ trợ real-time communication
- **DTOs**: Validation với Pydantic

## 🚀 Chức năng chính

### 1. Thu thập dữ liệu commit
- **Webhook endpoints**: Nhận webhook từ GitHub/GitLab
- **Tự động xử lý**: Parse commit data và metadata
- **Validation**: Kiểm tra tính hợp lệ của dữ liệu đầu vào

### 2. Phân tích commit bằng AI
- **Tóm tắt commit**: Tạo summary từ commit message và changes
- **Phân loại**: Gắn tags tự động (feature, bugfix, hotfix, etc.)
- **Sentiment analysis**: Đánh giá tính tích cực/tiêu cực
- **Entity extraction**: Trích xuất entities từ commit

### 3. Tính toán metrics
- **Complexity Score**: Đánh giá độ phức tạp dựa trên:
  - Số file thay đổi
  - Số dòng code thay đổi
  - Loại file (config, core files)
- **Impact Score**: Đánh giá tác động:
  - Core files affected
  - Configuration changes
  - Deletions (refactoring indicators)

### 4. API truy vấn dữ liệu
- **Search commits**: Tìm kiếm với filters
- **Get statistics**: Thống kê commit theo project/author
- **Chat with AI**: Đặt câu hỏi về commits

## 🛠️ Tech Stack

### Backend Framework
- **FastAPI**: REST API framework
- **SQLAlchemy**: ORM với async support
- **Pydantic**: Data validation

### Database & Cache
- **SQLite**: Primary database
- **Redis**: Caching và event bus

### AI & ML
- **Ollama**: Local LLM for commit analysis
- **Custom prompts**: Specialized for code analysis

### Background Processing
- **Celery**: Task queue
- **Redis**: Message broker

### DevOps
- **Docker**: Containerization
- **docker-compose**: Multi-service orchestration

## 📁 Cấu trúc thư mục

```
src/
├── domain/           # Business logic core
│   ├── entities/     # Domain entities
│   ├── repositories/ # Repository interfaces
│   └── services/     # Service interfaces
├── application/      # Use cases
│   ├── use_cases/    # Business workflows
│   └── events/       # Application events
├── infrastructure/   # Technical details
│   ├── persistence/  # Database implementation
│   ├── ai/          # AI service implementation
│   ├── cache/       # Cache implementations
│   └── queue/       # Queue implementations
└── interface/        # External interfaces
    ├── api/         # REST endpoints
    ├── dto/         # Data transfer objects
    └── websocket/   # WebSocket handlers
```

## 🔄 Workflow chính

### 1. Receive Webhook
```
GitHub/GitLab → Webhook → FastAPI → ProcessCommitUseCase
```

### 2. Process Commit
```
Parse Data → Validate → Save to DB → Trigger Analysis
```

### 3. AI Analysis (Background)
```
Queue Task → Ollama Analysis → Update DB → Cache Results
```

### 4. Query & Search
```
API Request → Search UseCase → Cache Check → DB Query → Response
```

## 🎯 Use Cases chính

### Developer Analytics
- Phân tích coding patterns của developers
- Đánh giá quality và impact của commits
- Tracking productivity metrics

