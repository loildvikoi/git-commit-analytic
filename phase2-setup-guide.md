# Phase 2 Setup Guide - RAG System Implementation

## 📋 Prerequisites

Ensure Phase 1 is working correctly:
- FastAPI server running
- Ollama installed and running
- Database (MySQL/SQLite) configured
- Redis (optional but recommended)

## 🚀 Installation Steps

### 1. Install New Dependencies

```bash
# Create/update requirements file
cat >> requirements/phase2.txt << EOF
# Vector Database & Embeddings
chromadb>=0.4.0
sentence-transformers>=2.2.0

# Search & RAG
rank-bm25>=0.2.2
langchain>=0.1.0
langchain-community>=0.1.0

# ML Dependencies
torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Optional but recommended
faiss-cpu>=1.7.4
EOF

# Install dependencies
pip install -r requirements/phase2.txt
```

### 2. Download Embedding Model

The embedding model will be downloaded automatically on first use, but you can pre-download:

```python
from sentence_transformers import SentenceTransformer

# Pre-download model
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model downloaded successfully!")
```

### 3. Run Database Migration

```bash
# Run migration script
python src/migrations/add_documents_table.py
```

### 4. Update Environment Variables

Add to your `.env` file:

```env
# Phase 2 Settings
EMBEDDING_MODEL="all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR="./data/chroma"
MAX_CHUNK_SIZE="500"
CHUNK_OVERLAP="50"
HYBRID_SEARCH_SEMANTIC_WEIGHT="0.7"
RAG_MAX_CONTEXT_LENGTH="2000"
RAG_TEMPERATURE="0.3"
```

### 5. Create Required Directories

```bash
# Create directories for Phase 2
mkdir -p data/chroma
mkdir -p models
mkdir -p logs
```

## 🏃 Running Phase 2

### 1. Start the Server

```bash
# Start with Phase 2 features
python src/main.py

# Or with uvicorn
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Verify Installation

```bash
# Check health
curl http://localhost:8000/api/v1/rag/health

# Should return:
{
  "status": "healthy",
  "components": {
    "vector_store": "healthy",
    "embedding_service": "healthy",
    "search_service": "healthy",
    "rag_service": "healthy"
  }
}
```

### 3. Sync Existing Data

```bash
# Sync existing commits to document store
curl -X POST http://localhost:8000/api/v1/documents/sync-commits \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 100,
    "skip_existing": true
  }'
```

## 🧪 Testing Phase 2

### Run Test Suite

```bash
# Run full test
python test/test_phase2_rag.py

# Run specific tests
python test/test_phase2_rag.py index   # Test document indexing
python test/test_phase2_rag.py search  # Test hybrid search
python test/test_phase2_rag.py chat    # Test RAG chat
python test/test_phase2_rag.py perf    # Test performance
```

## 📊 Performance Tuning

### 1. Embedding Model Selection

Choose based on your needs:

| Model | Speed | Quality | Dimensions | Use Case |
|-------|-------|---------|------------|----------|
| all-MiniLM-L6-v2 | Fast | Good | 384 | General use |
| all-mpnet-base-v2 | Medium | Best | 768 | High accuracy |
| all-MiniLM-L12-v2 | Medium | Good | 384 | Balanced |

### 2. ChromaDB Configuration

```python
# For better performance with large datasets
Settings(
    chroma_db_impl="duckdb+parquet",  # Use for >100k documents
    persist_directory="./data/chroma",
    anonymized_telemetry=False
)
```

### 3. Caching Strategy

- **Search Cache**: 5 minutes (300s)
- **RAG Chat Cache**: 10 minutes (600s)
- **Embeddings**: Permanent until document update

## 🐛 Troubleshooting

### Issue: "No module named 'chromadb'"
```bash
pip install chromadb --upgrade
```

### Issue: "Embedding model download fails"
```bash
# Manual download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Issue: "ChromaDB collection already exists"
```python
# Reset ChromaDB
import chromadb
client = chromadb.PersistentClient(path="./data/chroma")
client.delete_collection("git_analytics_docs")
```

### Issue: "Out of memory during embedding"
```python
# Reduce batch size in sentence_transformer_service.py
embeddings = self.model.encode(
    texts,
    batch_size=8,  # Reduce from 32
    convert_to_tensor=False
)
```

## 📈 Monitoring & Metrics

### Key Metrics to Track

1. **Search Performance**
   - P95 latency < 2s
   - Cache hit rate > 60%
   - Embedding generation < 100ms

2. **RAG Quality**
   - Answer confidence > 0.7
   - Sources per answer: 3-5
   - User satisfaction > 4/5

3. **System Health**
   - Vector store document count
   - Memory usage < 2GB
   - CPU usage < 70%

### Logging

Check logs for performance:
```bash
tail -f logs/app.log | grep -E "RAG|Search|Embedding"
```

## 🚢 Production Deployment

### 1. Use Production Models

```python
# In production, consider using:
EMBEDDING_MODEL = "all-mpnet-base-v2"  # Better quality
OLLAMA_MODEL = "llama2:13b"  # Larger model
```

### 2. Scale ChromaDB

```python
# For production with >100k documents
client = chromadb.HttpClient(
    host="chroma-server",
    port=8000
)
```

### 3. Add Monitoring

```python
# Add to your code
from prometheus_client import Counter, Histogram

search_requests = Counter('rag_search_requests_total', 'Total search requests')
search_latency = Histogram('rag_search_latency_seconds', 'Search latency')
```

## 🎯 Next Steps

1. **Fine-tune embeddings** for your domain
2. **Implement feedback loop** to improve RAG
3. **Add more document types** (PRs, issues, docs)
4. **Implement cross-lingual search**
5. **Add conversation memory** for chat context

## 📚 Resources

- [ChromaDB Docs](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [BM25 Algorithm](https://github.com/dorianbrown/rank_bm25)

---

**Support**: If you encounter issues, check the logs first, then refer to the troubleshooting section above.