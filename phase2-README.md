# Phase 2 Setup Guide - RAG System Implementation

## 📋 Prerequisites

Ensure Phase 1 is working correctly:
- FastAPI server running
- Ollama installed and running
- Database (MySQL/SQLite) configured
- Redis (optional but recommended)

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

## 📚 Resources

- [ChromaDB Docs](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [BM25 Algorithm](https://github.com/dorianbrown/rank_bm25)
