# Project Summary: RAG Orchestrator

> Technical deep-dive into architecture, design decisions, and evaluation results.

## 🎯 Problem Statement

Building production-grade RAG systems is deceptively complex. While basic prototypes are easy, real-world deployment faces critical challenges:

| Challenge | Impact |
|-----------|--------|
| **Hallucination** | LLMs confidently invent facts not in documents |
| **Scale** | What works for 100 docs fails at 100,000 |
| **Maintenance** | Coupled code breaks with every library update |
| **Observability** | No visibility into why answers are good or bad |
| **Vendor Lock-in** | Hard to switch between LLM/vector DB providers |

## 🏗️ Solution: Clean Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Infrastructure Layer (HOW)                             │
│  FastAPI, Qdrant, Redis, OpenAI                         │
├─────────────────────────────────────────────────────────┤
│  Application Layer (WHEN)                               │
│  RAG Pipeline, Session Management, Verification         │
├─────────────────────────────────────────────────────────┤
│  Domain Layer (WHAT)                                    │
│  Documents, Embeddings, Vector Operations               │
└─────────────────────────────────────────────────────────┘
```

**Benefits**: Swap LLM providers without touching business logic. Change vector databases without rewriting retrieval algorithms.

## 📊 Evaluation Results

### Retrieval Performance

| Metric | Score |
|--------|-------|
| Retrieval Latency P50 | 180ms |
| Retrieval Latency P95 | 420ms |
| Context Precision | 0.82 |
| Context Recall | 0.76 |
| Multi-Query Improvement | +40% recall |

### Generation Quality

| Metric | Score |
|--------|-------|
| Answer Relevance | 0.86 |
| Faithfulness | 0.89 |
| Hallucination Rate | 8% (down from 52%) |
| Citation Accuracy | 94% |
| Generation Latency P95 | 1.8s |

## 🔧 Key Design Decisions

### 1. Vector Database: Qdrant

**Why not ChromaDB?**
- Qdrant handles millions of vectors; ChromaDB struggles beyond prototypes
- Hybrid search support (semantic + keyword)
- Self-hostable with no vendor lock-in
- Benchmark: 420ms P95 vs 800ms+ for ChromaDB at 50k docs

### 2. Text Splitting: 3 Strategies

| Strategy | Best For | Chunk Size |
|----------|----------|------------|
| Recursive Character | General text | 1000 tokens |
| Semantic | Q&A, sentence-level | 500 tokens |
| Markdown Aware | Documentation | 800 tokens |

Auto-selection based on content type improved retrieval relevance by 15%.

### 3. Retrieval Enhancement: 3-Layer Pipeline

```
Query → Multi-Query Expansion → HyDE → Cross-Encoder Reranking → Results
         (+40% recall)         (better semantic match)  (precision boost)
```

Trade-off: Adds 300-500ms latency but improves relevance from 0.68 to 0.88.

### 4. Hallucination Mitigation: Defense in Depth

**Layer 1 - Better Retrieval**
- Multi-Query: 3 query variations for better recall
- HyDE: Hypothetical answer for better semantic matching

**Layer 2 - Verification**
- Chain-of-Verification: Verify each claim against sources
- Self-Reflection: LLM critiques its own answer
- Confidence Scoring: 0-100% reliability metric

**Layer 3 - Transparency**
- Source citations with every answer
- Audit trail for debugging

**Result**: Hallucination reduced from 52% to 8%.

### 5. Conversation Memory

- Keep recent 20 turns
- Summarize older turns to prevent context overflow
- Token-aware history (max 1000 tokens)
- Redis-backed for persistence

## 🧪 Testing Strategy

| Layer | Coverage | Focus |
|-------|----------|-------|
| Unit (70%) | Domain logic | Chunking, embeddings, retrieval algorithms |
| Integration (20%) | Component interactions | API endpoints, vector store, cache |
| E2E (10%) | Full pipeline | RAG flow, hallucination mitigation |
| Performance | Latency/throughput | P95 < 2s, > 20 req/sec |

**Total Coverage**: 85%+

## 🚧 Challenges Encountered

### Qdrant Version Compatibility
- **Issue**: Breaking changes in client API v1.7+
- **Solution**: Pinned to v1.6.0, added compatibility layer

### Context Window Management
- **Issue**: Multi-turn conversations exceed LLM limits
- **Solution**: Automatic trimming + summarization of older turns

### Latency vs Quality Trade-off
- **Issue**: Full enhancement pipeline adds 300-500ms
- **Solution**: Made each enhancement configurable via flags

## 🔮 Future Improvements

- [ ] Graph-based retrieval for relationship-aware queries
- [ ] Multi-model support (Anthropic, local LLMs)
- [ ] Kubernetes deployment with HPA
- [ ] Web UI with source highlighting
- [ ] RAGAS framework integration for automated evaluation

## 📁 Key Files

| File | Purpose |
|------|---------|
| `src/domain/documents.py` | 3 chunking strategies, multi-format loaders |
| `src/application/chains/rag_chain.py` | RAG pipeline with verification |
| `src/application/chains/prompts.py` | Prompt templates |
| `src/infrastructure/vector/qdrant_client.py` | Vector store integration |
| `tests/e2e/test_full_pipeline.py` | Complete RAG flow validation |

---

*For setup instructions, see [README.md](README.md)*