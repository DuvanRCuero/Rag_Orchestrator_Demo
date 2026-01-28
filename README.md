# RAG Orchestrator

> Production-Grade LangChain RAG System with Hallucination Mitigation

A question-answering system built with Clean Architecture principles, featuring advanced retrieval techniques, hallucination mitigation, and production-ready infrastructure.

## ✨ Key Features

| Feature | Implementation |
|---------|----------------|
| **Multi-Strategy Retrieval** | Multi-Query Expansion, HyDE, Cross-Encoder Reranking |
| **Hallucination Mitigation** | Chain-of-Verification, Self-Reflection, Confidence Scoring |
| **Clean Architecture** | 3-layer separation (Domain/Application/Infrastructure) |
| **Production Ready** | Docker Compose, async everywhere, 85%+ test coverage |

## 📊 Results

- **Hallucination Rate**: Reduced from 52% → 8%
- **Retrieval Precision**: 0.82
- **Faithfulness Score**: 0.89
- **P95 Latency**: < 2s

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- OpenAI API Key ([Get one here](https://platform.openai.com/api-keys))

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/DuvanRCuero/RAG-Orchestrator
cd RAG-Orchestrator

# Create environment file
cp .env.example .env

# Add your OpenAI API key to .env
# OPENAI_API_KEY=sk-your-key-here

# Start all services (API + Qdrant + Redis)
make deploy
# Or: docker-compose up -d

# Verify services are running
make health

# Test with a query
curl -X POST "http://localhost:8000/api/v1/query/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "What vector database should I use for production?"}'
```

### Option 2: Local Development

```bash
# Clone and enter directory
git clone https://github.com/DuvanRCuero/RAG-Orchestrator
cd RAG-Orchestrator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Create environment file and add your API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-your-key-here

# Start infrastructure services only (Qdrant + Redis)
docker-compose up -d qdrant redis

# Run the API locally
uvicorn src.api.app:app --reload --port 8000

# In another terminal, ingest sample documents
python scripts/ingest_sample_data.py
```

### Verify Installation

```bash
# Health check
curl http://localhost:8000/api/v1/monitoring/health

# Interactive API docs
open http://localhost:8000/docs
```

## 📖 Usage

### Ask a Question

```bash
curl -X POST "http://localhost:8000/api/v1/query/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How can I reduce hallucinations in my RAG system?",
    "session_id": "user-123"
  }'
```

### Stream Response

```bash
curl -X POST "http://localhost:8000/api/v1/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain multi-query retrieval"}'
```

### Upload Custom Documents

```bash
curl -X POST "http://localhost:8000/api/v1/ingest/upload" \
  -F "file=@your-document.pdf"
```

## 🧪 Running Tests

```bash
# Run all tests
make test

# Run specific test suites
pytest tests/unit/ -v          # Unit tests
pytest tests/integration/ -v   # Integration tests
pytest tests/e2e/ -v           # End-to-end tests

# With coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

## 🏗️ Project Structure

```
RAG-Orchestrator/
├── src/
│   ├── domain/          # Business logic (documents, embeddings)
│   ├── application/     # RAG pipeline, memory, prompts
│   ├── infrastructure/  # Qdrant, OpenAI, Redis clients
│   └── api/             # FastAPI endpoints
├── tests/               # Unit, integration, e2e tests
├── data/raw/            # Sample documents
└── docker-compose.yml   # One-command deployment
```

## 🔧 Tech Stack

- **Framework**: LangChain + FastAPI
- **Vector Store**: Qdrant
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: GPT-4 / GPT-3.5-turbo
- **Cache**: Redis
- **Testing**: Pytest (85%+ coverage)

## 📚 Documentation

- [Project Summary](PROJECT_SUMMARY.md) - Technical deep dive & design decisions
- [Architecture Guide](docs/architecture.md)
- [Hallucination Mitigation](docs/hallucination.md)
- [Performance Optimization](docs/performance.md)

## 🛠️ Common Commands

```bash
make help          # Show all commands
make up            # Start services
make down          # Stop services
make logs          # View logs
make clean         # Remove containers and volumes
make health        # Check service health
```

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

*Built for the CODEBRANCH 2025 LangChain Challenge*