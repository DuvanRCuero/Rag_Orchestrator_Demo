import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, patch

# Environment variable overrides for tests - MUST be before any imports from src
os.environ["ENVIRONMENT"] = "testing"
os.environ["DEBUG"] = "True"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["REDIS_PASSWORD"] = "test-password"

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from src.api.app import create_application
from src.application.chains.memory import EnhancedConversationMemory
from src.application.chains.rag_chain import AdvancedRAGChain
from src.core.config import Settings
from src.core.schemas import DocumentChunk, DocumentType
from src.domain.documents import AdvancedDocumentProcessor
from src.domain.embeddings import EmbeddingService
from src.infrastructure.vector.qdrant_client import QdrantVectorStore


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the tests session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Test settings with overrides."""
    return Settings(
        ENVIRONMENT="testing",
        DEBUG=True,
        VECTOR_DB_TYPE="chroma",  # Use Chroma for tests (in-memory)
        QDRANT_URL="http://localhost:6333",
        EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
        EMBEDDING_DEVICE="cpu",
        LLM_PROVIDER="openai",
        OPENAI_MODEL="gpt-3.5-turbo",
        OPENAI_TEMPERATURE=0.0,
        CHUNK_SIZE=500,
        CHUNK_OVERLAP=100,
        RETRIEVAL_TOP_K=3,
        USE_HYBRID_SEARCH=False,
        MEMORY_TYPE="buffer",
    )


@pytest.fixture
def sample_documents() -> Dict[str, str]:
    """Sample documents for testing."""
    return {
        "langchain_production.md": """
# LangChain Production Deployment Guide

## Optimizing Performance

### Chunking Strategies
When deploying LangChain in production, chunking strategy is critical. 
Use recursive character splitting with overlap for code documentation.
For general text, semantic splitting works better.

### Embedding Models
For production, use sentence-transformers/all-mpnet-base-v2 for better quality.
For speed, use all-MiniLM-L6-v2.

### Vector Databases
Qdrant is recommended for production due to its performance and scalability.
ChromaDB is good for development and testing.

## Cost Optimization

### Caching
Implement Redis caching for embeddings and frequent queries.
This reduces API calls and improves response time.

### Token Management
Use tiktoken to count tokens and manage context window.
Always set max_tokens to prevent unexpected costs.

## Monitoring & Logging

### Metrics to Track
- Retrieval precision and recall
- Generation latency
- Token usage per request
- Error rates

### Tools
Use Prometheus and Grafana for monitoring.
Implement structured logging with JSON format.
""",
        "rag_best_practices.md": """
# RAG Best Practices

## Hallucination Mitigation

### Techniques
1. Chain-of-Verification: Verify each claim against context
2. Self-Reflection: Ask LLM to critique its own answer
3. Confidence Scoring: Quantify answer reliability

### Retrieval Optimization
- Use multi-query generation
- Implement HyDE (Hypothetical Document Embeddings)
- Apply cross-encoder reranking

## Scaling Strategies

### Horizontal Scaling
Deploy multiple RAG instances behind a load balancer.
Use shared vector database (Qdrant cluster).

### Caching Layers
1. Embedding cache (Redis)
2. Query result cache
3. Session cache for conversations

## Security Considerations

### API Security
Use API keys and rate limiting.
Implement request validation and sanitization.

### Data Privacy
Ensure PII filtering in documents.
Use on-premise models for sensitive data.
""",
    }


@pytest.fixture
def sample_questions() -> Dict[str, Dict[str, Any]]:
    """Sample questions and expected answer patterns for testing."""
    return {
        "optimization": {
            "query": "How do I optimize LangChain for production?",
            "expected_keywords": [
                "chunking",
                "caching",
                "monitoring",
                "Qdrant",
                "Redis",
            ],
            "expected_sources": 3,
        },
        "rag_best_practices": {
            "query": "What are RAG best practices?",
            "expected_keywords": ["hallucination", "retrieval", "scaling", "security"],
            "expected_sources": 2,
        },
        "cost_optimization": {
            "query": "How to reduce costs in RAG systems?",
            "expected_keywords": ["caching", "tokens", "monitoring", "Redis"],
            "expected_sources": 1,
        },
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI response for testing."""
    return """
Based on the provided context:

1. For production optimization, use Qdrant as vector database and Redis for caching.
2. Implement monitoring with Prometheus and Grafana.
3. Use sentence-transformers for embeddings and tiktoken for token management.
4. Apply chunking strategies based on document type.
"""


@pytest.fixture
def test_client(test_settings) -> TestClient:
    """Test client for FastAPI application."""
    with patch("src.core.config.settings", test_settings):
        app = create_application()
        with TestClient(app) as client:
            yield client


@pytest.fixture
def temp_document_files(sample_documents) -> Generator:
    """Create temporary document files for testing."""
    temp_files = []
    temp_dir = tempfile.mkdtemp()

    for filename, content in sample_documents.items():
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        temp_files.append(filepath)

    yield temp_files

    # Cleanup
    for filepath in temp_files:
        if os.path.exists(filepath):
            os.unlink(filepath)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)


@pytest_asyncio.fixture
async def mock_embedding_service() -> AsyncMock:
    """Mock embedding service for testing."""
    mock = AsyncMock(spec=EmbeddingService)
    mock.embed_texts.return_value = [
        [0.1] * 384,  # Mock embedding for chunk 1
        [0.2] * 384,  # Mock embedding for chunk 2
        [0.3] * 384,  # Mock embedding for chunk 3
    ]
    mock.embed_query.return_value = [0.15] * 384
    mock.embedding_dimension = 384
    return mock


@pytest_asyncio.fixture
async def mock_vector_store() -> AsyncMock:
    """Mock vector store for testing."""
    mock = AsyncMock(spec=QdrantVectorStore)

    # Mock chunks
    mock_chunks = [
        DocumentChunk(
            id="chunk_1",
            content="LangChain production deployment requires Qdrant and Redis.",
            metadata={"source": "langchain_production.md", "page": 1},
            document_id="doc_1",
            chunk_index=0,
        ),
        DocumentChunk(
            id="chunk_2",
            content="Use sentence-transformers for embeddings and implement caching.",
            metadata={"source": "langchain_production.md", "page": 2},
            document_id="doc_1",
            chunk_index=1,
        ),
    ]

    mock.search.return_value = mock_chunks[:2]
    mock.hybrid_search.return_value = mock_chunks
    mock.get_collection_stats.return_value = {
        "vectors_count": 100,
        "segments_count": 2,
        "config": {"vector_size": 384},
    }

    return mock


@pytest_asyncio.fixture
async def mock_llm_service() -> AsyncMock:
    """Mock LLM service for testing."""
    mock = AsyncMock()
    mock.generate.return_value = "LangChain should be optimized using Qdrant and Redis for production deployment."
    mock.generate_json.return_value = {
        "score": 85,
        "breakdown": {},
        "explanation": "Good answer",
    }
    mock.get_token_usage.return_value = {
        "total_tokens": 150,
        "prompt_tokens": 100,
        "completion_tokens": 50,
    }

    # Mock langchain_llm for LCEL chains
    mock_langchain_llm = MagicMock()
    mock_langchain_llm.ainvoke = AsyncMock(
        return_value="LangChain should be optimized using Qdrant and Redis for production deployment."
    )
    mock.langchain_llm = mock_langchain_llm

    # Mock streaming
    async def mock_stream(messages=None):
        """Mock streaming generation. Ignores messages parameter as this is a simple mock."""
        words = [
            "LangChain",
            " optimization",
            " requires",
            " Qdrant",
            " and",
            " Redis.",
        ]
        for word in words:
            yield word
            await asyncio.sleep(0.01)

    mock.stream_generation = mock_stream
    return mock


@pytest.fixture
def mock_document_processor() -> MagicMock:
    """Mock document processor for testing."""
    mock = MagicMock(spec=AdvancedDocumentProcessor)

    mock_chunks = [
        DocumentChunk(
            id=f"chunk_{i}",
            content=f"Test chunk {i} content",
            metadata={"source": "tests.md", "chunk_index": i},
            document_id="test_doc",
            chunk_index=i,
        )
        for i in range(3)
    ]

    mock.create_intelligent_chunks.return_value = mock_chunks
    mock.load_document.return_value = [
        {"content": "Test document content", "metadata": {}, "source": "tests.md"}
    ]

    return mock


@pytest.fixture
def mock_conversation_memory() -> MagicMock:
    """Mock conversation memory for testing."""
    mock = MagicMock(spec=EnhancedConversationMemory)
    mock.get_session.return_value.turns = []
    mock.get_conversation_for_context.return_value = ""
    return mock


@pytest_asyncio.fixture
async def test_rag_chain(
    mock_embedding_service,
    mock_vector_store,
    mock_llm_service,
    mock_conversation_memory,
) -> AdvancedRAGChain:
    """Test RAG chain with mocked dependencies."""
    with patch(
        "src.application.chains.rag_chain.EmbeddingService",
        return_value=mock_embedding_service,
    ), patch(
        "src.application.chains.rag_chain.QdrantVectorStore",
        return_value=mock_vector_store,
    ), patch(
        "src.application.chains.rag_chain.llm_service", mock_llm_service
    ), patch(
        "src.application.chains.rag_chain.conversation_memory", mock_conversation_memory
    ):
        chain = AdvancedRAGChain()
        
        # Directly mock the rag_chain_lcel's ainvoke to return a string
        chain.rag_chain_lcel = AsyncMock()
        chain.rag_chain_lcel.ainvoke = AsyncMock(
            return_value="LangChain should be optimized using Qdrant and Redis for production deployment."
        )
        
        yield chain


class AsyncContextManagerMock:
    """Helper for mocking async context managers."""

    def __init__(self, return_value=None):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch("openai.AsyncOpenAI") as mock:
        client_mock = AsyncMock()
        mock.return_value = client_mock

        # Mock embeddings
        embeddings_mock = AsyncMock()
        embeddings_mock.create.return_value = type(
            "obj",
            (object,),
            {
                "data": [
                    type("embedding", (object,), {"embedding": [0.1] * 1536}),
                    type("embedding", (object,), {"embedding": [0.2] * 1536}),
                ]
            },
        )
        client_mock.embeddings = embeddings_mock

        # Mock chat completions
        completions_mock = AsyncMock()
        completions_mock.create.return_value = type(
            "obj",
            (object,),
            {
                "choices": [
                    type(
                        "choice",
                        (object,),
                        {
                            "message": type(
                                "message", (object,), {"content": "Test response"}
                            )
                        },
                    )
                ]
            },
        )
        client_mock.chat.completions = completions_mock

        yield client_mock


@pytest.fixture
def test_data_dir():
    """Create a temporary directory for tests data."""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Cleanup
    import shutil

    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture
def expected_metrics():
    """Expected metrics for testing."""
    return {
        "retrieval": ["precision@k", "recall@k", "ndcg@k"],
        "generation": ["answer_relevance", "faithfulness", "coherence"],
        "system": ["latency", "throughput", "error_rate"],
    }
