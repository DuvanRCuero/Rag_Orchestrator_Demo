"""Test configuration helpers."""

from contextlib import contextmanager

from src.core.config_models import (
    AppConfig,
    LLMConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    RetrievalConfig,
    ChunkingConfig,
    MemoryConfig,
    get_config,
    set_config,
)


def create_test_config(**overrides) -> AppConfig:
    """Create a test configuration with sensible defaults."""
    defaults = {
        "llm": LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=500,
            api_key="test-key",
        ),
        "embedding": EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
            device="cpu",
        ),
        "vector_store": VectorStoreConfig(
            db_type="qdrant",
            url="http://localhost:6333",
            collection_name="test_collection",
        ),
        "retrieval": RetrievalConfig(
            top_k=3,
            score_threshold=0.5,
            use_hybrid_search=False,
            bm25_weight=0.3,
            semantic_weight=0.7,
        ),
        "chunking": ChunkingConfig(
            chunk_size=500,
            chunk_overlap=100,
            strategy="recursive_character",
        ),
        "memory": MemoryConfig(
            memory_type="buffer",
            window_size=5,
        ),
        "environment": "testing",
        "debug": True,
    }
    
    defaults.update(overrides)
    return AppConfig(**defaults)


@contextmanager
def with_test_config(config: AppConfig = None):
    """Context manager for using test configuration."""
    original = get_config()
    try:
        set_config(config or create_test_config())
        yield
    finally:
        set_config(original)
