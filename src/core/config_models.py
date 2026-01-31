"""Immutable configuration models for dependency injection."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMConfig:
    """LLM service configuration."""
    provider: str
    model: str
    temperature: float
    max_tokens: int
    api_key: Optional[str] = None


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding service configuration."""
    model: str
    dimension: int
    device: str


@dataclass(frozen=True)
class VectorStoreConfig:
    """Vector store configuration."""
    db_type: str
    url: str
    collection_name: str
    api_key: Optional[str] = None


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval configuration."""
    top_k: int
    score_threshold: float
    use_hybrid_search: bool
    bm25_weight: float
    semantic_weight: float


@dataclass(frozen=True)
class ChunkingConfig:
    """Text chunking configuration."""
    chunk_size: int
    chunk_overlap: int
    strategy: str


@dataclass(frozen=True)
class MemoryConfig:
    """Conversation memory configuration."""
    memory_type: str
    window_size: int


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration."""
    llm: LLMConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    chunking: ChunkingConfig
    memory: MemoryConfig
    environment: str
    debug: bool


def load_config() -> AppConfig:
    """Load configuration from settings."""
    from src.core.config import settings
    
    return AppConfig(
        llm=LLMConfig(
            provider=settings.LLM_PROVIDER,
            model=settings.OPENAI_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            max_tokens=settings.OPENAI_MAX_TOKENS,
            api_key=settings.OPENAI_API_KEY,
        ),
        embedding=EmbeddingConfig(
            model=settings.EMBEDDING_MODEL,
            dimension=settings.EMBEDDING_DIMENSION,
            device=settings.EMBEDDING_DEVICE,
        ),
        vector_store=VectorStoreConfig(
            db_type=settings.VECTOR_DB_TYPE,
            url=settings.QDRANT_URL,
            collection_name=settings.QDRANT_COLLECTION_NAME,
            api_key=settings.QDRANT_API_KEY,
        ),
        retrieval=RetrievalConfig(
            top_k=settings.RETRIEVAL_TOP_K,
            score_threshold=settings.RETRIEVAL_SCORE_THRESHOLD,
            use_hybrid_search=settings.USE_HYBRID_SEARCH,
            bm25_weight=settings.BM25_WEIGHT,
            semantic_weight=settings.SEMANTIC_WEIGHT,
        ),
        chunking=ChunkingConfig(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            strategy=settings.TEXT_SPLITTER,
        ),
        memory=MemoryConfig(
            memory_type=settings.MEMORY_TYPE,
            window_size=settings.MEMORY_WINDOW_SIZE,
        ),
        environment=settings.ENVIRONMENT,
        debug=settings.DEBUG,
    )


# Default config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get or create the application configuration."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: AppConfig) -> None:
    """Set configuration (useful for testing)."""
    global _config
    _config = config
