from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable parsing."""

    # API Settings
    APP_NAME: str = "RAG Orchestrator"
    API_VERSION: str = "v1"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # Vector Database
    VECTOR_DB_TYPE: str = "qdrant"  # Options: qdrant, chroma, pinecone
    QDRANT_URL: Optional[str] = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "rag_documents"
    EMBEDDING_DIMENSION: int = 384  # all-MiniLM-L6-v2 dimension

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"  # "cuda" if GPU available

    # LLM
    LLM_PROVIDER: str = "openai"  # Options: openai, anthropic, local
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_TEMPERATURE: float = 0.1
    OPENAI_MAX_TOKENS: int = 1000

    # Anthropic (optional)
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-sonnet-20240229"
    ANTHROPIC_MAX_TOKENS: int = 1000

    # Local LLM / Ollama (optional)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LOCAL_MODEL: str = "llama2"

    # Text Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TEXT_SPLITTER: str = (
        "recursive_character"  # Options: recursive_character, token, semantic
    )

    # Retrieval
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_SCORE_THRESHOLD: float = 0.7
    USE_HYBRID_SEARCH: bool = True
    BM25_WEIGHT: float = 0.3
    SEMANTIC_WEIGHT: float = 0.7

    # Memory
    MEMORY_TYPE: str = "buffer_window"  # Options: buffer, window, summary
    MEMORY_WINDOW_SIZE: int = 10

    # Cache Configuration
    CACHE_TYPE: str = "memory"  # Options: redis, memory
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_EMBEDDING_TTL: int = 604800  # 7 days in seconds
    CACHE_QUERY_TTL: int = 3600  # 1 hour in seconds
    CACHE_SESSION_TTL: int = 1800  # 30 minutes in seconds
    CACHE_MAX_CONNECTIONS: int = 10

    # Redis (if you have it in your .env, add it here)
    # redis_url: Optional[str] = None

    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Changed from "forbid" to "ignore" to allow extra env vars
    )

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def validate_openai_key(cls, v, info):
        # Skip validation in testing environment
        if info.data.get("ENVIRONMENT") == "testing":
            return v
        if info.data.get("LLM_PROVIDER") == "openai" and not v:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI LLM")
        return v


settings = Settings()