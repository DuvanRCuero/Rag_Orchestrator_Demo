"""Dependency injection container."""

from src.core.config_models import AppConfig, get_config


class Container:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._config = None
        return cls._instance

    def __init__(self, config: AppConfig = None):
        if not hasattr(self, '_config') or self._config is None:
            self._config = config or get_config()

    @classmethod
    def with_config(cls, config: AppConfig) -> 'Container':
        """Create container with custom config (for testing)."""
        instance = cls.__new__(cls)
        instance._initialized = False
        instance._config = config
        return instance

    def _init_services(self):
        from src.infrastructure.llm import LLMFactory
        from src.infrastructure.vector.qdrant_client import QdrantVectorStore
        from src.domain.embeddings import EmbeddingService
        from src.application.chains.rag_chain import AdvancedRAGChain
        from src.application.chains.memory import EnhancedConversationMemory
        from src.infrastructure.cache import RedisCache, MemoryCache

        # Initialize cache service based on configuration
        cache_config = self._config.cache
        if cache_config.type == "redis":
            try:
                self._cache_service = RedisCache(
                    redis_url=cache_config.redis_url,
                    max_connections=cache_config.max_connections,
                    default_ttl=cache_config.embedding_ttl,
                )
            except Exception as e:
                # Fall back to memory cache if Redis is unavailable
                from src.core.logging import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Redis unavailable, falling back to memory cache: {e}")
                self._cache_service = MemoryCache(
                    max_size=10000,
                    default_ttl=cache_config.embedding_ttl
                )
        else:
            # Use in-memory cache
            self._cache_service = MemoryCache(
                max_size=10000,
                default_ttl=cache_config.embedding_ttl
            )

        self._llm = LLMFactory.create(self._config.llm.provider, self._config.llm)
        self._embeddings = EmbeddingService(
            self._config.embedding,
            cache_service=self._cache_service
        )
        self._vector_store = QdrantVectorStore(
            store_config=self._config.vector_store,
            retrieval_config=self._config.retrieval,
        )
        self._memory = EnhancedConversationMemory()

        self._rag_chain = AdvancedRAGChain(
            vector_store=self._vector_store,
            embedding_service=self._embeddings,
            llm_service=self._llm,
            conversation_memory=self._memory,
        )
        self._initialized = True

    def _ensure_initialized(self):
        if not self._initialized:
            self._init_services()

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def rag_chain(self):
        self._ensure_initialized()
        return self._rag_chain

    @property
    def vector_store(self):
        self._ensure_initialized()
        return self._vector_store

    @property
    def embedding_service(self):
        self._ensure_initialized()
        return self._embeddings

    @property
    def llm_service(self):
        self._ensure_initialized()
        return self._llm

    @property
    def conversation_memory(self):
        self._ensure_initialized()
        return self._memory

    @property
    def cache_service(self):
        self._ensure_initialized()
        return self._cache_service

    @property
    def retrieval_service(self):
        self._ensure_initialized()
        if not hasattr(self, '_retrieval_service'):
            from src.application.services import RetrievalService
            self._retrieval_service = RetrievalService(
                vector_store=self._vector_store,
                embedding_service=self._embeddings,
                cache_service=self._cache_service,
            )
        return self._retrieval_service

    @property
    def generation_service(self):
        self._ensure_initialized()
        if not hasattr(self, '_generation_service'):
            from src.application.services import GenerationService
            self._generation_service = GenerationService(llm_service=self._llm)
        return self._generation_service

    @property
    def verification_service(self):
        self._ensure_initialized()
        if not hasattr(self, '_verification_service'):
            from src.application.services import VerificationService
            self._verification_service = VerificationService(llm_service=self._llm)
        return self._verification_service

    @property
    def query_enhancement_service(self):
        self._ensure_initialized()
        if not hasattr(self, '_query_enhancement_service'):
            from src.application.services import QueryEnhancementService
            self._query_enhancement_service = QueryEnhancementService(
                llm_service=self._llm,
                embedding_service=self._embeddings,
            )
        return self._query_enhancement_service

    @property
    def ask_question_use_case(self):
        self._ensure_initialized()
        if not hasattr(self, '_ask_question_use_case'):
            from src.application.use_cases import AskQuestionUseCase
            self._ask_question_use_case = AskQuestionUseCase(
                retrieval_service=self.retrieval_service,
                generation_service=self.generation_service,
                verification_service=self.verification_service,
                query_enhancement_service=self.query_enhancement_service,
                conversation_memory=self._memory,
            )
        return self._ask_question_use_case

    @property
    def stream_answer_use_case(self):
        self._ensure_initialized()
        if not hasattr(self, '_stream_answer_use_case'):
            from src.application.use_cases import StreamAnswerUseCase
            self._stream_answer_use_case = StreamAnswerUseCase(
                retrieval_service=self.retrieval_service,
                generation_service=self.generation_service,
                conversation_memory=self._memory,
            )
        return self._stream_answer_use_case


container = Container()
