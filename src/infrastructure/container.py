"""Dependency injection container."""


class Container:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _init_services(self):
        from src.infrastructure.llm import LLMFactory
        from src.infrastructure.vector.qdrant_client import QdrantVectorStore
        from src.domain.embeddings import EmbeddingService
        from src.application.chains.rag_chain import AdvancedRAGChain
        from src.application.chains.memory import EnhancedConversationMemory

        self._llm = LLMFactory.create()
        self._embeddings = EmbeddingService()
        self._vector_store = QdrantVectorStore()
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


container = Container()
