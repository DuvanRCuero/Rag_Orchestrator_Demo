import asyncio
from typing import Any, Dict, List, Optional

import backoff
import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from src.core.config import settings
from src.core.config_models import EmbeddingConfig, get_config
from src.core.exceptions import IngestionError
from src.core.logging import get_logger
from src.domain.interfaces.embedding_service import EmbeddingServiceInterface
from src.infrastructure.cache.interfaces import CacheInterface
from src.infrastructure.cache.key_builder import CacheKeyBuilder

logger = get_logger(__name__)


class EmbeddingService(EmbeddingServiceInterface):
    """Production-grade embedding service with caching, batching, and fallback."""

    def __init__(
        self, config: EmbeddingConfig = None, cache: CacheInterface = None
    ):
        self.config = config or get_config().embedding
        self.cache = cache  # L2 Redis cache (injected)
        
        self.model_name = self.config.model
        self.device = self.config.device
        self.dimension = self.config.dimension
        self.batch_size = 32
        self._local_cache: Dict[str, List[float]] = {}  # L1 in-memory cache
        self._model = None
        self._client = None

        # Determine which embedding backend to use
        self.use_openai = self.model_name.startswith("text-embedding")

    def _get_dimension(self) -> int:
        """Get embedding dimension without loading the model."""
        if self.use_openai:
            return self._get_openai_dimension()
        else:
            # Default dimensions for common models
            dim_map = {
                "sentence-transformers/all-MiniLM-L6-v2": 384,
                "sentence-transformers/all-mpnet-base-v2": 768,
                "sentence-transformers/all-MiniLM-L12-v2": 384,
            }
            return dim_map.get(self.model_name, 384)

    def _get_openai_dimension(self) -> int:
        """Get embedding dimension for OpenAI models."""
        dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dim_map.get(self.model_name, 1536)

    def _initialize_model(self):
        """Lazy initialization of the model."""
        if self.use_openai:
            if self._client is None:
                # Get API key from main config
                api_key = get_config().llm.api_key
                self._client = AsyncOpenAI(api_key=api_key)
                logger.info(
                    "openai_embedding_service_initialized",
                    model=self.model_name,
                    dimension=self.dimension,
                )
        else:
            if self._model is None:
                self._model = SentenceTransformer(
                    self.model_name, device=self.device
                )
                self.dimension = self._model.get_sentence_embedding_dimension()
                logger.info(
                    "embedding_service_initialized",
                    model=self.model_name,
                    dimension=self.dimension,
                    device=self.device,
                )

    @property
    def model(self):
        """Get the model, initializing if needed."""
        if not self.use_openai and self._model is None:
            self._initialize_model()
        return self._model

    @property
    def client(self):
        """Get the OpenAI client, initializing if needed."""
        if self.use_openai and self._client is None:
            self._initialize_model()
        return self._client

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts with batching and two-level caching."""
        if not texts:
            return []

        # Check L1 cache (in-memory) and L2 cache (Redis)
        uncached_texts = []
        embeddings = []
        cache_indices = []

        for idx, text in enumerate(texts):
            cache_key = CacheKeyBuilder.embedding_key(text, self.model_name)
            
            # Check L1 cache first
            if cache_key in self._local_cache:
                embeddings.append(self._local_cache[cache_key])
                cache_indices.append(idx)
                continue
            
            # Check L2 cache (Redis) if available
            if self.cache:
                try:
                    cached_embedding = await self.cache.get(cache_key)
                    if cached_embedding:
                        self._local_cache[cache_key] = cached_embedding
                        embeddings.append(cached_embedding)
                        cache_indices.append(idx)
                        continue
                except Exception as e:
                    logger.warning(
                        "cache_get_failed", 
                        key=cache_key, 
                        error=str(e)
                    )
            
            uncached_texts.append((idx, text, cache_key))

        # Process uncached texts
        if uncached_texts:
            uncached_indices = [idx for idx, _, _ in uncached_texts]
            uncached_texts_only = [text for _, text, _ in uncached_texts]
            cache_keys = [key for _, _, key in uncached_texts]

            if self.use_openai:
                batch_embeddings = await self._embed_openai_batch(uncached_texts_only)
            else:
                batch_embeddings = self._embed_local_batch(uncached_texts_only)

            # Update caches and assemble final embeddings
            result_embeddings = [None] * len(texts)

            # Fill cached embeddings
            for cache_idx in cache_indices:
                result_embeddings[cache_idx] = embeddings.pop(0)

            # Fill new embeddings and update caches
            cache_items = {}
            for i, (orig_idx, text, cache_key) in enumerate(uncached_texts):
                embedding = batch_embeddings[i]
                # Update L1 cache
                self._local_cache[cache_key] = embedding
                result_embeddings[orig_idx] = embedding
                # Prepare for L2 cache batch update
                cache_items[cache_key] = embedding

            # Batch update L2 cache (Redis)
            if self.cache and cache_items:
                try:
                    await self.cache.mset(
                        cache_items,
                        ttl=get_config().cache.embedding_ttl,
                    )
                except Exception as e:
                    logger.warning(
                        "cache_mset_failed",
                        items_count=len(cache_items),
                        error=str(e),
                    )

            return result_embeddings

        return embeddings

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=30)
    async def _embed_openai_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using OpenAI API with retry logic."""
        try:
            response = await self.client.embeddings.create(
                model=self.model_name, input=texts, encoding_format="float"
            )

            embeddings = [data.embedding for data in response.data]

            # Ensure correct order
            if len(embeddings) != len(texts):
                raise IngestionError(
                    detail="Mismatch between input texts and embeddings received"
                )

            return embeddings

        except Exception as e:
            raise IngestionError(
                detail=f"OpenAI embedding failed: {str(e)}",
                metadata={"batch_size": len(texts), "model": self.model_name},
            )

    def _embed_local_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using local sentence-transformers."""
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device,
            )

            return embeddings.tolist()

        except Exception as e:
            raise IngestionError(
                detail=f"Local embedding failed: {str(e)}",
                metadata={"batch_size": len(texts), "model": self.model_name},
            )

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        embeddings = await self.embed_texts([query])
        return embeddings[0] if embeddings else []

    def get_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not embedding1 or not embedding2:
            return 0.0

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Normalize
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def clear_cache(self):
        """Clear embedding cache."""
        self._local_cache.clear()
        logger.info("local_cache_cleared")

    @property
    def embedding_dimension(self) -> int:
        return self.dimension
