import asyncio
from typing import Any, Dict, List, Optional

import backoff
import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from src.core.config import settings
from src.core.exceptions import IngestionError
from src.domain.interfaces.embedding_service import EmbeddingServiceInterface


class EmbeddingService(EmbeddingServiceInterface):
    """Production-grade embedding service with caching, batching, and fallback."""

    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self.device = settings.EMBEDDING_DEVICE
        self.batch_size = 32
        self.cache: Dict[str, List[float]] = {}
        self._model = None
        self._client = None

        # Determine which embedding backend to use
        self.use_openai = self.model_name.startswith("text-embedding")
        self.dimension = self._get_dimension()

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
                self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                print(
                    f"Initialized OpenAI embedding service: {self.model_name}, dim={self.dimension}"
                )
        else:
            if self._model is None:
                self._model = SentenceTransformer(
                    self.model_name, device=self.device
                )
                self.dimension = self._model.get_sentence_embedding_dimension()
                print(
                    f"Initialized embedding service: {self.model_name}, dim={self.dimension}"
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
        """Embed multiple texts with batching and caching."""
        if not texts:
            return []

        # Check cache first
        uncached_texts = []
        embeddings = []
        cache_indices = []

        for idx, text in enumerate(texts):
            if text in self.cache:
                embeddings.append(self.cache[text])
                cache_indices.append(idx)
            else:
                uncached_texts.append((idx, text))

        # Process uncached texts
        if uncached_texts:
            uncached_indices = [idx for idx, _ in uncached_texts]
            uncached_texts_only = [text for _, text in uncached_texts]

            if self.use_openai:
                batch_embeddings = await self._embed_openai_batch(uncached_texts_only)
            else:
                batch_embeddings = self._embed_local_batch(uncached_texts_only)

            # Update cache and assemble final embeddings
            result_embeddings = [None] * len(texts)

            # Fill cached embeddings
            for cache_idx in cache_indices:
                result_embeddings[cache_idx] = embeddings.pop(0)

            # Fill new embeddings
            for i, (orig_idx, text) in enumerate(uncached_texts):
                embedding = batch_embeddings[i]
                self.cache[text] = embedding
                result_embeddings[orig_idx] = embedding

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
        self.cache.clear()

    @property
    def embedding_dimension(self) -> int:
        return self.dimension
