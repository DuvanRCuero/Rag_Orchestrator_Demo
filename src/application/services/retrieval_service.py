"""Retrieval Service - handles all retrieval logic."""

import time
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

from src.domain.interfaces import EmbeddingServiceInterface, VectorStore
from src.core.config import settings
from src.core.schemas import DocumentChunk


@dataclass
class RetrievalResult:
    chunks: List[DocumentChunk]
    retrieval_time: float
    query_variations: List[str] = field(default_factory=list)
    retrieval_method: str = "semantic"
    reranked: bool = False


class RetrievalService:

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingServiceInterface,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
    ) -> RetrievalResult:
        start_time = time.time()

        query_embedding = await self.embedding_service.embed_query(query)

        if use_hybrid and settings.USE_HYBRID_SEARCH:
            chunks = await self.vector_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                top_k=top_k,
            )
            method = "hybrid"
        else:
            chunks = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
            )
            method = "semantic"

        return RetrievalResult(
            chunks=chunks,
            retrieval_time=time.time() - start_time,
            retrieval_method=method,
        )

    async def retrieve_with_multi_query(
        self,
        query: str,
        query_variations: List[str],
        top_k: int = 5,
    ) -> RetrievalResult:
        start_time = time.time()
        all_chunks = []

        all_queries = [query] + query_variations
        for q in all_queries:
            result = await self.retrieve(q, top_k=top_k * 2, use_hybrid=True)
            all_chunks.extend(result.chunks)

        unique_chunks = self._deduplicate_chunks(all_chunks)

        return RetrievalResult(
            chunks=unique_chunks[:top_k],
            retrieval_time=time.time() - start_time,
            query_variations=query_variations,
            retrieval_method="multi_query_hybrid",
        )

    def _deduplicate_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        seen = set()
        unique = []
        for chunk in chunks:
            # Use SHA-256 for better collision resistance
            import hashlib
            content_hash = hashlib.sha256(chunk.content[:100].encode()).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(chunk)
        return unique
