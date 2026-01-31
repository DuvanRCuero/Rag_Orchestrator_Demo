"""Retrieval Service - handles all retrieval logic."""

import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Optional

from src.domain.interfaces import EmbeddingServiceInterface, VectorReader
from src.domain.interfaces.cache_service import CacheService
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
        vector_store: VectorReader,
        embedding_service: EmbeddingServiceInterface,
        cache_service: Optional[CacheService] = None,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.cache_service = cache_service
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def _generate_cache_key(self, query: str, top_k: int, use_hybrid: bool, filters: Optional[dict] = None) -> str:
        """Generate cache key for query results.
        
        Args:
            query: Query text
            top_k: Number of results
            use_hybrid: Whether hybrid search is used
            filters: Optional filters applied
            
        Returns:
            Hash-based cache key
        """
        # Create a stable representation of the query parameters
        cache_data = {
            "query": query,
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "filters": filters or {}
        }
        
        # Serialize to JSON with sorted keys for consistency
        cache_string = json.dumps(cache_data, sort_keys=True)
        
        # Generate hash
        cache_hash = hashlib.sha256(cache_string.encode('utf-8')).hexdigest()
        
        return f"query:{cache_hash}"

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
    ) -> RetrievalResult:
        # Check cache if available
        if self.cache_service:
            cache_key = self._generate_cache_key(query, top_k, use_hybrid)
            cached_result = await self.cache_service.get(cache_key)
            
            if cached_result is not None:
                self.cache_hits += 1
                # Reconstruct RetrievalResult from cached data
                return RetrievalResult(
                    chunks=[DocumentChunk(**chunk_dict) for chunk_dict in cached_result["chunks"]],
                    retrieval_time=cached_result["retrieval_time"],
                    retrieval_method=cached_result["retrieval_method"],
                    reranked=cached_result.get("reranked", False),
                )
            
            self.cache_misses += 1
        
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

        result = RetrievalResult(
            chunks=chunks,
            retrieval_time=time.time() - start_time,
            retrieval_method=method,
        )
        
        # Cache the result if cache service is available
        if self.cache_service:
            # Serialize chunks for caching
            cache_data = {
                "chunks": [chunk.__dict__ for chunk in chunks],
                "retrieval_time": result.retrieval_time,
                "retrieval_method": method,
                "reranked": False,
            }
            
            # Use query TTL from config (1 hour)
            from src.core.config_models import get_config
            cache_config = get_config().cache
            await self.cache_service.set(
                cache_key,
                cache_data,
                ttl_seconds=cache_config.query_ttl
            )
        
        return result

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
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache hits, misses, and hit rate
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total": total,
            "hit_rate_percent": round(hit_rate, 2)
        }
    
    async def invalidate_cache(self) -> bool:
        """Invalidate all query result cache entries.
        
        This should be called when documents are updated.
        
        Returns:
            True if successful, False otherwise
        """
        if self.cache_service:
            # In a production system, you'd want to track query cache keys
            # For now, we just clear the entire cache
            return await self.cache_service.clear()
        return False
