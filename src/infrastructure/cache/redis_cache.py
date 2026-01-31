"""Production Redis cache with fallback in-memory cache."""

import asyncio
import json
from typing import Any, Dict, List, Optional

from src.core.logging import get_logger
from src.infrastructure.cache.interfaces import CacheInterface

logger = get_logger(__name__)


class RedisCache(CacheInterface):
    """Production Redis cache with connection pooling."""

    def __init__(self, url: str, default_ttl: int = 86400):
        """
        Initialize Redis cache.
        
        Args:
            url: Redis connection URL
            default_ttl: Default TTL in seconds (default: 24 hours)
        """
        self.url = url
        self.default_ttl = default_ttl
        self._pool = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
        }

    async def get_pool(self):
        """Get or create Redis connection pool."""
        if self._pool is None:
            try:
                from redis import asyncio as aioredis
                
                self._pool = await aioredis.from_url(
                    self.url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=20,
                )
                logger.info("redis_cache_initialized", url=self.url)
            except Exception as e:
                logger.error("redis_connection_failed", error=str(e), url=self.url)
                raise
        return self._pool

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache by key."""
        try:
            pool = await self.get_pool()
            value = await pool.get(key)
            if value:
                self._stats["hits"] += 1
                return json.loads(value)
            self._stats["misses"] += 1
            return None
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("redis_get_failed", key=key, error=str(e))
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        try:
            pool = await self.get_pool()
            ttl = ttl or self.default_ttl
            result = await pool.setex(key, ttl, json.dumps(value))
            return bool(result)
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("redis_set_failed", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            pool = await self.get_pool()
            result = await pool.delete(key)
            return bool(result)
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("redis_delete_failed", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            pool = await self.get_pool()
            result = await pool.exists(key)
            return bool(result)
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("redis_exists_failed", key=key, error=str(e))
            return False

    async def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        try:
            pool = await self.get_pool()
            values = await pool.mget(keys)
            result = {}
            for key, value in zip(keys, values):
                if value:
                    result[key] = json.loads(value)
                    self._stats["hits"] += 1
                else:
                    self._stats["misses"] += 1
            return result
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("redis_mget_failed", keys_count=len(keys), error=str(e))
            return {}

    async def mset(
        self, items: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Set multiple values in cache."""
        try:
            pool = await self.get_pool()
            ttl = ttl or self.default_ttl
            
            # Use pipeline for batch operations
            async with pool.pipeline(transaction=True) as pipe:
                for key, value in items.items():
                    await pipe.setex(key, ttl, json.dumps(value))
                await pipe.execute()
            return True
        except Exception as e:
            self._stats["errors"] += 1
            logger.error("redis_mset_failed", items_count=len(items), error=str(e))
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            pool = await self.get_pool()
            info = await pool.info("memory")
            keys_count = await pool.dbsize()
            
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / total_requests if total_requests > 0 else 0
            )
            
            return {
                "hit_rate": round(hit_rate, 2),
                "total_hits": self._stats["hits"],
                "total_misses": self._stats["misses"],
                "total_errors": self._stats["errors"],
                "memory_usage_mb": round(
                    info.get("used_memory", 0) / (1024 * 1024), 2
                ),
                "keys_count": keys_count,
            }
        except Exception as e:
            logger.error("redis_stats_failed", error=str(e))
            return {
                "hit_rate": 0,
                "total_hits": self._stats["hits"],
                "total_misses": self._stats["misses"],
                "total_errors": self._stats["errors"],
                "memory_usage_mb": 0,
                "keys_count": 0,
                "error": str(e),
            }

    async def close(self):
        """Close Redis connection."""
        if self._pool:
            await self._pool.close()
            self._pool = None


class InMemoryCache(CacheInterface):
    """Fallback in-memory cache for testing/development."""

    def __init__(self, default_ttl: int = 86400):
        """
        Initialize in-memory cache.
        
        Args:
            default_ttl: Default TTL in seconds (default: 24 hours)
        """
        self.default_ttl = default_ttl
        self._cache: Dict[str, Any] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
        }

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache by key."""
        value = self._cache.get(key)
        if value is not None:
            self._stats["hits"] += 1
        else:
            self._stats["misses"] += 1
        return value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        self._cache[key] = value
        # Note: In-memory cache doesn't enforce TTL
        return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._cache

    async def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        result = {}
        for key in keys:
            value = self._cache.get(key)
            if value is not None:
                result[key] = value
                self._stats["hits"] += 1
            else:
                self._stats["misses"] += 1
        return result

    async def mset(
        self, items: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Set multiple values in cache."""
        self._cache.update(items)
        return True

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (
            self._stats["hits"] / total_requests if total_requests > 0 else 0
        )
        
        return {
            "hit_rate": round(hit_rate, 2),
            "total_hits": self._stats["hits"],
            "total_misses": self._stats["misses"],
            "memory_usage_mb": 0,  # Not tracked for in-memory
            "keys_count": len(self._cache),
        }

    async def close(self):
        """Close connection (no-op for in-memory)."""
        pass

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
