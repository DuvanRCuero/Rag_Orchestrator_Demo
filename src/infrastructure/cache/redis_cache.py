"""Redis cache implementation with async operations."""

import logging
from typing import Any, Optional, List
import msgpack
import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from src.domain.interfaces.cache_service import CacheService

logger = logging.getLogger(__name__)


class RedisCache(CacheService):
    """Redis cache implementation with async operations.
    
    Features:
    - Connection pooling for efficient resource usage
    - msgpack serialization for efficient storage of embeddings
    - TTL management with configurable defaults
    - Graceful error handling
    - Health check support
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        max_connections: int = 10,
        default_ttl: int = 3600,
        encoding: str = "utf-8",
    ):
        """Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
            max_connections: Maximum number of connections in pool
            default_ttl: Default TTL in seconds
            encoding: String encoding for keys
        """
        self.redis_url = redis_url
        self.max_connections = max_connections
        self.default_ttl = default_ttl
        self.encoding = encoding
        
        # Initialize connection pool
        self._pool = ConnectionPool.from_url(
            redis_url,
            max_connections=max_connections,
            decode_responses=False,  # We'll handle encoding manually for binary data
        )
        self._client: Optional[redis.Redis] = None
    
    async def _ensure_connected(self) -> redis.Redis:
        """Ensure Redis client is connected."""
        if self._client is None:
            self._client = redis.Redis(connection_pool=self._pool)
        return self._client
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value using msgpack for efficient storage.
        
        Args:
            value: Value to serialize
            
        Returns:
            Serialized bytes
        """
        return msgpack.packb(value, use_bin_type=True)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from msgpack.
        
        Args:
            data: Serialized bytes
            
        Returns:
            Deserialized value
        """
        return msgpack.unpackb(data, raw=False)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        try:
            client = await self._ensure_connected()
            data = await client.get(key)
            if data is None:
                return None
            return self._deserialize(data)
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set a value in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds (None = use default TTL)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            client = await self._ensure_connected()
            serialized = self._serialize(value)
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
            
            if ttl > 0:
                await client.setex(key, ttl, serialized)
            else:
                await client.set(key, serialized)
            return True
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if key didn't exist
        """
        try:
            client = await self._ensure_connected()
            result = await client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            client = await self._ensure_connected()
            result = await client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False
    
    async def get_many(self, keys: List[str]) -> dict:
        """Get multiple values from cache using pipeline for efficiency.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to values (excludes missing keys)
        """
        if not keys:
            return {}
        
        try:
            client = await self._ensure_connected()
            # Use pipeline for efficient bulk operations
            pipe = client.pipeline()
            for key in keys:
                pipe.get(key)
            
            results = await pipe.execute()
            
            # Build result dictionary
            result = {}
            for key, data in zip(keys, results):
                if data is not None:
                    try:
                        result[key] = self._deserialize(data)
                    except Exception as e:
                        logger.error(f"Deserialization error for key {key}: {e}")
            
            return result
        except Exception as e:
            logger.error(f"Redis get_many error: {e}")
            return {}
    
    async def set_many(self, mapping: dict, ttl_seconds: Optional[int] = None) -> bool:
        """Set multiple values in cache using pipeline for efficiency.
        
        Args:
            mapping: Dictionary mapping keys to values
            ttl_seconds: Time to live in seconds (None = use default TTL)
            
        Returns:
            True if all successful, False otherwise
        """
        if not mapping:
            return True
        
        try:
            client = await self._ensure_connected()
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
            
            # Use pipeline for efficient bulk operations
            pipe = client.pipeline()
            for key, value in mapping.items():
                serialized = self._serialize(value)
                if ttl > 0:
                    pipe.setex(key, ttl, serialized)
                else:
                    pipe.set(key, serialized)
            
            await pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Redis set_many error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries.
        
        WARNING: This flushes the entire database!
        Use with caution in production.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            client = await self._ensure_connected()
            await client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check if Redis service is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            client = await self._ensure_connected()
            return await client.ping()
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    async def close(self):
        """Close Redis connection pool."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
