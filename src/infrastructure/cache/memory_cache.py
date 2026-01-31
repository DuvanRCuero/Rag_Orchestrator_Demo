"""In-memory cache implementation with TTL support."""

import asyncio
import time
from typing import Any, Optional, List
from cachetools import TTLCache

from src.domain.interfaces.cache_service import CacheService


class MemoryCache(CacheService):
    """In-memory cache implementation using cachetools.TTLCache.
    
    This serves as a fallback when Redis is unavailable or for testing.
    Uses TTLCache for automatic expiration of entries.
    """
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        """Initialize in-memory cache.
        
        Args:
            max_size: Maximum number of entries to store
            default_ttl: Default TTL in seconds
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        # Using TTLCache with default TTL
        self._cache = TTLCache(maxsize=max_size, ttl=default_ttl)
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        async with self._lock:
            return self._cache.get(key)
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set a value in cache with optional TTL.
        
        Note: TTLCache doesn't support per-key TTL, so we use the default TTL.
        For per-key TTL, we store expiration metadata.
        """
        try:
            async with self._lock:
                if ttl_seconds is not None:
                    # Store with expiration timestamp
                    expires_at = time.time() + ttl_seconds
                    self._cache[key] = {"value": value, "expires_at": expires_at}
                else:
                    # Store without expiration metadata
                    self._cache[key] = {"value": value, "expires_at": None}
            return True
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        async with self._lock:
            if key not in self._cache:
                return False
            
            # Check custom expiration
            entry = self._cache[key]
            if isinstance(entry, dict) and "expires_at" in entry:
                expires_at = entry["expires_at"]
                if expires_at is not None and time.time() > expires_at:
                    # Expired, remove it
                    del self._cache[key]
                    return False
            return True
    
    async def get_many(self, keys: List[str]) -> dict:
        """Get multiple values from cache."""
        result = {}
        async with self._lock:
            current_time = time.time()
            for key in keys:
                if key in self._cache:
                    entry = self._cache[key]
                    if isinstance(entry, dict) and "value" in entry:
                        # Check expiration
                        expires_at = entry.get("expires_at")
                        if expires_at is None or current_time <= expires_at:
                            result[key] = entry["value"]
                        else:
                            # Expired, remove it
                            del self._cache[key]
                    else:
                        # Legacy format without expiration
                        result[key] = entry
        return result
    
    async def set_many(self, mapping: dict, ttl_seconds: Optional[int] = None) -> bool:
        """Set multiple values in cache with optional TTL."""
        try:
            async with self._lock:
                if ttl_seconds is not None:
                    expires_at = time.time() + ttl_seconds
                    for key, value in mapping.items():
                        self._cache[key] = {"value": value, "expires_at": expires_at}
                else:
                    for key, value in mapping.items():
                        self._cache[key] = {"value": value, "expires_at": None}
            return True
        except Exception:
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            async with self._lock:
                self._cache.clear()
            return True
        except Exception:
            return False
    
    async def health_check(self) -> bool:
        """Check if cache service is healthy."""
        try:
            # Simple check: try to set and get a value
            test_key = "__health_check__"
            await self.set(test_key, "ok", ttl_seconds=1)
            result = await self.get(test_key)
            await self.delete(test_key)
            return result is not None and result.get("value") == "ok"
        except Exception:
            return False
