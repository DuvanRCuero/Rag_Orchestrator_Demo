"""Cache service interface for dependency injection."""

from abc import ABC, abstractmethod
from typing import Any, Optional, List


class CacheService(ABC):
    """Abstract interface for cache services.
    
    Provides a unified interface for different caching backends
    (Redis, in-memory, etc.) with TTL management.
    """
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set a value in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds (None = no expiration)
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if key didn't exist
        """
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_many(self, keys: List[str]) -> dict:
        """Get multiple values from cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to values (excludes missing keys)
        """
        pass
    
    @abstractmethod
    async def set_many(self, mapping: dict, ttl_seconds: Optional[int] = None) -> bool:
        """Set multiple values in cache with optional TTL.
        
        Args:
            mapping: Dictionary mapping keys to values
            ttl_seconds: Time to live in seconds (None = no expiration)
            
        Returns:
            True if all successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if cache service is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
