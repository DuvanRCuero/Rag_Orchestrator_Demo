"""Tests for cache infrastructure."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Set environment variables before importing
os.environ["ENVIRONMENT"] = "testing"
os.environ["OPENAI_API_KEY"] = "test-key"

from src.infrastructure.cache import (
    CacheInterface,
    InMemoryCache,
    RedisCache,
    CacheKeyBuilder,
)


class TestCacheKeyBuilder:
    """Test cache key building."""

    def test_embedding_key(self):
        """Test embedding key generation."""
        key = CacheKeyBuilder.embedding_key("test text", "model-v1")
        assert key.startswith("emb:model-v1:")
        assert len(key.split(":")[-1]) == 16  # Hash length

    def test_embedding_key_consistency(self):
        """Test that same text produces same key."""
        key1 = CacheKeyBuilder.embedding_key("test", "model")
        key2 = CacheKeyBuilder.embedding_key("test", "model")
        assert key1 == key2

    def test_embedding_key_different_text(self):
        """Test that different text produces different keys."""
        key1 = CacheKeyBuilder.embedding_key("test1", "model")
        key2 = CacheKeyBuilder.embedding_key("test2", "model")
        assert key1 != key2

    def test_query_result_key(self):
        """Test query result key generation."""
        key = CacheKeyBuilder.query_result_key("query", 5, "params123")
        assert key.startswith("qry:")
        assert "5" in key
        assert "params123" in key

    def test_session_key(self):
        """Test session key generation."""
        key = CacheKeyBuilder.session_key("session-123")
        assert key == "sess:session-123"


class TestInMemoryCache:
    """Test in-memory cache."""

    @pytest.fixture
    def cache(self):
        """Create cache instance."""
        return InMemoryCache()

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        """Test setting and getting values."""
        await cache.set("key1", "value1")
        value = await cache.get("key1")
        assert value == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, cache):
        """Test getting nonexistent key."""
        value = await cache.get("nonexistent")
        assert value is None

    @pytest.mark.asyncio
    async def test_delete(self, cache):
        """Test deleting keys."""
        await cache.set("key1", "value1")
        result = await cache.delete("key1")
        assert result is True
        value = await cache.get("key1")
        assert value is None

    @pytest.mark.asyncio
    async def test_exists(self, cache):
        """Test checking key existence."""
        await cache.set("key1", "value1")
        assert await cache.exists("key1") is True
        assert await cache.exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_mget(self, cache):
        """Test getting multiple values."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        result = await cache.mget(["key1", "key2", "key3"])
        assert result == {"key1": "value1", "key2": "value2"}

    @pytest.mark.asyncio
    async def test_mset(self, cache):
        """Test setting multiple values."""
        items = {"key1": "value1", "key2": "value2"}
        result = await cache.mset(items)
        assert result is True
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"

    @pytest.mark.asyncio
    async def test_stats(self, cache):
        """Test cache statistics."""
        await cache.set("key1", "value1")
        await cache.get("key1")  # hit
        await cache.get("key2")  # miss
        
        stats = await cache.get_stats()
        assert stats["total_hits"] == 1
        assert stats["total_misses"] == 1
        assert stats["keys_count"] == 1
        assert 0 <= stats["hit_rate"] <= 1

    @pytest.mark.asyncio
    async def test_close(self, cache):
        """Test closing cache."""
        await cache.close()  # Should not raise

    def test_clear(self, cache):
        """Test clearing cache."""
        cache.clear()
        # Should not raise


class TestRedisCache:
    """Test Redis cache."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        mock = AsyncMock()
        mock.get = AsyncMock(return_value=None)
        mock.setex = AsyncMock(return_value=True)
        mock.delete = AsyncMock(return_value=1)
        mock.exists = AsyncMock(return_value=1)
        mock.mget = AsyncMock(return_value=[])
        mock.dbsize = AsyncMock(return_value=0)
        mock.info = AsyncMock(return_value={"used_memory": 1024})
        mock.pipeline = MagicMock()
        mock.close = AsyncMock()
        return mock

    @pytest.fixture
    def cache(self, mock_redis):
        """Create cache instance with mocked Redis."""
        cache = RedisCache("redis://localhost:6379")
        cache._pool = mock_redis
        return cache

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache, mock_redis):
        """Test setting and getting values."""
        mock_redis.setex.return_value = True
        mock_redis.get.return_value = '"value1"'
        
        result = await cache.set("key1", "value1")
        assert result is True
        
        value = await cache.get("key1")
        assert value == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, cache, mock_redis):
        """Test getting nonexistent key."""
        mock_redis.get.return_value = None
        value = await cache.get("nonexistent")
        assert value is None

    @pytest.mark.asyncio
    async def test_delete(self, cache, mock_redis):
        """Test deleting keys."""
        mock_redis.delete.return_value = 1
        result = await cache.delete("key1")
        assert result is True

    @pytest.mark.asyncio
    async def test_exists(self, cache, mock_redis):
        """Test checking key existence."""
        mock_redis.exists.return_value = 1
        assert await cache.exists("key1") is True
        
        mock_redis.exists.return_value = 0
        assert await cache.exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_mget(self, cache, mock_redis):
        """Test getting multiple values."""
        mock_redis.mget.return_value = ['"value1"', '"value2"', None]
        result = await cache.mget(["key1", "key2", "key3"])
        assert result == {"key1": "value1", "key2": "value2"}

    @pytest.mark.asyncio
    async def test_mset(self, cache, mock_redis):
        """Test setting multiple values."""
        # Create mock pipeline
        mock_pipeline = AsyncMock()
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=None)
        mock_pipeline.setex = AsyncMock()
        mock_pipeline.execute = AsyncMock(return_value=[True, True])
        mock_redis.pipeline.return_value = mock_pipeline
        
        items = {"key1": "value1", "key2": "value2"}
        result = await cache.mset(items)
        assert result is True

    @pytest.mark.asyncio
    async def test_stats(self, cache, mock_redis):
        """Test cache statistics."""
        mock_redis.info.return_value = {"used_memory": 1048576}
        mock_redis.dbsize.return_value = 100
        
        stats = await cache.get_stats()
        assert "hit_rate" in stats
        assert "total_hits" in stats
        assert "total_misses" in stats
        assert "memory_usage_mb" in stats
        assert "keys_count" in stats
        assert stats["keys_count"] == 100

    @pytest.mark.asyncio
    async def test_close(self, cache, mock_redis):
        """Test closing cache."""
        await cache.close()
        mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_get(self, cache, mock_redis):
        """Test error handling on get."""
        mock_redis.get.side_effect = Exception("Connection error")
        value = await cache.get("key1")
        assert value is None

    @pytest.mark.asyncio
    async def test_error_handling_set(self, cache, mock_redis):
        """Test error handling on set."""
        mock_redis.setex.side_effect = Exception("Connection error")
        result = await cache.set("key1", "value1")
        assert result is False


class TestCacheInterface:
    """Test cache interface."""

    def test_interface_methods(self):
        """Test that interface has required methods."""
        assert hasattr(CacheInterface, "get")
        assert hasattr(CacheInterface, "set")
        assert hasattr(CacheInterface, "delete")
        assert hasattr(CacheInterface, "exists")
        assert hasattr(CacheInterface, "mget")
        assert hasattr(CacheInterface, "mset")
