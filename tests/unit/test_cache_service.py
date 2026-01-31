"""Unit tests for cache service implementations."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from src.infrastructure.cache import RedisCache, MemoryCache
from src.domain.interfaces.cache_service import CacheService


@pytest.mark.asyncio
class TestMemoryCache:
    """Tests for in-memory cache implementation."""

    async def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = MemoryCache(max_size=100, default_ttl=3600)
        
        # Set a value
        result = await cache.set("test_key", "test_value", ttl_seconds=60)
        assert result is True
        
        # Get the value
        value = await cache.get("test_key")
        assert value is not None
        assert value["value"] == "test_value"
        assert "expires_at" in value
    
    async def test_get_nonexistent_key(self):
        """Test getting a non-existent key returns None."""
        cache = MemoryCache(max_size=100, default_ttl=3600)
        
        value = await cache.get("nonexistent")
        assert value is None
    
    async def test_delete(self):
        """Test deleting a key."""
        cache = MemoryCache(max_size=100, default_ttl=3600)
        
        await cache.set("test_key", "test_value")
        assert await cache.exists("test_key") is True
        
        result = await cache.delete("test_key")
        assert result is True
        
        assert await cache.exists("test_key") is False
    
    async def test_delete_nonexistent_key(self):
        """Test deleting a non-existent key returns False."""
        cache = MemoryCache(max_size=100, default_ttl=3600)
        
        result = await cache.delete("nonexistent")
        assert result is False
    
    async def test_exists(self):
        """Test checking if a key exists."""
        cache = MemoryCache(max_size=100, default_ttl=3600)
        
        await cache.set("test_key", "test_value")
        assert await cache.exists("test_key") is True
        assert await cache.exists("nonexistent") is False
    
    async def test_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = MemoryCache(max_size=100, default_ttl=3600)
        
        # Set with 1 second TTL
        await cache.set("test_key", "test_value", ttl_seconds=1)
        assert await cache.exists("test_key") is True
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Key should be expired
        assert await cache.exists("test_key") is False
    
    async def test_get_many(self):
        """Test getting multiple values."""
        cache = MemoryCache(max_size=100, default_ttl=3600)
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        
        result = await cache.get_many(["key1", "key2", "key4"])
        
        assert "key1" in result
        assert result["key1"] == "value1"
        assert "key2" in result
        assert result["key2"] == "value2"
        assert "key4" not in result  # Non-existent key
    
    async def test_set_many(self):
        """Test setting multiple values."""
        cache = MemoryCache(max_size=100, default_ttl=3600)
        
        mapping = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        
        result = await cache.set_many(mapping, ttl_seconds=60)
        assert result is True
        
        # Verify all values were set
        for key, value in mapping.items():
            cached = await cache.get(key)
            assert cached is not None
            assert cached["value"] == value
    
    async def test_clear(self):
        """Test clearing all cache entries."""
        cache = MemoryCache(max_size=100, default_ttl=3600)
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        result = await cache.clear()
        assert result is True
        
        assert await cache.exists("key1") is False
        assert await cache.exists("key2") is False
    
    async def test_health_check(self):
        """Test health check returns True for functioning cache."""
        cache = MemoryCache(max_size=100, default_ttl=3600)
        
        result = await cache.health_check()
        assert result is True
    
    async def test_embedding_storage(self):
        """Test storing and retrieving embeddings."""
        cache = MemoryCache(max_size=100, default_ttl=3600)
        
        # Simulate embedding vector
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        await cache.set("embed:test", embedding, ttl_seconds=3600)
        
        retrieved = await cache.get("embed:test")
        assert retrieved is not None
        assert retrieved["value"] == embedding


@pytest.mark.asyncio
class TestRedisCache:
    """Tests for Redis cache implementation with mocked Redis."""

    @pytest.fixture
    async def mock_redis_client(self):
        """Create a mock Redis client."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)
        mock_client.set = AsyncMock(return_value=True)
        mock_client.setex = AsyncMock(return_value=True)
        mock_client.delete = AsyncMock(return_value=1)
        mock_client.exists = AsyncMock(return_value=1)
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.flushdb = AsyncMock(return_value=True)
        mock_client.pipeline = MagicMock()
        
        # Setup pipeline mock
        mock_pipe = AsyncMock()
        mock_pipe.get = MagicMock(return_value=mock_pipe)
        mock_pipe.set = MagicMock(return_value=mock_pipe)
        mock_pipe.setex = MagicMock(return_value=mock_pipe)
        mock_pipe.execute = AsyncMock(return_value=[])
        mock_client.pipeline.return_value = mock_pipe
        
        return mock_client
    
    async def test_set_and_get(self, mock_redis_client):
        """Test basic set and get operations with mocked Redis."""
        with patch('src.infrastructure.cache.redis_cache.redis.Redis') as mock_redis:
            mock_redis.return_value = mock_redis_client
            
            cache = RedisCache(redis_url="redis://localhost:6379/0")
            cache._client = mock_redis_client
            
            # Test set
            result = await cache.set("test_key", "test_value", ttl_seconds=60)
            assert result is True
            mock_redis_client.setex.assert_called_once()
            
            # Test get (mock return value)
            import msgpack
            mock_redis_client.get.return_value = msgpack.packb("test_value", use_bin_type=True)
            
            value = await cache.get("test_key")
            assert value == "test_value"
            mock_redis_client.get.assert_called_with("test_key")
    
    async def test_get_nonexistent_key(self, mock_redis_client):
        """Test getting a non-existent key returns None."""
        cache = RedisCache(redis_url="redis://localhost:6379/0")
        cache._client = mock_redis_client
        
        mock_redis_client.get.return_value = None
        
        value = await cache.get("nonexistent")
        assert value is None
    
    async def test_delete(self, mock_redis_client):
        """Test deleting a key."""
        cache = RedisCache(redis_url="redis://localhost:6379/0")
        cache._client = mock_redis_client
        
        mock_redis_client.delete.return_value = 1
        
        result = await cache.delete("test_key")
        assert result is True
        mock_redis_client.delete.assert_called_with("test_key")
    
    async def test_exists(self, mock_redis_client):
        """Test checking if a key exists."""
        cache = RedisCache(redis_url="redis://localhost:6379/0")
        cache._client = mock_redis_client
        
        mock_redis_client.exists.return_value = 1
        assert await cache.exists("test_key") is True
        
        mock_redis_client.exists.return_value = 0
        assert await cache.exists("nonexistent") is False
    
    async def test_health_check(self, mock_redis_client):
        """Test health check."""
        cache = RedisCache(redis_url="redis://localhost:6379/0")
        cache._client = mock_redis_client
        
        mock_redis_client.ping.return_value = True
        result = await cache.health_check()
        assert result is True
        
        mock_redis_client.ping.side_effect = Exception("Connection failed")
        result = await cache.health_check()
        assert result is False
    
    async def test_msgpack_serialization(self, mock_redis_client):
        """Test msgpack serialization for embeddings."""
        cache = RedisCache(redis_url="redis://localhost:6379/0")
        cache._client = mock_redis_client
        
        # Test embedding vector
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Serialize
        serialized = cache._serialize(embedding)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = cache._deserialize(serialized)
        assert deserialized == embedding
    
    async def test_get_many(self, mock_redis_client):
        """Test getting multiple values."""
        cache = RedisCache(redis_url="redis://localhost:6379/0")
        cache._client = mock_redis_client
        
        import msgpack
        
        # Setup pipeline mock
        mock_pipe = AsyncMock()
        mock_pipe.get = MagicMock(return_value=mock_pipe)
        mock_pipe.execute = AsyncMock(return_value=[
            msgpack.packb("value1", use_bin_type=True),
            msgpack.packb("value2", use_bin_type=True),
            None  # key3 doesn't exist
        ])
        mock_redis_client.pipeline.return_value = mock_pipe
        
        result = await cache.get_many(["key1", "key2", "key3"])
        
        assert "key1" in result
        assert result["key1"] == "value1"
        assert "key2" in result
        assert result["key2"] == "value2"
        assert "key3" not in result
    
    async def test_set_many(self, mock_redis_client):
        """Test setting multiple values."""
        cache = RedisCache(redis_url="redis://localhost:6379/0")
        cache._client = mock_redis_client
        
        # Setup pipeline mock
        mock_pipe = AsyncMock()
        mock_pipe.setex = MagicMock(return_value=mock_pipe)
        mock_pipe.execute = AsyncMock(return_value=[True, True, True])
        mock_redis_client.pipeline.return_value = mock_pipe
        
        mapping = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        
        result = await cache.set_many(mapping, ttl_seconds=60)
        assert result is True
    
    async def test_error_handling(self, mock_redis_client):
        """Test error handling returns False on failures."""
        cache = RedisCache(redis_url="redis://localhost:6379/0")
        cache._client = mock_redis_client
        
        # Simulate error
        mock_redis_client.set.side_effect = Exception("Redis error")
        
        result = await cache.set("test_key", "test_value")
        assert result is False


@pytest.mark.asyncio
class TestCacheKeyGeneration:
    """Tests for cache key generation in embedding service."""
    
    async def test_hash_based_cache_keys(self):
        """Test that cache keys are generated using hash of text."""
        from src.domain.embeddings import EmbeddingService
        from src.core.config_models import EmbeddingConfig
        
        config = EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
            device="cpu"
        )
        
        service = EmbeddingService(config=config, cache_service=None)
        
        # Generate cache keys for same text
        text = "This is a test sentence"
        key1 = service._generate_cache_key(text)
        key2 = service._generate_cache_key(text)
        
        # Keys should be identical for same text
        assert key1 == key2
        
        # Keys should be different for different text
        key3 = service._generate_cache_key("Different text")
        assert key1 != key3
        
        # Keys should include model name
        assert "sentence-transformers/all-MiniLM-L6-v2" in key1


@pytest.mark.asyncio
class TestEmbeddingCacheIntegration:
    """Integration tests for embedding service with cache."""
    
    async def test_embedding_caching_with_memory_cache(self):
        """Test embedding service uses memory cache correctly."""
        from src.domain.embeddings import EmbeddingService
        from src.core.config_models import EmbeddingConfig
        from src.infrastructure.cache import MemoryCache
        
        config = EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
            device="cpu"
        )
        
        cache = MemoryCache(max_size=100, default_ttl=3600)
        service = EmbeddingService(config=config, cache_service=cache)
        
        # Mock the embedding methods to avoid actual model loading
        service._embed_local_batch = MagicMock(return_value=[[0.1] * 384])
        service.use_openai = False
        
        # First call should miss cache
        texts = ["test sentence"]
        await service.embed_texts(texts)
        
        assert service.cache_misses == 1
        assert service.cache_hits == 0
        
        # Second call should hit cache
        await service.embed_texts(texts)
        
        assert service.cache_misses == 1
        assert service.cache_hits == 1
    
    async def test_cache_statistics(self):
        """Test cache statistics tracking."""
        from src.domain.embeddings import EmbeddingService
        from src.core.config_models import EmbeddingConfig
        from src.infrastructure.cache import MemoryCache
        
        config = EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
            device="cpu"
        )
        
        cache = MemoryCache(max_size=100, default_ttl=3600)
        service = EmbeddingService(config=config, cache_service=cache)
        
        # Mock embedding
        service._embed_local_batch = MagicMock(return_value=[[0.1] * 384, [0.2] * 384])
        service.use_openai = False
        
        # Embed texts
        await service.embed_texts(["text1", "text2"])
        await service.embed_texts(["text1", "text2"])  # Should hit cache
        
        stats = service.get_cache_stats()
        
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["total"] == 4
        assert stats["hit_rate_percent"] == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
