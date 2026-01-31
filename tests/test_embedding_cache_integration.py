#!/usr/bin/env python3
"""Test EmbeddingService with cache integration."""

import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set environment variables
os.environ["ENVIRONMENT"] = "testing"
os.environ["OPENAI_API_KEY"] = "test-key"

from src.infrastructure.cache import InMemoryCache
from src.core.config_models import EmbeddingConfig


async def test_embedding_service_with_cache():
    """Test EmbeddingService with cache integration."""
    print("\nTesting EmbeddingService with cache...")
    
    # Import after setting env vars
    from src.domain.embeddings import EmbeddingService
    
    # Create cache
    cache = InMemoryCache()
    
    # Create config
    config = EmbeddingConfig(
        model="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        device="cpu"
    )
    
    # Mock the sentence transformer model
    embedding_service = EmbeddingService(config=config, cache=cache)
    
    # Mock the model's encode method
    mock_model = MagicMock()
    mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    mock_model.encode.return_value = type('obj', (object,), {'tolist': lambda: mock_embeddings})()
    embedding_service._model = mock_model
    
    # Test embedding with cache miss
    texts = ["hello world", "test text"]
    embeddings = await embedding_service.embed_texts(texts)
    
    assert len(embeddings) == 2, f"Expected 2 embeddings, got {len(embeddings)}"
    assert mock_model.encode.called, "Model should be called on cache miss"
    
    # Reset mock
    mock_model.encode.reset_mock()
    
    # Test embedding with cache hit (same texts)
    embeddings2 = await embedding_service.embed_texts(texts)
    
    assert len(embeddings2) == 2, f"Expected 2 embeddings, got {len(embeddings2)}"
    assert not mock_model.encode.called, "Model should NOT be called on cache hit"
    assert embeddings == embeddings2, "Cached embeddings should match"
    
    # Check cache stats
    stats = await cache.get_stats()
    print(f"Cache stats: {stats}")
    assert stats["total_hits"] > 0, "Cache should have hits"
    
    print("✓ EmbeddingService with cache tests passed")


async def test_embedding_service_two_level_cache():
    """Test two-level cache (L1 in-memory + L2 Redis)."""
    print("\nTesting two-level cache...")
    
    from src.domain.embeddings import EmbeddingService
    
    # Create L2 cache (Redis mock)
    l2_cache = InMemoryCache()  # Using InMemoryCache to simulate Redis
    
    # Pre-populate L2 cache
    from src.infrastructure.cache import CacheKeyBuilder
    
    text = "cached in L2"
    model = "test-model"
    key = CacheKeyBuilder.embedding_key(text, model)
    l2_embedding = [0.9, 0.8, 0.7]
    await l2_cache.set(key, l2_embedding)
    
    # Create embedding service with L2 cache
    config = EmbeddingConfig(
        model=model,
        dimension=384,
        device="cpu"
    )
    
    embedding_service = EmbeddingService(config=config, cache=l2_cache)
    
    # Mock the model
    mock_model = MagicMock()
    embedding_service._model = mock_model
    
    # Test embedding - should hit L2 cache
    embeddings = await embedding_service.embed_texts([text])
    
    assert len(embeddings) == 1, f"Expected 1 embedding, got {len(embeddings)}"
    assert embeddings[0] == l2_embedding, f"Expected {l2_embedding}, got {embeddings[0]}"
    assert not mock_model.encode.called, "Model should NOT be called on L2 cache hit"
    
    # Check that it's now in L1 cache too
    assert key in embedding_service._local_cache, "Should be in L1 cache now"
    assert embedding_service._local_cache[key] == l2_embedding, "L1 cache should match L2"
    
    print("✓ Two-level cache tests passed")


async def test_cache_fallback_on_error():
    """Test graceful degradation when cache fails."""
    print("\nTesting cache fallback on error...")
    
    from src.domain.embeddings import EmbeddingService
    
    # Create a mock cache that fails
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(side_effect=Exception("Redis connection failed"))
    mock_cache.mset = AsyncMock(side_effect=Exception("Redis connection failed"))
    
    # Create embedding service
    config = EmbeddingConfig(
        model="test-model",
        dimension=384,
        device="cpu"
    )
    
    embedding_service = EmbeddingService(config=config, cache=mock_cache)
    
    # Mock the model
    mock_model = MagicMock()
    mock_embeddings = [[0.1, 0.2, 0.3]]
    mock_model.encode.return_value = type('obj', (object,), {'tolist': lambda: mock_embeddings})()
    embedding_service._model = mock_model
    
    # Test embedding - should work despite cache failure
    texts = ["test"]
    embeddings = await embedding_service.embed_texts(texts)
    
    assert len(embeddings) == 1, f"Expected 1 embedding, got {len(embeddings)}"
    assert mock_model.encode.called, "Model should be called when cache fails"
    
    print("✓ Cache fallback tests passed")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Running EmbeddingService Cache Integration Tests")
    print("=" * 60)
    
    try:
        await test_embedding_service_with_cache()
        await test_embedding_service_two_level_cache()
        await test_cache_fallback_on_error()
        
        print("\n" + "=" * 60)
        print("✓ All integration tests passed!")
        print("=" * 60)
        return 0
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
