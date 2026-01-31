#!/usr/bin/env python3
"""Standalone test for cache infrastructure."""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set environment variables
os.environ["ENVIRONMENT"] = "testing"
os.environ["OPENAI_API_KEY"] = "test-key"

from src.infrastructure.cache import (
    InMemoryCache,
    CacheKeyBuilder,
)


async def test_cache_key_builder():
    """Test cache key builder."""
    print("Testing CacheKeyBuilder...")
    
    # Test embedding key
    key = CacheKeyBuilder.embedding_key("test text", "model-v1")
    assert key.startswith("emb:model-v1:"), f"Expected key to start with 'emb:model-v1:', got {key}"
    assert len(key.split(":")[-1]) == 16, "Hash should be 16 characters"
    
    # Test consistency
    key1 = CacheKeyBuilder.embedding_key("test", "model")
    key2 = CacheKeyBuilder.embedding_key("test", "model")
    assert key1 == key2, "Same text should produce same key"
    
    # Test query result key
    query_key = CacheKeyBuilder.query_result_key("query", 5, "params")
    assert query_key.startswith("qry:"), f"Expected query key to start with 'qry:', got {query_key}"
    
    # Test session key
    session_key = CacheKeyBuilder.session_key("session-123")
    assert session_key == "sess:session-123", f"Expected 'sess:session-123', got {session_key}"
    
    print("✓ CacheKeyBuilder tests passed")


async def test_in_memory_cache():
    """Test in-memory cache."""
    print("\nTesting InMemoryCache...")
    
    cache = InMemoryCache()
    
    # Test set and get
    await cache.set("key1", "value1")
    value = await cache.get("key1")
    assert value == "value1", f"Expected 'value1', got {value}"
    
    # Test get nonexistent
    value = await cache.get("nonexistent")
    assert value is None, f"Expected None, got {value}"
    
    # Test delete
    await cache.set("key2", "value2")
    result = await cache.delete("key2")
    assert result is True, "Delete should return True"
    value = await cache.get("key2")
    assert value is None, "Deleted key should return None"
    
    # Test exists
    await cache.set("key3", "value3")
    assert await cache.exists("key3") is True, "Key should exist"
    assert await cache.exists("nonexistent") is False, "Nonexistent key should not exist"
    
    # Test mget
    await cache.set("key4", "value4")
    await cache.set("key5", "value5")
    result = await cache.mget(["key4", "key5", "key6"])
    assert result == {"key4": "value4", "key5": "value5"}, f"Expected dict with key4 and key5, got {result}"
    
    # Test mset
    items = {"key7": "value7", "key8": "value8"}
    result = await cache.mset(items)
    assert result is True, "mset should return True"
    assert await cache.get("key7") == "value7", "key7 should be set"
    assert await cache.get("key8") == "value8", "key8 should be set"
    
    # Test stats
    stats = await cache.get_stats()
    assert "hit_rate" in stats, "Stats should include hit_rate"
    assert "total_hits" in stats, "Stats should include total_hits"
    assert "total_misses" in stats, "Stats should include total_misses"
    assert "keys_count" in stats, "Stats should include keys_count"
    assert stats["keys_count"] >= 4, f"Expected at least 4 keys, got {stats['keys_count']}"
    
    # Test close
    await cache.close()
    
    print("✓ InMemoryCache tests passed")


async def test_cache_with_embeddings():
    """Test cache with embedding-like data."""
    print("\nTesting cache with embeddings...")
    
    cache = InMemoryCache()
    
    # Simulate embedding storage
    text = "This is a test sentence"
    model = "test-model-v1"
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    key = CacheKeyBuilder.embedding_key(text, model)
    await cache.set(key, embedding, ttl=3600)
    
    # Retrieve embedding
    cached_embedding = await cache.get(key)
    assert cached_embedding == embedding, f"Expected {embedding}, got {cached_embedding}"
    
    # Test batch operations with embeddings
    texts = ["text1", "text2", "text3"]
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]
    
    items = {
        CacheKeyBuilder.embedding_key(text, model): emb
        for text, emb in zip(texts, embeddings)
    }
    
    await cache.mset(items, ttl=3600)
    
    # Retrieve batch
    keys = [CacheKeyBuilder.embedding_key(text, model) for text in texts]
    cached_items = await cache.mget(keys)
    
    assert len(cached_items) == 3, f"Expected 3 cached items, got {len(cached_items)}"
    
    print("✓ Cache with embeddings tests passed")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Cache Infrastructure Tests")
    print("=" * 60)
    
    try:
        await test_cache_key_builder()
        await test_in_memory_cache()
        await test_cache_with_embeddings()
        
        print("\n" + "=" * 60)
        print("✓ All cache tests passed!")
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
