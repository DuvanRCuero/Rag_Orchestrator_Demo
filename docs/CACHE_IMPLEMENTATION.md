# Redis Cache Layer Documentation

## Overview

This document describes the Redis caching layer implementation for the RAG Orchestrator, which provides persistent, distributed caching for embeddings and query results.

## Architecture

### Cache Service Interface

The cache layer is built around the `CacheService` interface defined in `src/domain/interfaces/cache_service.py`. This interface provides:

- **Async operations**: All operations are async for non-blocking I/O
- **TTL management**: Configurable time-to-live for cache entries
- **Batch operations**: `get_many` and `set_many` for efficient bulk operations
- **Health checks**: Monitor cache service availability

### Implementations

#### 1. Redis Cache (`RedisCache`)

**Location**: `src/infrastructure/cache/redis_cache.py`

**Features**:
- Connection pooling for efficient resource usage
- msgpack serialization for compact storage of embeddings
- Automatic TTL management
- Graceful error handling
- Pipeline operations for bulk requests

**Configuration**:
```python
CACHE_TYPE=redis
REDIS_URL=redis://localhost:6379/0
CACHE_MAX_CONNECTIONS=10
```

#### 2. Memory Cache (`MemoryCache`)

**Location**: `src/infrastructure/cache/memory_cache.py`

**Features**:
- Uses `cachetools.TTLCache` for automatic expiration
- Thread-safe with async locks
- Fallback when Redis is unavailable
- Per-key TTL support

**Configuration**:
```python
CACHE_TYPE=memory
```

## Configuration

### Environment Variables

Add these to your `.env` file:

```env
# Cache Configuration
CACHE_TYPE=redis                    # Options: redis, memory
REDIS_URL=redis://localhost:6379/0  # Redis connection URL
CACHE_EMBEDDING_TTL=604800          # 7 days in seconds
CACHE_QUERY_TTL=3600                # 1 hour in seconds
CACHE_SESSION_TTL=1800              # 30 minutes in seconds
CACHE_MAX_CONNECTIONS=10            # Redis connection pool size
```

### Cache TTL Strategy

| Cache Type | TTL | Rationale |
|------------|-----|-----------|
| Embeddings | 7 days | Embeddings are expensive to compute and rarely change |
| Query Results | 1 hour | Results may become stale as documents are updated |
| Session Data | 30 minutes | User sessions are short-lived |

## Usage

### Embedding Service Integration

The `EmbeddingService` automatically uses the cache service when initialized:

```python
from src.domain.embeddings import EmbeddingService
from src.infrastructure.cache import RedisCache

# Create cache service
cache = RedisCache(redis_url="redis://localhost:6379/0")

# Create embedding service with cache
embedding_service = EmbeddingService(config, cache_service=cache)

# Embeddings are automatically cached
embeddings = await embedding_service.embed_texts(["text1", "text2"])

# Get cache statistics
stats = embedding_service.get_cache_stats()
print(f"Hit rate: {stats['hit_rate_percent']}%")
```

### Retrieval Service Integration

The `RetrievalService` caches query results:

```python
from src.application.services import RetrievalService

# Create retrieval service with cache
retrieval_service = RetrievalService(
    vector_store=vector_store,
    embedding_service=embedding_service,
    cache_service=cache
)

# Query results are automatically cached
result = await retrieval_service.retrieve("query", top_k=5)

# Invalidate cache when documents are updated
await retrieval_service.invalidate_cache()
```

### Dependency Injection

The cache service is automatically configured in the DI container:

```python
from src.infrastructure.container import container

# Cache service is automatically initialized based on CACHE_TYPE
cache_service = container.cache_service
embedding_service = container.embedding_service  # Uses cache
retrieval_service = container.retrieval_service  # Uses cache
```

## Cache Key Generation

### Embedding Cache Keys

Format: `embed:{model_name}:{sha256_hash}`

Example:
```
embed:sentence-transformers/all-MiniLM-L6-v2:a4b3c2d1e5f6...
```

Benefits:
- Model-specific caching prevents collisions
- SHA-256 hash ensures uniqueness
- Consistent keys for identical text

### Query Result Cache Keys

Format: `query:{sha256_hash}`

The hash includes:
- Query text
- top_k parameter
- use_hybrid flag
- Applied filters

This ensures cache hits only for identical queries with same parameters.

## Monitoring

### Cache Statistics

Both embedding and retrieval services track cache performance:

```python
# Embedding service stats
stats = embedding_service.get_cache_stats()
# Returns: {
#   "hits": 150,
#   "misses": 50,
#   "total": 200,
#   "hit_rate_percent": 75.0
# }

# Retrieval service stats
stats = retrieval_service.get_cache_stats()
```

### Health Checks

Check cache service health:

```python
is_healthy = await cache_service.health_check()
```

## Docker Deployment

### Redis Configuration

The `docker-compose.yml` includes Redis with persistence:

```yaml
redis:
  image: redis:7.2-alpine
  ports:
    - "6379:6379"
  volumes:
    - redis_data:/data
    - ./docker/redis/redis.conf:/usr/local/etc/redis/redis.conf:ro
  command: redis-server /usr/local/etc/redis/redis.conf
```

### Starting Services

```bash
# Start all services including Redis
docker-compose up -d

# Check Redis health
docker-compose ps redis

# View Redis logs
docker-compose logs -f redis
```

## Performance Considerations

### Embedding Cache

- **Before**: ~100ms per embedding (OpenAI API)
- **After**: ~1ms (cache hit)
- **Expected hit rate**: 60-80% in production

### Query Result Cache

- **Before**: ~200-500ms per query (vector search + embedding)
- **After**: ~5ms (cache hit)
- **Expected hit rate**: 30-50% in production

### Memory Usage

Typical cache sizes:
- Embedding: ~1.5KB per entry (384-dim float32)
- Query result: ~5-10KB per entry (5 document chunks)

With 10,000 cached embeddings:
- Memory: ~15MB
- Redis recommended: 1GB RAM minimum

## Failover and Resilience

### Automatic Fallback

If Redis becomes unavailable, the system automatically falls back to in-memory cache:

```python
# In container.py
try:
    cache_service = RedisCache(redis_url=config.redis_url)
except Exception:
    logger.warning("Redis unavailable, falling back to memory cache")
    cache_service = MemoryCache(max_size=10000)
```

### Graceful Degradation

All cache operations handle failures gracefully:
- Failed `get` returns `None` (cache miss)
- Failed `set` returns `False` (continues without caching)
- Application continues working with slower performance

## Testing

### Unit Tests

Run cache tests:

```bash
pytest tests/unit/test_cache_service.py -v
```

Test coverage includes:
- Memory cache operations
- Redis cache with mocked client
- TTL expiration
- Serialization/deserialization
- Cache key generation
- Integration with embedding service

### Integration Tests

Test with real Redis:

```bash
# Start Redis
docker-compose up -d redis

# Run integration tests
CACHE_TYPE=redis pytest tests/integration/test_cache_integration.py
```

## Troubleshooting

### Redis Connection Errors

**Symptom**: `ConnectionError: Error connecting to Redis`

**Solutions**:
1. Check Redis is running: `docker-compose ps redis`
2. Verify REDIS_URL is correct
3. Check firewall/network settings
4. System will fall back to memory cache automatically

### High Memory Usage

**Symptom**: Redis using excessive memory

**Solutions**:
1. Lower TTL values to expire entries faster
2. Reduce `max_connections` in cache config
3. Implement cache size limits
4. Use Redis `maxmemory` policy in config

### Cache Inconsistency

**Symptom**: Stale results after document updates

**Solutions**:
1. Call `retrieval_service.invalidate_cache()` after updates
2. Lower `CACHE_QUERY_TTL` for faster expiration
3. Implement selective cache invalidation by document ID

## Future Enhancements

Potential improvements:

1. **Cache Warming**: Pre-populate cache with common queries
2. **Selective Invalidation**: Invalidate only affected queries when documents change
3. **Cache Sharding**: Distribute cache across multiple Redis instances
4. **Compression**: Further reduce storage with compression
5. **Cache Metrics**: Export metrics to Prometheus/Grafana
6. **Cache Replication**: Redis replication for high availability

## References

- [Redis Documentation](https://redis.io/docs/)
- [cachetools Documentation](https://cachetools.readthedocs.io/)
- [msgpack Documentation](https://msgpack.org/)
