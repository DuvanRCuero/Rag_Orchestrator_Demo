"""Build consistent cache keys with namespacing."""

import hashlib


class CacheKeyBuilder:
    """Build consistent cache keys with namespacing."""

    EMBEDDING_PREFIX = "emb"
    QUERY_PREFIX = "qry"
    SESSION_PREFIX = "sess"

    @staticmethod
    def embedding_key(text: str, model: str) -> str:
        """Build cache key for embeddings."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{CacheKeyBuilder.EMBEDDING_PREFIX}:{model}:{text_hash}"

    @staticmethod
    def query_result_key(query: str, top_k: int, params_hash: str) -> str:
        """Build cache key for query results."""
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"{CacheKeyBuilder.QUERY_PREFIX}:{query_hash}:{top_k}:{params_hash}"

    @staticmethod
    def session_key(session_id: str) -> str:
        """Build cache key for sessions."""
        return f"{CacheKeyBuilder.SESSION_PREFIX}:{session_id}"
