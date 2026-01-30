import platform
from datetime import datetime
from typing import Any, Dict

import psutil
from fastapi import APIRouter, HTTPException, status

from src.api.v1.dependencies import get_vector_store
from src.core.config import settings

router = APIRouter()


@router.get(
    "/health",
    summary="Health check",
    description="Check the health status of all system components.",
)
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "RAG Orchestrator API",
            "version": "1.0.0",
        }

        # Check system resources
        system_info = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }

        health_status["system"] = system_info

        # Check vector store
        try:
            vector_store = get_vector_store()
            vector_stats = await vector_store.get_collection_stats()
            health_status["vector_store"] = {
                "status": "healthy",
                "vectors_count": vector_stats.get("vectors_count", 0),
                "collection": vector_stats.get("config", {}),
            }
        except Exception as e:
            health_status["vector_store"] = {"status": "unhealthy", "error": str(e)}
            health_status["status"] = "degraded"

        # Check embedding service
        health_status["embedding_service"] = {
            "status": "healthy",
            "model": settings.EMBEDDING_MODEL,
        }

        # Check LLM service
        health_status["llm_service"] = {
            "status": "healthy" if settings.OPENAI_API_KEY else "unconfigured",
            "provider": settings.LLM_PROVIDER,
            "model": settings.OPENAI_MODEL,
        }

        # Determine overall status
        if health_status["vector_store"]["status"] == "unhealthy":
            health_status["status"] = "unhealthy"

        return health_status

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}",
        )


@router.get(
    "/metrics",
    summary="Get system metrics",
    description="Get detailed system metrics and performance statistics.",
)
async def get_metrics():
    """Get system metrics."""
    try:
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu": {
                    "percent": psutil.cpu_percent(interval=0.1),
                    "count": psutil.cpu_count(logical=True),
                    "freq": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                },
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent,
                    "used": psutil.virtual_memory().used,
                },
                "disk": {
                    "total": psutil.disk_usage("/").total,
                    "used": psutil.disk_usage("/").used,
                    "free": psutil.disk_usage("/").free,
                    "percent": psutil.disk_usage("/").percent,
                },
            },
            "api": {
                "requests_served": 0,  # Would track with middleware
                "average_response_time": 0.0,
                "error_rate": 0.0,
            },
            "rag": {
                "total_queries": 0,
                "average_retrieval_time": 0.0,
                "average_generation_time": 0.0,
                "average_confidence": 0.0,
            },
        }

        # Add vector store metrics
        try:
            vector_store = get_vector_store()
            vector_stats = await vector_store.get_collection_stats()
            metrics["vector_store"] = vector_stats
        except:
            metrics["vector_store"] = {"status": "unavailable"}

        return metrics

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}",
        )


@router.get(
    "/config",
    summary="Get configuration",
    description="Get current system configuration (excluding secrets).",
)
async def get_configuration():
    """Get system configuration."""
    try:
        config = {
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
            "vector_db": {
                "type": settings.VECTOR_DB_TYPE,
                "collection": settings.QDRANT_COLLECTION_NAME,
                "url": settings.QDRANT_URL
                if settings.ENVIRONMENT != "production"
                else "[REDACTED]",
            },
            "embeddings": {
                "model": settings.EMBEDDING_MODEL,
                "dimension": settings.EMBEDDING_DIMENSION,
                "device": settings.EMBEDDING_DEVICE,
            },
            "llm": {
                "provider": settings.LLM_PROVIDER,
                "model": settings.OPENAI_MODEL,
                "temperature": settings.OPENAI_TEMPERATURE,
                "max_tokens": settings.OPENAI_MAX_TOKENS,
            },
            "text_processing": {
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "splitter": settings.TEXT_SPLITTER,
            },
            "retrieval": {
                "top_k": settings.RETRIEVAL_TOP_K,
                "score_threshold": settings.RETRIEVAL_SCORE_THRESHOLD,
                "hybrid_search": settings.USE_HYBRID_SEARCH,
                "bm25_weight": settings.BM25_WEIGHT,
                "semantic_weight": settings.SEMANTIC_WEIGHT,
            },
            "memory": {
                "type": settings.MEMORY_TYPE,
                "window_size": settings.MEMORY_WINDOW_SIZE,
            },
        }

        return config

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get configuration: {str(e)}",
        )
