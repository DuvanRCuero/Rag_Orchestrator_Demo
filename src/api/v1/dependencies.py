from typing import Any, Dict

from fastapi import Depends, HTTPException, status

from src.application.chains.memory import conversation_memory
from src.application.chains.rag_chain import rag_chain
from src.core.config import settings
from src.infrastructure.vector.qdrant_client import QdrantVectorStore


def get_rag_chain():
    """Dependency to get RAG chain instance."""
    return rag_chain


def get_vector_store():
    """Dependency to get vector store instance."""
    return QdrantVectorStore()


def get_conversation_memory():
    """Dependency to get conversation memory."""
    return conversation_memory


def validate_api_key(api_key: str = None):
    """API key validation (simplified for now)."""
    if settings.ENVIRONMENT == "production" and not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required"
        )
    return True


def get_session_id(session_id: str = None):
    """Get or generate session ID."""
    if not session_id:
        import uuid

        session_id = str(uuid.uuid4())
    return session_id
