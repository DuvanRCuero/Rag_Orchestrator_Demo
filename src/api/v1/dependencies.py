from typing import Any, Dict

from fastapi import Depends, HTTPException, status

from src.infrastructure.container import container
from src.core.config import settings


def get_rag_chain():
    """Dependency to get RAG chain instance."""
    return container.rag_chain


def get_vector_store():
    """Dependency to get vector store instance."""
    return container.vector_store


def get_embedding_service():
    """Dependency to get embedding service."""
    return container.embedding_service


def get_conversation_memory():
    """Dependency to get conversation memory."""
    return container.conversation_memory


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


def get_ask_question_use_case():
    """Dependency to get ask question use case."""
    return container.ask_question_use_case


def get_stream_answer_use_case():
    """Dependency to get stream answer use case."""
    return container.stream_answer_use_case
