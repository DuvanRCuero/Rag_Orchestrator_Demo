from typing import Any, Dict

from fastapi import HTTPException, status


class RAGException(HTTPException):
    """Base exception for RAG system."""

    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: str = "An error occurred",
        error_code: str = "RAG_ERROR",
        metadata: Dict[str, Any] = None,
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.metadata = metadata or {}


class IngestionError(RAGException):
    """Raised when document ingestion fails."""

    def __init__(self, detail: str = "Document ingestion failed", **kwargs):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="INGESTION_ERROR",
            **kwargs,
        )


class RetrievalError(RAGException):
    """Raised when retrieval fails."""

    def __init__(self, detail: str = "Failed to retrieve relevant documents", **kwargs):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            error_code="RETRIEVAL_ERROR",
            **kwargs,
        )


class GenerationError(RAGException):
    """Raised when LLM generation fails."""

    def __init__(self, detail: str = "Failed to generate answer", **kwargs):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code="GENERATION_ERROR",
            **kwargs,
        )


class ValidationError(RAGException):
    """Raised when input validation fails."""

    def __init__(self, detail: str = "Invalid input", **kwargs):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="VALIDATION_ERROR",
            **kwargs,
        )
