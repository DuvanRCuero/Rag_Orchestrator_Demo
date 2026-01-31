"""Common request validators."""

from fastapi import HTTPException, status


def validate_query_not_empty(request) -> None:
    """Validate that query is not empty."""
    if hasattr(request, 'query') and not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty",
        )


def validate_top_k_range(request, min_k: int = 1, max_k: int = 50) -> None:
    """Validate top_k is within acceptable range."""
    if hasattr(request, 'top_k') and request.top_k is not None:
        if request.top_k < min_k or request.top_k > max_k:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"top_k must be between {min_k} and {max_k}",
            )


def validate_session_id_format(session_id: str) -> None:
    """Validate session ID format if provided."""
    if session_id and len(session_id) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID too long (max 100 characters)",
        )
