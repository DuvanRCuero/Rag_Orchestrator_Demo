"""API decorators for DRY error handling."""

from functools import wraps
from typing import Callable, Type, Union, List

from fastapi import HTTPException, status

from src.core.exceptions import RAGException
from src.core.logging import get_logger

logger = get_logger(__name__)


def handle_exceptions(
    operation_name: str,
    error_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    reraise_types: List[Type[Exception]] = None,
):
    """
    Decorator for consistent exception handling in endpoints.
    
    Args:
        operation_name: Human-readable name for the operation
        error_status: HTTP status code for unhandled exceptions
        reraise_types: Exception types to reraise without wrapping
    """
    reraise_types = reraise_types or [HTTPException, RAGException]
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except tuple(reraise_types):
                raise
            except Exception as e:
                logger.error(
                    "endpoint_error",
                    operation=operation_name,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=error_status,
                    detail=f"{operation_name} failed: {str(e)}",
                )
        return wrapper
    return decorator


def validate_request(validators: List[Callable]):
    """
    Decorator for request validation.
    
    Args:
        validators: List of validation functions that raise on failure
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for validator in validators:
                validator(*args, **kwargs)
            return await func(*args, **kwargs)
        return wrapper
    return decorator
