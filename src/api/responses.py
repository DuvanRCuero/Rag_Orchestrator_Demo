"""Common response builders for DRY responses."""

from typing import Any, Dict, List, Optional
from datetime import datetime


def success_response(
    data: Any = None,
    message: str = "Success",
    **extras,
) -> Dict[str, Any]:
    """Build a standard success response."""
    response = {
        "status": "success",
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if data is not None:
        response["data"] = data
    response.update(extras)
    return response


def error_response(
    message: str,
    code: str = "ERROR",
    details: Any = None,
) -> Dict[str, Any]:
    """Build a standard error response."""
    response = {
        "status": "error",
        "code": code,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if details is not None:
        response["details"] = details
    return response


def paginated_response(
    items: List[Any],
    total: int,
    page: int = 1,
    page_size: int = 10,
) -> Dict[str, Any]:
    """Build a paginated response."""
    return {
        "status": "success",
        "data": items,
        "pagination": {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "has_next": page * page_size < total,
            "has_prev": page > 1,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
