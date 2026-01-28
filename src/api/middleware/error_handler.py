import traceback
from typing import Union

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from src.core.exceptions import RAGException


async def global_exception_handler(
    request: Request, exc: Union[Exception, RAGException]
):
    """Global exception handler for all exceptions."""

    if isinstance(exc, RAGException):
        # Handle custom RAG exceptions
        error_response = {
            "error": {
                "code": exc.error_code,
                "message": exc.detail,
                "type": exc.__class__.__name__,
                "metadata": exc.metadata,
                "timestamp": request.state.timestamp
                if hasattr(request.state, "timestamp")
                else None,
            }
        }
        return JSONResponse(status_code=exc.status_code, content=error_response)

    elif isinstance(exc, HTTPException):
        # Handle FastAPI HTTP exceptions
        error_response = {
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "type": exc.__class__.__name__,
                "timestamp": request.state.timestamp
                if hasattr(request.state, "timestamp")
                else None,
            }
        }
        return JSONResponse(status_code=exc.status_code, content=error_response)

    else:
        # Handle unexpected exceptions
        error_response = {
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "type": exc.__class__.__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc() if request.app.debug else None,
                "timestamp": request.state.timestamp
                if hasattr(request.state, "timestamp")
                else None,
            }
        }
        return JSONResponse(status_code=500, content=error_response)
