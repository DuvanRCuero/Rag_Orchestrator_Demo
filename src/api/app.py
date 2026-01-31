import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from src.api.middleware.error_handler import global_exception_handler
from src.api.middleware.logging_middleware import LoggingMiddleware
from src.api.v1.router import api_router
from src.core.config import settings
from src.core.exceptions import RAGException
from src.core.logging import setup_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Setup logging first
    setup_logging()
    
    # Startup
    logger.info(
        "application_starting",
        environment=settings.ENVIRONMENT,
        debug=settings.DEBUG,
        vector_db=settings.VECTOR_DB_TYPE,
        llm_provider=settings.LLM_PROVIDER,
        model=settings.OPENAI_MODEL,
    )

    # Initialize services here if needed
    yield

    # Shutdown
    logger.info("application_shutdown")


def create_application() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="RAG Orchestrator API",
        description="Production-grade RAG system with LangChain and advanced hallucination mitigation",
        version="1.0.0",
        docs_url=None,  # Customize Swagger UI
        redoc_url="/redoc",
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan,
        debug=settings.DEBUG,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add logging middleware
    app.add_middleware(LoggingMiddleware)

    # Add exception handlers
    app.add_exception_handler(RAGException, global_exception_handler)
    app.add_exception_handler(HTTPException, global_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)

    # Include API routers
    app.include_router(api_router, prefix="/api/v1")

    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title="RAG Orchestrator API",
            version="1.0.0",
            description="""
            ## Production RAG System Features:

            ### Advanced Retrieval
            - Multi-query generation for better recall
            - HyDE (Hypothetical Document Embeddings)
            - Cross-encoder reranking
            - Hybrid search (semantic + BM25)

            ### Hallucination Mitigation
            - Chain-of-Verification
            - Self-reflection critique
            - Confidence scoring
            - Source citations

            ### Conversation Management
            - Multi-turn conversations
            - Session-based memory
            - Context window management

            ### Monitoring & Evaluation
            - Performance metrics
            - Quality scoring
            - Token usage tracking
            """,
            routes=app.routes,
        )

        # Customize Swagger UI
        openapi_schema["tags"] = [
            {
                "name": "ingestion",
                "description": "Document ingestion and processing endpoints",
            },
            {"name": "query", "description": "Query and conversation endpoints"},
            {"name": "sessions", "description": "Conversation session management"},
            {"name": "monitoring", "description": "System monitoring and metrics"},
            {"name": "evaluation", "description": "RAG evaluation and testing"},
        ]

        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
            }
        }

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "RAG Orchestrator API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui():
        return get_swagger_ui_html(
            openapi_url="/api/v1/openapi.json",
            title="RAG Orchestrator API - Swagger UI",
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
            swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
        )

    return app


async def log_requests(request: Request, call_next):
    """Middleware to log requests."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # Bind request context to logger
    request_logger = logger.bind(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )

    request_logger.info("request_started")

    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000

        request_logger.info(
            "request_completed",
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(round(duration_ms, 2))

        return response

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        request_logger.exception(
            "request_failed",
            duration_ms=round(duration_ms, 2),
            error=str(e),
        )
        raise


# Create application instance
app = create_application()
