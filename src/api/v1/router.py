from fastapi import APIRouter

from src.api.v1.endpoints import (evaluation, ingest, monitoring, query,
                                  sessions)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
api_router.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
api_router.include_router(evaluation.router, prefix="/evaluate", tags=["evaluation"])
