#!/usr/bin/env python3
"""
RAG Orchestrator API - Production Entry Point
"""

import uvicorn
from src.api.app import app


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
        workers=4
    )