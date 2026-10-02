"""Aggregator for the version 1 API surface.

Only the routers that exist are mounted, so this package stays importable while
the remaining endpoints are still unimplemented.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import auth, inference, reports, upload, users

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users.router, prefix="/users", tags=["Users"])
api_router.include_router(upload.router, prefix="/upload", tags=["Log Ingestion"])
api_router.include_router(inference.router, prefix="/predict", tags=["Detection"])
api_router.include_router(reports.router, prefix="/reports", tags=["Reports"])

__all__ = ["api_router"]
