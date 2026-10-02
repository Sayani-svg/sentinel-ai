"""Database engine and session factory for Sentinel AI.

The FastAPI ``get_db`` dependency is *not* defined here. It is re-exported from
:mod:`app.core.dependencies`, which is where routers and tests reference it, so
that ``app.database.get_db`` and ``app.core.dependencies.get_db`` are the same
function object rather than two implementations that have to be kept in step.
Two copies is how a test ends up overriding the dependency the router does not
use, and silently keeps the production engine out of reach of the suite instead
of in reach of it.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.core.dependencies import get_db as get_db


def create_database_engine(settings: Settings) -> Engine:
    """Create a configured synchronous SQLAlchemy engine.

    Pool sizing and recycling come from application Settings. ``pool_pre_ping``
    is enabled so stale PostgreSQL connections are discarded before use.

    Args:
        settings: Application settings containing the database URL and pool
            configuration.

    Returns:
        Engine: Synchronous SQLAlchemy engine bound to the configured
        PostgreSQL database.
    """
    return create_engine(
        settings.DATABASE_URL,
        echo=settings.DATABASE_ECHO,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_timeout=settings.DATABASE_POOL_TIMEOUT,
        pool_recycle=settings.DATABASE_POOL_RECYCLE,
        pool_pre_ping=True,
    )


_settings: Settings = get_settings()
engine: Engine = create_database_engine(_settings)
SessionLocal: sessionmaker[Session] = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)
