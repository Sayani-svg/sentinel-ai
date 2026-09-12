"""Database engine and session factory for Sentinel AI."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings


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


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and close it after the caller finishes.

    The lifecycle matches the FastAPI ``get_db`` dependency: open a session
    from ``SessionLocal``, yield it to the caller, then close it.

    Yields:
        Session: A SQLAlchemy ORM session bound to the application engine.
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
