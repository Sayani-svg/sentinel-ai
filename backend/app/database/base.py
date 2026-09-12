"""SQLAlchemy declarative base for Sentinel AI ORM models."""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Canonical declarative base class for all SQLAlchemy ORM models.

    All backend models must subclass this type so they share a single
    metadata registry. Model modules are not imported here; registration
    is handled by the models package.

    Attributes:
        metadata: Shared SQLAlchemy ``MetaData`` used for table mapping
            and Alembic schema operations.
    """

    __abstract__ = True
