"""Prediction ORM model."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class Prediction(Base):
    """Persisted attack classification prediction.

    ``severity`` is plain text rather than an enum column. This schema does not
    use database enum types, and a ``Mapped[Literal[...]]`` annotation is
    resolved by SQLAlchemy into :class:`sqlalchemy.Enum`, which brings the type
    into the metadata, into Alembic autogenerate output, and -- through its
    ``validate_strings`` bind processor -- rejects unexpected values with a
    ``LookupError`` at write time.

    The permitted values are ``Critical``, ``High``, ``Medium`` and ``Low``.
    Validating them belongs to the prediction slice, which will declare them the
    same way :mod:`app.schemas.user` declares the user roles: as a ``StrEnum``
    on the request and response schemas, with the value set checked when a
    prediction is created rather than by the database.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    upload_id: Mapped[int] = mapped_column(ForeignKey("uploaded_logs.id"))
    attack_type: Mapped[str]
    confidence: Mapped[float]
    severity: Mapped[str]
    created_at: Mapped[datetime]
