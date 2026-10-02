"""User ORM model."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class User(Base):
    """Persisted application user.

    ``role`` is plain text rather than an enum column. This schema does not use
    database enum types, and a ``Mapped[Literal[...]]`` annotation is resolved
    by SQLAlchemy into :class:`sqlalchemy.Enum`, which brings the type into the
    metadata, into Alembic autogenerate output, and -- through its
    ``validate_strings`` bind processor -- rejects unexpected values with a
    ``LookupError`` at write time.

    The permitted values are defined by :class:`app.schemas.user.UserRole` and
    enforced by the application: registration always writes
    :attr:`~app.schemas.user.UserRole.VIEWER`, and
    :func:`app.core.dependencies.get_current_active_user` refuses any account
    whose stored role is not recognised before a route can act on it.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    email: Mapped[str] = mapped_column(unique=True)
    password_hash: Mapped[str]
    role: Mapped[str]
    created_at: Mapped[datetime]
