"""Public database package exports for Sentinel AI.

``Base`` is re-exported eagerly because it is pure declarative metadata and
carries no configuration dependency. ``SessionLocal``, ``engine`` and
``get_db`` originate from :mod:`app.database.session`, whose module scope
builds the engine from ``Settings.DATABASE_URL``; those three are therefore
resolved lazily through :pep:`562` module ``__getattr__``.

Without this deferral, touching ``app.database.base`` -- which every ORM model
and Alembic's ``env.py`` do -- would execute the parent package first and
transitively import :mod:`app.database.session`, forcing ``DATABASE_URL`` and
``SECRET_KEY`` to be present at import time for metadata-only consumers such
as ``alembic revision --autogenerate``.

``get_db`` is re-exported here for backwards compatibility only. Its single
definition is in :mod:`app.core.dependencies`, so the name obtained from this
package is that same function object; resolving it through the lazy hook below
keeps this package importable without configuring settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.database.base import Base

if TYPE_CHECKING:
    from app.database.session import SessionLocal as SessionLocal
    from app.database.session import engine as engine
    from app.database.session import get_db as get_db

__all__ = [
    "Base",
    "SessionLocal",
    "engine",
    "get_db",
]

_SESSION_EXPORTS = frozenset({"SessionLocal", "engine", "get_db"})


def __getattr__(name: str) -> Any:
    """Import and return the session attribute that defines ``name``.

    Args:
        name: Attribute name requested on the ``app.database`` package.

    Returns:
        Any: The requested export from :mod:`app.database.session`.

    Raises:
        AttributeError: If the name is not a public database export.
    """
    if name not in _SESSION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from app.database import session

    return getattr(session, name)


def __dir__() -> list[str]:
    """Return the public export names for introspection.

    Returns:
        list[str]: Sorted list of public ``app.database`` export names.
    """
    return sorted(__all__)
