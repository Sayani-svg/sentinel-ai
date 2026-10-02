"""Core package exports for Sentinel AI.

Exports are resolved lazily through :pep:`562` module ``__getattr__`` rather
than imported eagerly. :mod:`app.core.logger`, :mod:`app.core.security` and
:mod:`app.core.dependencies` each evaluate ``get_settings()`` at module scope,
so eagerly importing them from this package would make the mere act of
importing :mod:`app.core.config` -- as Alembic's ``env.py`` does -- require
``DATABASE_URL`` and ``SECRET_KEY`` to be configured.

The public surface is unchanged: every name below remains importable as
``from app.core import <name>``. :data:`__all__` keeps its original order, with
the authorization guards appended at the end.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.core.config import Settings as Settings
    from app.core.config import get_settings as get_settings
    from app.core.dependencies import get_current_active_user as get_current_active_user
    from app.core.dependencies import get_current_user as get_current_user
    from app.core.dependencies import get_db as get_db
    from app.core.dependencies import get_request_id as get_request_id
    from app.core.dependencies import require_admin as require_admin
    from app.core.dependencies import require_analyst_or_above as require_analyst_or_above
    from app.core.dependencies import require_roles as require_roles
    from app.core.dependencies import require_viewer_or_above as require_viewer_or_above
    from app.core.logger import get_logger as get_logger
    from app.core.logger import log_exception as log_exception
    from app.core.logger import log_execution_time as log_execution_time
    from app.core.security import AuthenticationError as AuthenticationError
    from app.core.security import AuthorizationError as AuthorizationError
    from app.core.security import ExpiredTokenError as ExpiredTokenError
    from app.core.security import InvalidTokenError as InvalidTokenError
    from app.core.security import create_access_token as create_access_token
    from app.core.security import decode_access_token as decode_access_token
    from app.core.security import get_current_subject as get_current_subject
    from app.core.security import hash_password as hash_password
    from app.core.security import mask_sensitive as mask_sensitive
    from app.core.security import verify_password as verify_password

__all__ = [
    "Settings",
    "get_settings",
    "get_logger",
    "log_exception",
    "log_execution_time",
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
    "get_current_subject",
    "mask_sensitive",
    "AuthenticationError",
    "InvalidTokenError",
    "ExpiredTokenError",
    "AuthorizationError",
    "get_db",
    "get_current_user",
    "get_current_active_user",
    "get_request_id",
    "require_roles",
    "require_viewer_or_above",
    "require_analyst_or_above",
    "require_admin",
]

_EXPORT_MODULES: dict[str, str] = {
    "Settings": "app.core.config",
    "get_settings": "app.core.config",
    "get_logger": "app.core.logger",
    "log_exception": "app.core.logger",
    "log_execution_time": "app.core.logger",
    "hash_password": "app.core.security",
    "verify_password": "app.core.security",
    "create_access_token": "app.core.security",
    "decode_access_token": "app.core.security",
    "get_current_subject": "app.core.security",
    "mask_sensitive": "app.core.security",
    "AuthenticationError": "app.core.security",
    "InvalidTokenError": "app.core.security",
    "ExpiredTokenError": "app.core.security",
    "AuthorizationError": "app.core.security",
    "get_db": "app.core.dependencies",
    "get_current_user": "app.core.dependencies",
    "get_current_active_user": "app.core.dependencies",
    "get_request_id": "app.core.dependencies",
    "require_roles": "app.core.dependencies",
    "require_viewer_or_above": "app.core.dependencies",
    "require_analyst_or_above": "app.core.dependencies",
    "require_admin": "app.core.dependencies",
}


def __getattr__(name: str) -> Any:
    """Import and return the submodule attribute that defines ``name``.

    Args:
        name: Attribute name requested on the ``app.core`` package.

    Returns:
        Any: The requested export from its defining submodule.

    Raises:
        AttributeError: If the name is not a public core export.
    """
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    """Return the public export names for introspection.

    Returns:
        list[str]: Sorted list of public ``app.core`` export names.
    """
    return sorted(__all__)
