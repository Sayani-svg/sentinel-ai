"""Public middleware package exports for Sentinel AI.

Every middleware module resolves application settings at import time (through
:mod:`app.core.logger` and the ``Settings`` annotation), so the exports below
are resolved lazily via :pep:`562` module ``__getattr__`` rather than imported
eagerly. That keeps ``import app.middleware`` free of configuration side
effects, matching the deferral already used by :mod:`app.core` and
:mod:`app.database`.

The public surface is unchanged by this deferral: every name below remains
importable as ``from app.middleware import <name>``, and :data:`__all__` is
preserved in its original order.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.middleware.cors import CORSMiddlewareOptions as CORSMiddlewareOptions
    from app.middleware.cors import build_cors_options as build_cors_options
    from app.middleware.cors import install_cors_middleware as install_cors_middleware
    from app.middleware.rate_limit import RateLimitDecision as RateLimitDecision
    from app.middleware.rate_limit import RateLimitMiddleware as RateLimitMiddleware
    from app.middleware.rate_limit import (
        SlidingWindowRateLimiter as SlidingWindowRateLimiter,
    )
    from app.middleware.rate_limit import (
        install_rate_limit_middleware as install_rate_limit_middleware,
    )
    from app.middleware.rate_limit import rate_limit_headers as rate_limit_headers
    from app.middleware.rate_limit import resolve_client_key as resolve_client_key
    from app.middleware.security_headers import (
        SecurityHeadersMiddleware as SecurityHeadersMiddleware,
    )
    from app.middleware.security_headers import (
        build_security_headers as build_security_headers,
    )
    from app.middleware.security_headers import (
        install_security_headers_middleware as install_security_headers_middleware,
    )

__all__ = [
    "CORSMiddlewareOptions",
    "build_cors_options",
    "install_cors_middleware",
    "RateLimitDecision",
    "SlidingWindowRateLimiter",
    "RateLimitMiddleware",
    "rate_limit_headers",
    "resolve_client_key",
    "install_rate_limit_middleware",
    "SecurityHeadersMiddleware",
    "build_security_headers",
    "install_security_headers_middleware",
]

_EXPORT_MODULES: dict[str, str] = {
    "CORSMiddlewareOptions": "app.middleware.cors",
    "build_cors_options": "app.middleware.cors",
    "install_cors_middleware": "app.middleware.cors",
    "RateLimitDecision": "app.middleware.rate_limit",
    "SlidingWindowRateLimiter": "app.middleware.rate_limit",
    "RateLimitMiddleware": "app.middleware.rate_limit",
    "rate_limit_headers": "app.middleware.rate_limit",
    "resolve_client_key": "app.middleware.rate_limit",
    "install_rate_limit_middleware": "app.middleware.rate_limit",
    "SecurityHeadersMiddleware": "app.middleware.security_headers",
    "build_security_headers": "app.middleware.security_headers",
    "install_security_headers_middleware": "app.middleware.security_headers",
}


def __getattr__(name: str) -> Any:
    """Import and return the submodule attribute that defines ``name``.

    Args:
        name: Attribute name requested on the ``app.middleware`` package.

    Returns:
        Any: The requested export from its defining submodule.

    Raises:
        AttributeError: If the name is not a public middleware export.
    """
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    """Return the public export names for introspection.

    Returns:
        list[str]: Sorted list of public ``app.middleware`` export names.
    """
    return sorted(__all__)