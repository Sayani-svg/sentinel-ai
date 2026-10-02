"""Cross-Origin Resource Sharing configuration for Sentinel AI.

The middleware is installed through :func:`install_cors_middleware` so that the
effective allow-list always derives from application settings instead of being
duplicated at the call site. Operational defaults that are not security
relevant enough to warrant a settings field (pre-flight cache lifetime and the
response headers the SPA needs to read) live here as module constants, matching
the convention already used by :mod:`app.core.logger`.
"""

from __future__ import annotations

from typing import TypedDict

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.core.config import Settings
from app.core.logger import get_logger

logger = get_logger(__name__)

CORS_MAX_AGE_SECONDS: int = 600
CORS_EXPOSED_HEADERS: tuple[str, ...] = (
    "Content-Disposition",
    "X-RateLimit-Limit",
    "X-RateLimit-Remaining",
    "X-RateLimit-Reset",
)


class CORSMiddlewareOptions(TypedDict):
    """Keyword arguments accepted by :class:`starlette.middleware.cors.CORSMiddleware`.

    Attributes:
        allow_origins: Browser origins permitted to call the API.
        allow_credentials: Whether cookies and ``Authorization`` headers may be
            sent cross-origin.
        allow_methods: HTTP verbs permitted for pre-flight requests.
        allow_headers: Request headers permitted for pre-flight requests.
        expose_headers: Response headers the browser is allowed to surface to
            client-side JavaScript.
        max_age: Seconds a browser may cache the pre-flight response.
    """

    allow_origins: list[str]
    allow_credentials: bool
    allow_methods: list[str]
    allow_headers: list[str]
    expose_headers: list[str]
    max_age: int


def build_cors_options(settings: Settings) -> CORSMiddlewareOptions:
    """Translate application settings into CORS middleware keyword arguments.

    Args:
        settings: Application settings containing the origin allow-list and the
            permitted methods and headers.

    Returns:
        CORSMiddlewareOptions: Fully populated middleware keyword arguments.

    Notes:
        A wildcard origin combined with credentials is rejected by browsers, so
        ``ALLOW_ALL_ORIGINS`` forces credentials off and emits a warning rather
        than silently emitting an unusable header combination.
    """
    allow_origins: list[str] = settings.cors_origins
    allow_credentials: bool = settings.CORS_ALLOW_CREDENTIALS

    if allow_origins == ["*"]:
        allow_credentials = False
        logger.warning(
            "ALLOW_ALL_ORIGINS is enabled; CORS credentials were disabled because "
            "browsers reject the wildcard origin together with credentials."
        )

    return CORSMiddlewareOptions(
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=list(settings.CORS_ALLOW_METHODS),
        allow_headers=list(settings.CORS_ALLOW_HEADERS),
        expose_headers=list(CORS_EXPOSED_HEADERS),
        max_age=CORS_MAX_AGE_SECONDS,
    )


def install_cors_middleware(app: FastAPI, settings: Settings) -> None:
    """Attach the CORS middleware to a FastAPI application.

    Args:
        app: Application instance to configure.
        settings: Application settings used to derive the origin allow-list.

    Raises:
        RuntimeError: If the application has already started serving requests.
    """
    options = build_cors_options(settings)
    app.add_middleware(CORSMiddleware, **options)
    logger.info(
        "CORS middleware installed for %d origin(s) with credentials=%s.",
        len(options["allow_origins"]),
        options["allow_credentials"],
    )