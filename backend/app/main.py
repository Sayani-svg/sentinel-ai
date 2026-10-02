"""FastAPI application factory and ASGI entry point for Sentinel AI.

The module exposes :func:`create_app` so the application can be constructed with
explicit settings (tests, embedding, alternative deployments) and a module level
``app`` instance so ASGI servers can address ``app.main:app``.

Startup deliberately does not open a database connection. The engine is created
lazily on first use, which keeps ``uvicorn app.main:app`` importable in
environments that only need to validate configuration or render OpenAPI, and
means a transient database outage degrades request handling instead of
preventing the process from booting.

Middleware is installed in reverse execution order, because Starlette runs the
last registered middleware outermost:

1. security headers
2. rate limiting
3. CORS

so an effective request travels CORS -> rate limiting -> security headers ->
route handler. Rate limiting therefore sees pre-flight requests as well, which
is intentional, while documentation endpoints are exempt because they are not
part of the public API surface.

Versioned routers are mounted under ``API_V1_PREFIX`` after the middleware is
installed, and importing them touches no database engine: the session is
created per request by :func:`~app.core.dependencies.get_db`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1 import api_router
from app.core.config import Settings, get_settings
from app.core.logger import get_logger, log_exception
from app.middleware import (
    install_cors_middleware,
    install_rate_limit_middleware,
    install_security_headers_middleware,
)

logger = get_logger(__name__)

INTERNAL_SERVER_ERROR_MESSAGE: str = "Internal server error."


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Log application lifecycle events around the serving phase.

    Args:
        application: The FastAPI instance being started. The resolved
            :class:`~app.core.config.Settings` are read from ``application.state``.

    Yields:
        None: Control returns to the ASGI server while the app serves traffic.
    """
    settings: Settings = application.state.settings
    logger.info(
        "Starting %s v%s in '%s' environment (API prefix '%s').",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.ENVIRONMENT,
        settings.API_V1_PREFIX,
    )
    logger.info(
        "Feature flags: rate_limiting=%s audit_logging=%s reports=%s shap=%s model_metrics=%s.",
        settings.ENABLE_RATE_LIMITING,
        settings.ENABLE_AUDIT_LOGGING,
        settings.ENABLE_REPORTS,
        settings.ENABLE_SHAP,
        settings.ENABLE_MODEL_METRICS,
    )
    try:
        yield
    finally:
        logger.info("Shutting down %s v%s.", settings.APP_NAME, settings.APP_VERSION)


def register_exception_handlers(application: FastAPI) -> None:
    """Install consistent logging and JSON bodies for uncaught HTTP failures.

    :class:`~starlette.exceptions.HTTPException` covers every deliberate status
    code raised by routes and dependencies. Anything that escapes it is an
    unhandled server fault, whose full traceback is logged before a generic
    response is returned so internals are never leaked to the caller.

    Starlette routes handlers registered for ``Exception`` through its outermost
    ``ServerErrorMiddleware`` and re-raises afterwards so the process still
    surfaces the fault. Two consequences are accepted deliberately: unhandled
    500 responses do not carry the security headers applied by
    :mod:`app.middleware.security_headers`, and ASGI test clients must be
    constructed with ``raise_server_exceptions=False`` to observe them.

    Args:
        application: Application instance to configure.
    """

    @application.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Log and serialise a deliberate HTTP error.

        Args:
            request: Request that triggered the exception.
            exc: The raised HTTP exception.

        Returns:
            JSONResponse: ``{"detail": ...}`` response preserving ``exc`` headers.
        """
        message = (
            exc.detail
            if isinstance(exc.detail, str)
            else "The request could not be completed."
        )
        if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            logger.error(
                "%s %s failed with status %d: %s",
                request.method,
                request.url.path,
                exc.status_code,
                message,
            )
        else:
            logger.warning(
                "%s %s rejected with status %d: %s",
                request.method,
                request.url.path,
                exc.status_code,
                message,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": message},
            headers=getattr(exc, "headers", None),
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_exception(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Log the traceback of an unhandled fault and return HTTP 500.

        Args:
            request: Request that triggered the exception.
            exc: The unhandled exception.

        Returns:
            JSONResponse: Generic HTTP 500 response.
        """
        log_exception(
            logger,
            f"Unhandled exception while processing {request.method} {request.url.path}",
            exc,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": INTERNAL_SERVER_ERROR_MESSAGE},
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a fully configured Sentinel AI FastAPI application.

    Args:
        settings: Optional pre-resolved settings. When omitted the cached
            :func:`~app.core.config.get_settings` instance is used.

    Returns:
        FastAPI: Application with middleware and exception handlers installed
        and the resolved settings exposed on ``app.state.settings``.

    Raises:
        pydantic.ValidationError: If required settings are missing or invalid
            and no ``settings`` argument is supplied.
    """
    resolved_settings: Settings = settings if settings is not None else get_settings()

    application = FastAPI(
        title=resolved_settings.API_TITLE,
        description=resolved_settings.API_DESCRIPTION,
        version=resolved_settings.APP_VERSION,
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings

    install_security_headers_middleware(application, resolved_settings)
    install_rate_limit_middleware(application, resolved_settings)
    install_cors_middleware(application, resolved_settings)
    register_exception_handlers(application)
    application.include_router(api_router, prefix=resolved_settings.API_V1_PREFIX)

    logger.info(
        "Application '%s' initialised with %d route(s).",
        resolved_settings.APP_NAME,
        len(application.routes),
    )
    return application


app = create_app()