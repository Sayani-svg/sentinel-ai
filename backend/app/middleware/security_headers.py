"""Standardized HTTP security headers for Sentinel AI.

Every response emitted by the API carries :data:`BASE_SECURITY_HEADERS`.
Transport- and framing-related directives that would break the Swagger UI or
plain-HTTP local development are withheld until ``ENVIRONMENT`` is
``production``, where :data:`PRODUCTION_SECURITY_HEADERS` is layered on top.
"""

from __future__ import annotations

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.config import Settings
from app.core.logger import get_logger

logger = get_logger(__name__)

BASE_SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "0",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-site",
    "Permissions-Policy": "geolocation=(), camera=(), microphone=(), payment=()",
    "X-Permitted-Cross-Domain-Policies": "none",
}

PRODUCTION_SECURITY_HEADERS: dict[str, str] = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
        "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    ),
}


def build_security_headers(settings: Settings) -> dict[str, str]:
    """Resolve the security headers that apply to the configured environment.

    Args:
        settings: Application settings consulted for the environment name.

    Returns:
        dict[str, str]: Fresh mapping of header name to header value.
    """
    headers = dict(BASE_SECURITY_HEADERS)
    if settings.is_production:
        headers.update(PRODUCTION_SECURITY_HEADERS)
    return headers


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach the configured security headers to every outgoing response.

    Attributes:
        headers: Header name to header value mapping applied to each response.
    """

    def __init__(
        self,
        app: ASGIApp,
        dispatch: RequestResponseEndpoint | None = None,
        *,
        settings: Settings,
    ) -> None:
        """Initialise the middleware.

        Args:
            app: ASGI application being wrapped.
            dispatch: Optional custom dispatch callable forwarded to Starlette.
            settings: Application settings used to resolve the header set.
        """
        super().__init__(app, dispatch)
        self.headers = build_security_headers(settings)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Run the downstream handler and stamp security headers on the result.

        Args:
            request: Incoming request.
            call_next: Downstream ASGI continuation.

        Returns:
            Response: Downstream response with security headers applied.
        """
        response = await call_next(request)
        for header, value in self.headers.items():
            response.headers[header] = value
        return response


def install_security_headers_middleware(app: FastAPI, settings: Settings) -> None:
    """Attach the security headers middleware to a FastAPI application.

    Args:
        app: Application instance to configure.
        settings: Application settings used to resolve the header set.

    Raises:
        RuntimeError: If the application has already started serving requests.
    """
    app.add_middleware(SecurityHeadersMiddleware, settings=settings)
    logger.info(
        "Security headers middleware installed with %d header(s).",
        len(build_security_headers(settings)),
    )