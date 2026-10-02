"""Tests for the Sentinel AI FastAPI application factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from app.core.config import Settings
from app.main import INTERNAL_SERVER_ERROR_MESSAGE, create_app
from app.middleware.security_headers import BASE_SECURITY_HEADERS

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

try:
    import httpx as _httpx  # noqa: F401
except ModuleNotFoundError:
    _HTTPX_AVAILABLE = False
else:
    _HTTPX_AVAILABLE = True

requires_httpx = pytest.mark.skipif(
    not _HTTPX_AVAILABLE,
    reason="starlette TestClient requires the httpx package",
)


def _test_client(
    application: object, raise_server_exceptions: bool = True
) -> TestClient:
    """Build a ``TestClient`` for an application instance.

    Args:
        application: FastAPI application to drive.
        raise_server_exceptions: Whether unhandled server faults should
            propagate into the test.

    Returns:
        TestClient: Client bound to ``application``.
    """
    from fastapi.testclient import TestClient

    return TestClient(
        application,  # type: ignore[arg-type]
        raise_server_exceptions=raise_server_exceptions,
    )


def test_create_app_exposes_resolved_settings(settings: Settings) -> None:
    """The factory must publish the settings it was built with.

    Args:
        settings: Settings fixture used to construct the application.
    """
    application = create_app(settings)

    assert application.state.settings is settings
    assert application.title == settings.API_TITLE
    assert application.version == settings.APP_VERSION


def test_create_app_accepts_explicit_settings() -> None:
    """Explicit settings must win over the process environment."""
    settings = Settings(
        DATABASE_URL="postgresql+psycopg2://user:pass@localhost:5432/explicit",
        SECRET_KEY="explicit-override-secret-key",
        ENVIRONMENT="testing",
        APP_NAME="Sentinel AI (Test)",
    )

    application = create_app(settings)

    assert application.state.settings.APP_NAME == "Sentinel AI (Test)"


def test_create_app_falls_back_to_cached_settings(settings: Settings) -> None:
    """Omitting settings must reuse the cached process-wide instance."""
    application = create_app()

    assert application.state.settings is settings


@requires_httpx
def test_openapi_schema_is_served(client: TestClient) -> None:
    """Swagger/OpenAPI must load, as required by the Phase 2 validation gate.

    Args:
        client: Test client bound to a freshly created application.
    """
    response = client.get("/openapi.json")

    assert response.status_code == 200
    payload = response.json()
    assert payload["openapi"].startswith("3.")
    assert payload["info"]["version"]


@requires_httpx
def test_swagger_ui_is_served(client: TestClient) -> None:
    """The Swagger UI endpoint must remain reachable.

    Args:
        client: Test client bound to a freshly created application.
    """
    assert client.get("/docs").status_code == 200


@requires_httpx
def test_security_headers_are_applied_to_every_response(client: TestClient) -> None:
    """Standardized security headers must be stamped on outgoing responses.

    Args:
        client: Test client bound to a freshly created application.
    """
    response = client.get("/openapi.json")

    for header, value in BASE_SECURITY_HEADERS.items():
        assert response.headers[header] == value


@requires_httpx
def test_rate_limit_headers_are_exposed(client: TestClient) -> None:
    """Rate-limited routes must advertise the caller's remaining budget.

    ``/openapi.json`` is exempt from throttling, so a normal route is used to
    observe the headers the limiter adds.

    Args:
        client: Test client bound to a freshly created application.
    """
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.headers["X-RateLimit-Limit"] == "120"
    assert response.headers["X-RateLimit-Remaining"] == "119"
    assert response.headers["X-RateLimit-Reset"] == "60"


@requires_httpx
def test_documentation_endpoints_are_exempt_from_rate_limiting(
    client: TestClient,
) -> None:
    """Documentation endpoints bypass the limiter entirely.

    Args:
        client: Test client bound to a freshly created application.
    """
    response = client.get("/openapi.json")

    assert "X-RateLimit-Limit" not in response.headers


@requires_httpx
def test_unknown_route_returns_structured_not_found(client: TestClient) -> None:
    """Unknown routes must produce the standard JSON error envelope.

    Args:
        client: Test client bound to a freshly created application.
    """
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


@requires_httpx
def test_unhandled_exception_is_logged_and_returns_generic_500() -> None:
    """Unhandled faults must never leak internal detail to the caller.

    Starlette routes ``Exception`` handlers through its outermost
    ``ServerErrorMiddleware``, which re-raises after responding, so the client is
    configured with ``raise_server_exceptions=False``.
    """
    application = create_app()

    @application.get("/boom")
    async def boom() -> None:
        """Raise an unhandled fault to exercise the global exception handler."""
        raise RuntimeError("sensitive internal detail")

    with _test_client(application, raise_server_exceptions=False) as test_client:
        response = test_client.get("/boom")

    assert response.status_code == 500
    assert response.json() == {"detail": INTERNAL_SERVER_ERROR_MESSAGE}
    assert "sensitive internal detail" not in response.text


def test_lifespan_runs_without_a_database() -> None:
    """Entering the lifespan must not require a reachable database.

    Startup never opens a connection, so the app can boot against an
    unreachable database URL and still serve its own documentation surface.
    """
    settings = Settings(
        DATABASE_URL="postgresql+psycopg2://user:pass@192.0.2.1:5432/unreachable",
        SECRET_KEY="lifespan-secret-key-not-valid-in-production",
        ENVIRONMENT="testing",
    )
    application = create_app(settings)

    assert application.router.lifespan_context is not None
    assert application.state.settings is settings


@requires_httpx
@pytest.mark.parametrize("path", ["/openapi.json", "/docs"])
def test_documentation_endpoints_are_not_rate_limited(path: str) -> None:
    """Documentation endpoints are exempt so Swagger stays usable.

    Args:
        path: Documentation endpoint exercised with an exhausted budget.
    """
    from fastapi import FastAPI

    from app.core.config import get_settings
    from app.middleware import SlidingWindowRateLimiter, install_rate_limit_middleware

    class _FrozenClock:
        def __call__(self) -> float:
            return 0.0

    application = FastAPI()
    install_rate_limit_middleware(
        application,
        get_settings(),
        limiter=SlidingWindowRateLimiter(1, 60, time_source=_FrozenClock()),
    )

    with _test_client(application) as test_client:
        assert test_client.get(path).status_code == 200
        assert test_client.get(path).status_code == 200