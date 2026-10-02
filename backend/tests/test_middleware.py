"""Tests for the Sentinel AI middleware layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from app.core.config import Settings, get_settings
from app.middleware.cors import (
    CORS_EXPOSED_HEADERS,
    CORS_MAX_AGE_SECONDS,
    build_cors_options,
)
from app.middleware.rate_limit import (
    RATE_LIMIT_ERROR_MESSAGE,
    RateLimitDecision,
    SlidingWindowRateLimiter,
    install_rate_limit_middleware,
    rate_limit_headers,
    resolve_client_key,
)
from app.middleware.security_headers import (
    BASE_SECURITY_HEADERS,
    PRODUCTION_SECURITY_HEADERS,
    build_security_headers,
)

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

TEST_DATABASE_URL = "postgresql+psycopg2://user:pass@localhost:5432/middleware_test"
TEST_SECRET_KEY = "middleware-test-secret-key-not-valid-in-production"
PRODUCTION_SECRET_KEY = "production-middleware-secret-key-32-chars"


class _FakeClock:
    """Deterministic monotonic time source for rate limit assertions."""

    def __init__(self, start: float = 0.0) -> None:
        """Initialise the clock.

        Args:
            start: Initial value in seconds.
        """
        self._now = start

    def __call__(self) -> float:
        """Return the current fake time.

        Returns:
            float: Seconds elapsed on the fake timeline.
        """
        return self._now

    def advance(self, seconds: float) -> None:
        """Move the fake timeline forward.

        Args:
            seconds: Number of seconds to advance.
        """
        self._now += seconds


def _build_request(
    headers: dict[str, str] | None = None,
    client: tuple[str, int] | None = ("198.51.100.7", 51234),
) -> Request:
    """Build a minimal Starlette request for client-key resolution tests.

    Args:
        headers: Raw request headers.
        client: Client address tuple, or ``None`` for a peerless connection.

    Returns:
        Request: Request with the supplied scope values.
    """
    raw_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/probe",
            "raw_path": b"/probe",
            "query_string": b"",
            "root_path": "",
            "headers": raw_headers,
            "client": client,
            "server": ("testserver", 80),
        }
    )


def _test_settings(**overrides: object) -> Settings:
    """Build isolated settings for middleware configuration tests.

    Args:
        **overrides: Field values overriding the testing defaults.

    Returns:
        Settings: Validated settings instance.
    """
    defaults: dict[str, object] = {
        "DATABASE_URL": TEST_DATABASE_URL,
        "SECRET_KEY": TEST_SECRET_KEY,
        "ENVIRONMENT": "testing",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


# --- CORS -------------------------------------------------------------------


def test_build_cors_options_uses_configured_origins() -> None:
    """Configured origins, methods and headers must be forwarded verbatim."""
    settings = _test_settings(
        ALLOWED_ORIGINS=["http://localhost:3000/"],
        CORS_ALLOW_METHODS=["get", "post"],
    )

    options = build_cors_options(settings)

    assert options["allow_origins"] == ["http://localhost:3000"]
    assert options["allow_methods"] == ["GET", "POST"]
    assert options["allow_credentials"] is True
    assert options["expose_headers"] == list(CORS_EXPOSED_HEADERS)
    assert options["max_age"] == CORS_MAX_AGE_SECONDS


def test_build_cors_options_disables_credentials_for_wildcard() -> None:
    """A wildcard origin must not be combined with credentials."""
    settings = _test_settings(ALLOW_ALL_ORIGINS=True, CORS_ALLOW_CREDENTIALS=True)

    options = build_cors_options(settings)

    assert options["allow_origins"] == ["*"]
    assert options["allow_credentials"] is False


# --- Security headers -------------------------------------------------------


def test_build_security_headers_excludes_transport_directives_outside_production() -> None:
    """HSTS and CSP must not break Swagger or plain-HTTP local development."""
    headers = build_security_headers(_test_settings())

    assert headers == BASE_SECURITY_HEADERS
    assert "Strict-Transport-Security" not in headers
    assert "Content-Security-Policy" not in headers


def test_build_security_headers_adds_transport_directives_in_production() -> None:
    """Production must receive the full hardened header set."""
    headers = build_security_headers(
        _test_settings(ENVIRONMENT="production", SECRET_KEY=PRODUCTION_SECRET_KEY)
    )

    assert headers == {**BASE_SECURITY_HEADERS, **PRODUCTION_SECURITY_HEADERS}
    assert "max-age=31536000" in headers["Strict-Transport-Security"]


# --- Client key resolution --------------------------------------------------


def test_resolve_client_key_prefers_forwarded_for() -> None:
    """The left-most forwarded address identifies the originating client."""
    request = _build_request(
        headers={"X-Forwarded-For": "203.0.113.9, 10.0.0.1", "X-Real-IP": "10.0.0.1"}
    )

    assert resolve_client_key(request) == "203.0.113.9"


def test_resolve_client_key_falls_back_to_real_ip_then_peer() -> None:
    """Resolution must degrade to ``X-Real-IP`` and then the socket peer."""
    assert resolve_client_key(_build_request(headers={"X-Real-IP": "203.0.113.4"})) == (
        "203.0.113.4"
    )
    assert resolve_client_key(_build_request()) == "198.51.100.7"
    assert resolve_client_key(_build_request(client=None)) == "unknown"


# --- Sliding window limiter -------------------------------------------------


def test_limiter_permits_requests_up_to_the_budget() -> None:
    """The budget must be honoured exactly before the first rejection."""
    clock = _FakeClock()
    limiter = SlidingWindowRateLimiter(3, 60, time_source=clock)

    decisions = [limiter.check("client") for _ in range(4)]

    assert [decision.allowed for decision in decisions] == [True, True, True, False]
    assert [decision.remaining for decision in decisions] == [2, 1, 0, 0]
    assert decisions[-1].retry_after == 60


def test_limiter_budget_recovers_as_the_window_slides() -> None:
    """A full window must free its oldest slot once that slot ages out."""
    clock = _FakeClock()
    limiter = SlidingWindowRateLimiter(1, 60, time_source=clock)

    assert limiter.check("client").allowed is True
    assert limiter.check("client").allowed is False

    clock.advance(60)

    decision = limiter.check("client")
    assert decision.allowed is True
    assert decision.remaining == 0


def test_limiter_budgets_are_tracked_per_client() -> None:
    """One noisy client must not consume another client's budget."""
    clock = _FakeClock()
    limiter = SlidingWindowRateLimiter(1, 60, time_source=clock)

    assert limiter.check("noisy").allowed is True
    assert limiter.check("noisy").allowed is False
    assert limiter.check("quiet").allowed is True
    assert limiter.tracked_clients() == 2


def test_limiter_evicts_least_recent_client_at_capacity() -> None:
    """The tracking table must stay bounded under many distinct clients."""
    clock = _FakeClock()
    limiter = SlidingWindowRateLimiter(5, 60, time_source=clock, max_tracked_clients=2)

    limiter.check("first")
    clock.advance(1)
    limiter.check("second")
    clock.advance(1)
    limiter.check("third")

    assert limiter.tracked_clients() == 2
    assert limiter.check("first").allowed is True


def test_limiter_reset_clears_recorded_hits() -> None:
    """Reset must return the limiter to an empty tracking table."""
    limiter = SlidingWindowRateLimiter(1, 60, time_source=_FakeClock())

    assert limiter.check("client").allowed is True
    limiter.reset()
    assert limiter.tracked_clients() == 0
    assert limiter.check("client").allowed is True


def test_limiter_exposes_its_configuration() -> None:
    """The configured budget and window must be introspectable."""
    limiter = SlidingWindowRateLimiter(7, 30)

    assert limiter.limit == 7
    assert limiter.window_seconds == 30


@pytest.mark.parametrize(
    ("limit", "window_seconds", "max_clients"),
    [(0, 60, 1), (1, 0, 1), (1, 60, 0)],
)
def test_limiter_rejects_non_positive_configuration(
    limit: int, window_seconds: int, max_clients: int
) -> None:
    """Invalid configuration must fail loudly rather than silently misbehaving.

    Args:
        limit: Request budget under test.
        window_seconds: Window length under test.
        max_clients: Tracking capacity under test.
    """
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(
            limit, window_seconds, max_tracked_clients=max_clients
        )


def test_limiter_rejects_empty_client_key() -> None:
    """An empty key would silently merge unrelated clients."""
    limiter = SlidingWindowRateLimiter(1, 60, time_source=_FakeClock())

    with pytest.raises(ValueError):
        limiter.check("")


def test_rate_limit_headers_are_stringified() -> None:
    """Decision fields must be rendered as HTTP-safe strings."""
    headers = rate_limit_headers(
        RateLimitDecision(allowed=True, limit=10, remaining=4, retry_after=17)
    )

    assert headers == {
        "X-RateLimit-Limit": "10",
        "X-RateLimit-Remaining": "4",
        "X-RateLimit-Reset": "17",
    }


# --- Rate limit middleware --------------------------------------------------


def _rate_limited_client(limiter: SlidingWindowRateLimiter) -> TestClient:
    """Build a client for a bare app guarded by ``limiter``.

    Args:
        limiter: Limiter to enforce.

    Returns:
        TestClient: Client for the guarded application.
    """
    from fastapi.testclient import TestClient

    application = FastAPI()
    install_rate_limit_middleware(application, get_settings(), limiter=limiter)
    return TestClient(application)


@requires_httpx
def test_rate_limit_middleware_rejects_requests_over_budget() -> None:
    """Requests beyond the budget must fail with HTTP 429 and ``Retry-After``."""
    limiter = SlidingWindowRateLimiter(2, 60, time_source=_FakeClock())

    with _rate_limited_client(limiter) as client:
        assert client.get("/probe").status_code == 404
        assert client.get("/probe").status_code == 404
        response = client.get("/probe")

    assert response.status_code == 429
    assert response.json() == {"detail": RATE_LIMIT_ERROR_MESSAGE}
    assert response.headers["Retry-After"] == "60"
    assert response.headers["X-RateLimit-Remaining"] == "0"


@requires_httpx
def test_rate_limit_middleware_is_skipped_when_disabled() -> None:
    """A disabled feature flag must leave the middleware uninstalled."""
    from fastapi.testclient import TestClient

    application = FastAPI()
    settings = _test_settings(ENABLE_RATE_LIMITING=False)

    install_rate_limit_middleware(
        application, settings, limiter=SlidingWindowRateLimiter(1, 60)
    )

    with TestClient(application) as client:
        assert client.get("/probe").status_code == 404
        assert "X-RateLimit-Limit" not in client.get("/probe").headers