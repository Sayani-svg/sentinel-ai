"""Sliding-window API rate limiting for Sentinel AI.

The limiter is deliberately transport agnostic: :class:`SlidingWindowRateLimiter`
only answers questions about a client key, while
:class:`RateLimitMiddleware` translates that answer into HTTP semantics. The
time source is injectable so the window arithmetic can be verified
deterministically in tests without sleeping.

State is kept in process memory and guarded by a single lock, which matches the
single-process deployment topology described in ``docs/deployment.md``. A
multi-worker deployment needs a shared backend such as Redis, which this module
does not pretend to provide.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.core.config import Settings
from app.core.logger import get_logger

logger = get_logger(__name__)

RATE_LIMIT_REQUESTS: int = 120
RATE_LIMIT_WINDOW_SECONDS: int = 60
RATE_LIMIT_MAX_TRACKED_CLIENTS: int = 10_000
RATE_LIMIT_EXEMPT_PATHS: frozenset[str] = frozenset({"/docs", "/redoc", "/openapi.json"})
RATE_LIMIT_ERROR_MESSAGE: str = (
    "Rate limit exceeded. Please retry after the indicated number of seconds."
)

_UNKNOWN_CLIENT_KEY: str = "unknown"


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Outcome of evaluating a single request against a client's budget.

    Attributes:
        allowed: Whether the request may proceed.
        limit: Maximum number of requests permitted per window.
        remaining: Requests still available in the current window.
        retry_after: Whole seconds until the window frees up at least one slot.
    """

    allowed: bool
    limit: int
    remaining: int
    retry_after: int


def rate_limit_headers(decision: RateLimitDecision) -> dict[str, str]:
    """Render a decision as the standard ``X-RateLimit-*`` response headers.

    Args:
        decision: Result of a rate limit evaluation.

    Returns:
        dict[str, str]: Header name to header value mapping.
    """
    return {
        "X-RateLimit-Limit": str(decision.limit),
        "X-RateLimit-Remaining": str(decision.remaining),
        "X-RateLimit-Reset": str(decision.retry_after),
    }


def resolve_client_key(request: Request) -> str:
    """Determine the rate limiting identity of an incoming request.

    The forwarded headers are preferred because the API is deployed behind a
    reverse proxy in every documented environment, meaning ``request.client.host``
    resolves to the proxy address for all callers. The proxy is therefore
    responsible for overwriting any client-supplied ``X-Forwarded-For`` value;
    trusting it without that guarantee would allow a caller to bypass the limit.

    Args:
        request: Incoming Starlette request.

    Returns:
        str: Stable per-client key, or ``"unknown"`` when no peer address exists.
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip() or _UNKNOWN_CLIENT_KEY

    real_ip = request.headers.get("X-Real-IP")
    if real_ip and real_ip.strip():
        return real_ip.strip()

    if request.client is not None and request.client.host:
        return request.client.host

    return _UNKNOWN_CLIENT_KEY


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window counter keyed by client identity.

    Attributes:
        limit: Maximum number of requests permitted per window.
        window_seconds: Length of the sliding window in seconds.
    """

    def __init__(
        self,
        limit: int = RATE_LIMIT_REQUESTS,
        window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
        *,
        time_source: Callable[[], float] = time.monotonic,
        max_tracked_clients: int = RATE_LIMIT_MAX_TRACKED_CLIENTS,
    ) -> None:
        """Initialise the limiter.

        Args:
            limit: Maximum number of requests permitted per window.
            window_seconds: Length of the sliding window in seconds.
            time_source: Callable returning the current time in seconds. Must be
                monotonic; injectable for deterministic tests.
            max_tracked_clients: Upper bound on tracked clients before the least
                recently active entry is evicted.

        Raises:
            ValueError: If ``limit``, ``window_seconds`` or
                ``max_tracked_clients`` is smaller than one.
        """
        if limit < 1:
            raise ValueError("Rate limit must be at least 1 request.")
        if window_seconds < 1:
            raise ValueError("Rate limit window must be at least 1 second.")
        if max_tracked_clients < 1:
            raise ValueError("max_tracked_clients must be at least 1.")

        self._limit = limit
        self._window_seconds = window_seconds
        self._time_source = time_source
        self._max_tracked_clients = max_tracked_clients
        self._hits: dict[str, deque[float]] = {}
        self._lock = Lock()

    @property
    def limit(self) -> int:
        """Return the maximum number of requests permitted per window.

        Returns:
            int: Configured request budget per window.
        """
        return self._limit

    @property
    def window_seconds(self) -> int:
        """Return the sliding window length.

        Returns:
            int: Configured window length in seconds.
        """
        return self._window_seconds

    def check(self, key: str) -> RateLimitDecision:
        """Record an attempt for ``key`` and report whether it is permitted.

        The attempt is recorded even when it is rejected, so a client that keeps
        hammering the API cannot extend its own lockout indefinitely by retrying
        faster than the window slides.

        Args:
            key: Client identity produced by :func:`resolve_client_key`.

        Returns:
            RateLimitDecision: Allow/deny outcome plus the headers to expose.

        Raises:
            ValueError: If ``key`` is empty.
        """
        if not key:
            raise ValueError("Rate limit key must not be empty.")

        now = self._time_source()
        cutoff = now - self._window_seconds

        with self._lock:
            self._prune_expired(now)
            self._make_room_for(key)

            hits = self._hits.setdefault(key, deque())

            if len(hits) >= self._limit:
                return RateLimitDecision(
                    allowed=False,
                    limit=self._limit,
                    remaining=0,
                    retry_after=self._retry_after(hits[0], now),
                )

            hits.append(now)
            return RateLimitDecision(
                allowed=True,
                limit=self._limit,
                remaining=self._limit - len(hits),
                retry_after=self._retry_after(hits[0], now),
            )

    def reset(self) -> None:
        """Discard all recorded request timestamps."""
        with self._lock:
            self._hits.clear()

    def tracked_clients(self) -> int:
        """Return the number of clients currently holding recorded timestamps.

        Returns:
            int: Size of the internal tracking table.
        """
        with self._lock:
            return len(self._hits)

    def _retry_after(self, oldest_hit: float, now: float) -> int:
        """Return whole seconds until ``oldest_hit`` leaves the window.

        Args:
            oldest_hit: Timestamp of the earliest request in the window.
            now: Current timestamp from the configured time source.

        Returns:
            int: At least one second.
        """
        return max(1, math.ceil(oldest_hit - (now - self._window_seconds)))

    def _prune_expired(self, now: float) -> None:
        """Drop timestamps and clients that have fully left the window.

        Args:
            now: Current timestamp from the configured time source.
        """
        cutoff = now - self._window_seconds
        for client, hits in list(self._hits.items()):
            if not hits or hits[-1] <= cutoff:
                del self._hits[client]
            else:
                while hits and hits[0] <= cutoff:
                    hits.popleft()

    def _make_room_for(self, key: str) -> None:
        """Evict the least recently active client when the table is full.

        Args:
            key: Client identity that is about to be tracked.
        """
        if key in self._hits or len(self._hits) < self._max_tracked_clients:
            return

        evicted = min(self._hits, key=lambda client: self._hits[client][-1])
        del self._hits[evicted]
        logger.warning(
            "Rate limiter evicted client %s to stay within the %d client tracking limit.",
            evicted,
            self._max_tracked_clients,
        )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests from clients that exceed their sliding-window budget.

    Attributes:
        limiter: Sliding-window counter backing the middleware.
        exempt_paths: Paths served without consuming client budget.
    """

    def __init__(
        self,
        app: ASGIApp,
        dispatch: RequestResponseEndpoint | None = None,
        *,
        settings: Settings,
        limiter: SlidingWindowRateLimiter | None = None,
    ) -> None:
        """Initialise the middleware.

        Args:
            app: ASGI application being wrapped.
            dispatch: Optional custom dispatch callable forwarded to Starlette.
            settings: Application settings; the limiter is skipped entirely when
                ``ENABLE_RATE_LIMITING`` is disabled.
            limiter: Optional pre-built limiter, primarily a test seam.
        """
        super().__init__(app, dispatch)
        self._enabled = settings.ENABLE_RATE_LIMITING
        self.limiter = limiter if limiter is not None else SlidingWindowRateLimiter()
        self.exempt_paths = RATE_LIMIT_EXEMPT_PATHS

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Evaluate the client budget and short-circuit with HTTP 429 if spent.

        Args:
            request: Incoming request.
            call_next: Downstream ASGI continuation.

        Returns:
            Response: The downstream response, or a 429 response carrying the
                ``Retry-After`` and ``X-RateLimit-*`` headers.
        """
        if not self._enabled or request.url.path in self.exempt_paths:
            return await call_next(request)

        client_key = resolve_client_key(request)
        decision = self.limiter.check(client_key)

        if decision.allowed:
            response = await call_next(request)
        else:
            logger.warning(
                "Rate limit exceeded for client %s on %s %s.",
                client_key,
                request.method,
                request.url.path,
            )
            response = JSONResponse(
                status_code=429,
                content={"detail": RATE_LIMIT_ERROR_MESSAGE},
                headers={"Retry-After": str(decision.retry_after)},
            )

        response.headers.update(rate_limit_headers(decision))
        return response


def install_rate_limit_middleware(
    app: FastAPI,
    settings: Settings,
    *,
    limiter: SlidingWindowRateLimiter | None = None,
) -> None:
    """Attach the rate limiting middleware unless it is disabled by settings.

    Args:
        app: Application instance to configure.
        settings: Application settings controlling the feature flag.
        limiter: Optional pre-built limiter, primarily a test seam.

    Raises:
        RuntimeError: If the application has already started serving requests.
    """
    if not settings.ENABLE_RATE_LIMITING:
        logger.info("API rate limiting is disabled; middleware was not installed.")
        return

    app.add_middleware(RateLimitMiddleware, settings=settings, limiter=limiter)
    logger.info(
        "Rate limiting middleware installed with a budget of %d requests per %d seconds.",
        limiter.limit if limiter is not None else RATE_LIMIT_REQUESTS,
        limiter.window_seconds if limiter is not None else RATE_LIMIT_WINDOW_SECONDS,
    )