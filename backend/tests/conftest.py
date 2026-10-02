"""Shared pytest fixtures for the Sentinel AI backend test suite.

``Settings`` requires ``DATABASE_URL`` and ``SECRET_KEY``, so the values are
provisioned here at import time -- before pytest collects test modules, which is
why this happens at module scope rather than inside a fixture.

Two kinds of test run against this application:

* Tests that only exercise middleware, routing or OpenAPI use the ``client``
  fixture, which never opens a database.
* Tests that need persistence use ``db_client`` or ``db_session``, which run
  against a private in-memory SQLite database created per test. SQLite is used
  because it needs no server, so the suite never contacts the PostgreSQL URL in
  ``DATABASE_URL``. Both routes reach the database exclusively through
  :func:`~app.core.dependencies.get_db`, which tests override, so importing the
  application still never constructs the production engine.

Real environment variables always win, so exporting ``DATABASE_URL`` before
running the suite is harmless.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from sqlalchemy import Engine
    from sqlalchemy.orm import Session

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg2://sentinel:sentinel@localhost:5432/sentinel_test",
)
os.environ.setdefault("SECRET_KEY", "test-only-secret-key-not-valid-in-production")
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("ENABLE_FILE_LOGGING", "false")
# bcrypt's cost factor is read once, when app.core.security is imported, so it has
# to be set here rather than in a fixture. The lowest cost that is still a valid
# factor keeps the suite fast without changing which code path runs.
os.environ.setdefault("BCRYPT_ROUNDS", "4")

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import app.models  # noqa: E402,F401  registers every table on the shared metadata
from app.core.config import Settings, get_settings  # noqa: E402
from app.core.dependencies import get_db  # noqa: E402
from app.database.base import Base  # noqa: E402
from app.main import create_app  # noqa: E402

HTTPX_REQUIREMENT: str = "starlette TestClient requires the httpx package"


@pytest.fixture()
def settings() -> Settings:
    """Return the cached application settings used by the test suite.

    Returns:
        Settings: Validated settings built from the test environment.
    """
    return get_settings()


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """Yield a ``TestClient`` bound to a freshly created application.

    Yields:
        TestClient: Client driving the app through its full middleware stack
            and lifespan events. No database is reachable through this client;
            use ``db_client`` for tests that persist data.
    """
    pytest.importorskip("httpx", reason=HTTPX_REQUIREMENT)

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture()
def engine() -> Iterator[Engine]:
    """Yield a private in-memory database engine with the schema created.

    ``StaticPool`` keeps every session on the one in-memory connection, so
    tables created here stay visible to the sessions handed out below.
    ``check_same_thread`` is disabled because Starlette runs synchronous route
    handlers and dependencies on a worker thread.

    Yields:
        Engine: An engine backed by an empty SQLite database.
    """
    pytest.importorskip("httpx", reason=HTTPX_REQUIREMENT)

    test_engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    try:
        yield test_engine
    finally:
        Base.metadata.drop_all(bind=test_engine)
        test_engine.dispose()


@pytest.fixture()
def db_session(engine: Engine) -> Iterator[Session]:
    """Yield a session bound to the per-test database.

    Yields:
        Session: An open session that is rolled back and closed afterwards.
    """
    db = Session(bind=engine, expire_on_commit=True)
    try:
        yield db
    finally:
        db.rollback()
        db.close()


@pytest.fixture()
def db_app(engine: Engine) -> Iterator[FastAPI]:
    """Yield an application whose ``get_db`` resolves to the test database.

    Yields:
        FastAPI: An app with the database dependency overridden.
    """
    application = create_app()

    def override_get_db() -> Iterator[Session]:
        """Serve each request from the per-test database.

        Yields:
            Session: A session bound to the test engine.
        """
        db = Session(bind=engine, expire_on_commit=False)
        try:
            yield db
        finally:
            db.close()

    application.dependency_overrides[get_db] = override_get_db
    try:
        yield application
    finally:
        application.dependency_overrides.clear()


@pytest.fixture()
def db_client(db_app: FastAPI) -> Iterator[TestClient]:
    """Yield a ``TestClient`` whose requests reach the per-test database.

    Yields:
        TestClient: Client bound to :func:`db_app`.
    """
    with TestClient(db_app) as test_client:
        yield test_client
