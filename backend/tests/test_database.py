"""Tests for the database session layer and its single ``get_db`` dependency.

``get_db`` was once defined twice: once in :mod:`app.core.dependencies`, which
every router and every test override references, and again in
:mod:`app.database.session`, bound to the module-level production engine. Two
copies is a trap that fails quietly. A test that overrides the one the router
does not use gets no error -- it simply never reaches the test database and
tries to open the production connection instead, which either raises an obscure
connection error or, worse, runs against real data.

These tests pin the single identity that prevents it, and then check that the
identity is load-bearing rather than cosmetic: an override keyed on any published
alias has to reach a real route.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session

import app.core
import app.database
import app.database.session
from app.core.dependencies import get_db as canonical_get_db
from app.core.security import hash_password
from app.main import create_app
from app.models.user import User
from app.schemas.user import UserRole

LOGIN_URL: str = "/api/v1/auth/login"

VALID_PASSWORD: str = "correct-horse-battery-staple"
SEEDED_EMAIL: str = "ada@example.com"


def every_published_alias() -> list[tuple[str, object]]:
    """Return each name ``get_db`` is published under.

    ``app.database`` and ``app.core`` resolve their exports through :pep:`562`
    module ``__getattr__``, so these attributes materialise on access rather than
    sitting in the package ``__dict__``.

    Returns:
        list[tuple[str, object]]: Label and function for each published name.
    """
    return [
        ("app.core.dependencies", canonical_get_db),
        ("app.database.session", app.database.session.get_db),
        ("app.database", app.database.get_db),
        ("app.core", app.core.get_db),
    ]


@pytest.mark.parametrize(
    ("label", "published"),
    every_published_alias(),
    ids=[label for label, _ in every_published_alias()],
)
def test_every_published_name_is_the_same_dependency(
    label: str, published: object
) -> None:
    """Publishing ``get_db`` under four names must not produce four functions.

    Args:
        label: Name the function was reached through, for the failure message.
        published: The function that name resolved to.
    """
    assert published is canonical_get_db, f"{label} exports a different get_db"


def seed_a_login(engine: Engine) -> None:
    """Insert one account the login endpoint can authenticate.

    Args:
        engine: Test database engine.
    """
    with Session(bind=engine) as db:
        db.add(
            User(
                name="Ada Lovelace",
                email=SEEDED_EMAIL,
                password_hash=hash_password(VALID_PASSWORD),
                role=UserRole.VIEWER.value,
                created_at=datetime.now(timezone.utc),
            )
        )
        db.commit()


def test_an_override_keyed_on_an_alias_reaches_a_route_that_uses_another(
    engine: Engine,
) -> None:
    """The aliases are interchangeable as dependency-override keys.

    This is the test that would have caught the split. The override is keyed on
    ``app.database.get_db``; the login route resolves its session through
    ``app.core.dependencies.get_db``. The request can only succeed if those are
    one function -- had they diverged, the override would not apply and the route
    would have fallen through to the production engine instead of the per-test
    database.
    """
    seed_a_login(engine)
    application = create_app()

    def override_get_db() -> Iterator[Session]:
        """Serve the request from the per-test database.

        Yields:
            Session: A session bound to the test engine.
        """
        db = Session(bind=engine, expire_on_commit=False)
        try:
            yield db
        finally:
            db.close()

    application.dependency_overrides[app.database.get_db] = override_get_db

    with TestClient(application) as client:
        response = client.post(
            LOGIN_URL, json={"email": SEEDED_EMAIL, "password": VALID_PASSWORD}
        )

    assert response.status_code == 200, response.text
    assert "access_token" in response.json()


@pytest.mark.parametrize(
    "alias_name",
    ["app.database", "app.core", "app.database.session"],
)
def test_each_alias_resolves_the_dependency_a_route_declares(alias_name: str) -> None:
    """A route declaring the alias resolves the canonical function object.

    FastAPI keys ``dependency_overrides`` on the callable object itself, so a
    route written against any alias is only overridable while the aliases are one
    object. Declaring the dependency here mirrors what a router does.

    Args:
        alias_name: Package whose published name should be used.
    """
    published = {
        "app.database": app.database.get_db,
        "app.core": app.core.get_db,
        "app.database.session": app.database.session.get_db,
    }[alias_name]
    application: FastAPI = FastAPI()

    @application.get("/probe")
    def probe(db: Session = Depends(published)) -> dict[str, str]:
        """Report which dependency object this route was wired to.

        Args:
            db: Session produced by the dependency.

        Returns:
            dict[str, str]: The module the resolved dependency came from.
        """
        return {"dependency_module": published.__module__}

    assert published is canonical_get_db
    with TestClient(application) as client:
        body = client.get("/probe").json()

    assert body == {"dependency_module": "app.core.dependencies"}
