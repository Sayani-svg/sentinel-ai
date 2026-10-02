"""Focused tests for ``GET /api/v1/users/me``."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.core.security import create_access_token, hash_password
from app.models.user import User
from app.schemas.user import UserRole

if TYPE_CHECKING:
    from fastapi.testclient import TestClient as TestClientType

ME_URL = "/api/v1/users/me"

VALID_PASSWORD = "correct-horse-battery-staple"

#: ``users.role`` is a plain text column, so this value can be stored. It models
#: the realistic case the guard has to absorb: a row written by a build whose
#: role set differs from this one's.
UNRECOGNISED_ROLE = "Overlord"


def _fabricated_user(role: str, user_id: int = 4_242) -> User:
    """Build an unsaved account carrying an arbitrary role.

    Args:
        role: Role to place on the instance, verbatim.
        user_id: Primary key to report.

    Returns:
        User: A transient, unpersisted account.
    """
    return User(
        id=user_id,
        name="Corrupt Row",
        email="corrupt@example.com",
        password_hash=hash_password(VALID_PASSWORD),
        role=role,
        created_at=datetime.now(timezone.utc),
    )


def seed_user(
    engine: Engine,
    *,
    role: str,
    email: str,
    name: str = "Probe User",
) -> int:
    """Insert a user with an arbitrary role and return its id.

    Args:
        engine: Test database engine.
        role: Role stored on the row, verbatim.
        email: Email address for the new account.
        name: Display name for the new account.

    Returns:
        int: The generated primary key.
    """
    with Session(bind=engine) as db:
        user = User(
            name=name,
            email=email,
            password_hash=hash_password(VALID_PASSWORD),
            role=role,
            created_at=datetime.now(timezone.utc),
        )
        db.add(user)
        db.commit()
        return int(user.id)


def bearer(user_id: int) -> dict[str, str]:
    """Build an authorization header for an account id.

    Args:
        user_id: Account id to embed as the token subject.

    Returns:
        dict[str, str]: A bearer authorization header.
    """
    return {"Authorization": f"Bearer {create_access_token(subject=str(user_id))}"}


class TestCurrentUserEndpoint:
    """``GET /api/v1/users/me``."""

    @pytest.mark.parametrize("role", [role.value for role in UserRole])
    def test_endpoint_returns_the_profile_for_each_role(
        self, db_client: TestClientType, engine: Engine, role: str
    ) -> None:
        """Viewer, Analyst and Admin all resolve their own profile."""
        user_id = seed_user(engine, role=role, email=f"{role}@example.com", name="Ada Lovelace")

        response = db_client.get(ME_URL, headers=bearer(user_id))

        assert response.status_code == 200
        body = response.json()
        assert body == {
            "id": user_id,
            "name": "Ada Lovelace",
            "email": f"{role}@example.com".lower(),
            "role": role,
            "created_at": body["created_at"],
        }

    def test_response_never_exposes_the_password_hash(
        self, db_client: TestClientType, engine: Engine
    ) -> None:
        """The profile carries no secret material, even though the row does."""
        user_id = seed_user(engine, role=UserRole.VIEWER.value, email="ada@example.com")

        body = db_client.get(ME_URL, headers=bearer(user_id)).json()

        assert set(body) == {"id", "name", "email", "role", "created_at"}
        assert not any("password" in key.lower() for key in body)

    def test_email_is_returned_normalised(
        self, db_client: TestClientType, engine: Engine
    ) -> None:
        """A row written with mixed casing is returned in canonical form."""
        user_id = seed_user(engine, role=UserRole.ADMIN.value, email="Ada@Example.COM")

        body = db_client.get(ME_URL, headers=bearer(user_id)).json()

        assert body["email"] == "ada@example.com"

    def test_endpoint_requires_a_bearer_token(self, db_client: TestClientType) -> None:
        """An anonymous request is refused with a bearer challenge.

        The detail comes from the OAuth2 scheme rather than from
        :mod:`app.core.dependencies`, because no credential ever reaches it.
        """
        response = db_client.get(ME_URL)

        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"
        assert response.json() == {"detail": "Not authenticated"}

    @pytest.mark.parametrize(
        ("header", "detail"),
        [
            ({"Authorization": "Bearer"}, "Invalid authentication credentials"),
            ({"Authorization": "Basic YWRhOnNlY3JldA=="}, "Not authenticated"),
            ({"Authorization": "Bearer not-a-token"}, "Invalid authentication credentials"),
            ({"Authorization": "Bearer a.b.c"}, "Invalid authentication credentials"),
            ({"Authorization": "bearer lowercase-scheme"}, "Invalid authentication credentials"),
        ],
    )
    def test_endpoint_rejects_unusable_authorization_headers(
        self, db_client: TestClientType, header: dict[str, str], detail: str
    ) -> None:
        """Anything that is not a well-formed bearer credential is refused.

        A request that never presents the Bearer scheme is stopped by
        :data:`~app.core.dependencies.oauth2_scheme`, which has no credential to
        report on. One that does present it reaches the decoder, which reports
        the shared failure detail.
        """
        response = db_client.get(ME_URL, headers=header)

        assert response.status_code == 401
        assert response.json() == {"detail": detail}

    def test_endpoint_rejects_a_deleted_account(
        self, db_client: TestClientType, engine: Engine
    ) -> None:
        """A token stops working the moment its account is deleted."""
        user_id = seed_user(engine, role=UserRole.ANALYST.value, email="ada@example.com")
        with Session(bind=engine) as db:
            db.delete(db.get(User, user_id))
            db.commit()

        response = db_client.get(ME_URL, headers=bearer(user_id))

        assert response.status_code == 401

    def test_endpoint_rejects_a_non_numeric_subject(
        self, db_client: TestClientType
    ) -> None:
        """A subject that is not a user id cannot be resolved to an account."""
        token = create_access_token(subject="not-a-number")

        response = db_client.get(ME_URL, headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 401

    @pytest.mark.parametrize("subject", ["", "   "])
    def test_endpoint_rejects_a_blank_subject(
        self, db_client: TestClientType, subject: str
    ) -> None:
        """An empty subject claim is treated as a failure, not a lookup."""
        token = create_access_token(subject=subject)

        response = db_client.get(ME_URL, headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 401

    def test_endpoint_refuses_an_unrecognised_role(
        self, db_client: TestClientType, engine: Engine
    ) -> None:
        """A role outside the enum is refused before the profile is serialised.

        The column stores plain text, so such a row genuinely exists on disk.
        This is the test that proves the guard, not the database, is what stops
        it from reaching the response model.
        """
        user_id = seed_user(engine, role=UNRECOGNISED_ROLE, email="overlord@example.com")

        response = db_client.get(ME_URL, headers=bearer(user_id))

        assert response.status_code == 403
        assert response.json() == {
            "detail": "Account role is not permitted to access this resource"
        }

    def test_endpoint_refuses_an_unrecognised_role_without_a_stored_row(
        self, db_client: TestClientType, engine: Engine
    ) -> None:
        """The same guard applies when the role never reaches the database."""
        user_id = seed_user(engine, role=UserRole.VIEWER.value, email="ada@example.com")
        db_client.app.dependency_overrides[get_current_user] = lambda: _fabricated_user(
            UNRECOGNISED_ROLE
        )

        response = db_client.get(ME_URL, headers=bearer(user_id))

        assert response.status_code == 403

    def test_role_change_takes_effect_without_reissuing_the_token(
        self, db_client: TestClientType, engine: Engine
    ) -> None:
        """Authorization reads the database, so a promotion applies immediately."""
        user_id = seed_user(engine, role=UserRole.VIEWER.value, email="ada@example.com")
        headers = bearer(user_id)

        assert db_client.get(ME_URL, headers=headers).json()["role"] == "Viewer"

        with Session(bind=engine) as db:
            account = db.get(User, user_id)
            account.role = UserRole.ADMIN.value
            db.commit()

        assert db_client.get(ME_URL, headers=headers).json()["role"] == "Admin"

    def test_demotion_takes_effect_immediately(
        self, db_client: TestClientType, engine: Engine
    ) -> None:
        """Losing a role does not require waiting for the token to expire."""
        user_id = seed_user(engine, role=UserRole.ADMIN.value, email="ada@example.com")
        headers = bearer(user_id)

        assert db_client.get(ME_URL, headers=headers).status_code == 200

        with Session(bind=engine) as db:
            account = db.get(User, user_id)
            account.role = UserRole.VIEWER.value
            db.commit()

        assert db_client.get(ME_URL, headers=headers).json()["role"] == "Viewer"
