"""Focused tests for registration, login, hashing and role enforcement.

Persistence is provided by the ``engine``/``db_client`` fixtures in
``conftest.py``, which run against a private in-memory SQLite database, so
nothing here contacts the configured PostgreSQL URL.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.dependencies import (
    get_current_user,
    get_db,
    require_admin,
    require_analyst_or_above,
    require_roles,
    require_viewer_or_above,
)
from app.core.security import (
    BCRYPT_MAX_PASSWORD_BYTES,
    JWT_ISSUER,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.models.user import User
from app.schemas.user import UserRole
from app.services.auth_service import (
    INVALID_CREDENTIALS_MESSAGE,
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
    authenticate_user,
    register_user,
)

REGISTER_URL = "/api/v1/auth/register"
LOGIN_URL = "/api/v1/auth/login"

#: Satisfies the default PASSWORD_MIN_LENGTH of 12.
VALID_PASSWORD = "correct-horse-battery-staple"

#: Every character here encodes to three UTF-8 bytes, so 25 of them exceed the
#: 72-byte bcrypt limit while staying under bcrypt's 72-*character* limit. This
#: is what proves the limit is enforced in bytes rather than characters.
MULTIBYTE_OVERFLOW_PASSWORD = "\u5b89" * 25

#: ``users.role`` is a plain text column: the schema uses no database enum
#: types, so nothing below stops this value from being stored. It therefore
#: models the realistic failure mode the authorization layer has to absorb --
#: a row written by a build whose role set differs from this one's.
UNRECOGNISED_ROLE = "Overlord"


def registration_payload(**overrides: object) -> dict[str, object]:
    """Build a registration body, overriding individual fields.

    Args:
        **overrides: Field values replacing the defaults.

    Returns:
        dict[str, object]: A JSON-serialisable registration payload.
    """
    payload: dict[str, object] = {
        "name": "Ada Lovelace",
        "email": "ada@example.com",
        "password": VALID_PASSWORD,
    }
    payload.update(overrides)
    return payload


def login_payload(**overrides: object) -> dict[str, object]:
    """Build a login body, overriding individual fields.

    Args:
        **overrides: Field values replacing the defaults.

    Returns:
        dict[str, object]: A JSON-serialisable login payload.
    """
    payload: dict[str, object] = {"email": "ada@example.com", "password": VALID_PASSWORD}
    payload.update(overrides)
    return payload


def load_user_row(engine: Engine, email: str) -> dict[str, object]:
    """Read one user row without leaving a session open.

    The result is a plain mapping rather than an ORM instance so that no
    attribute can trigger a lazy load after the session has been closed.

    Args:
        engine: Test database engine.
        email: Email address of the row to read.

    Returns:
        dict[str, object]: The stored column values, or ``{}`` if absent.
    """
    columns = (
        User.id,
        User.name,
        User.email,
        User.password_hash,
        User.role,
        User.created_at,
    )
    with Session(bind=engine) as db:
        row = (
            db.execute(select(*columns).where(User.email == email))
            .one_or_none()
        )
    return dict(row._mapping) if row is not None else {}


def seed_user(
    engine: Engine,
    *,
    role: str,
    email: str = "probe@example.com",
    password: str = VALID_PASSWORD,
) -> int:
    """Insert a user with an arbitrary role and return its id.

    Roles are written directly rather than through the API so that the
    authorization guards can be exercised against values the public API cannot
    produce, including unrecognised ones.

    Args:
        engine: Test database engine.
        role: Role stored on the row, verbatim.
        email: Email address for the new account.
        password: Plaintext password that the row will be seeded with.

    Returns:
        int: The generated primary key.
    """
    with Session(bind=engine) as db:
        user = User(
            name="Probe User",
            email=email,
            password_hash=hash_password(password),
            role=role,
            created_at=datetime.now(timezone.utc),
        )
        db.add(user)
        db.commit()
        return int(user.id)


def auth_headers(user_id: int) -> dict[str, str]:
    """Build an authorization header for a seeded account.

    Args:
        user_id: Account id to embed as the token subject.

    Returns:
        dict[str, str]: A bearer authorization header.
    """
    token = create_access_token(subject=str(user_id))
    return {"Authorization": f"Bearer {token}"}


def fabricated_user(role: str, user_id: int = 4_242) -> User:
    """Build an unsaved account carrying an arbitrary role.

    Used by the guards that bypass the database entirely, so the guard logic can
    be exercised without a row or a session.

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


def _guard_paths() -> tuple[tuple[str, object], ...]:
    """Pair each probe path with the guard protecting it."""
    return (
        ("/viewer", require_viewer_or_above),
        ("/analyst", require_analyst_or_above),
        ("/admin", require_admin),
    )


def _probe_response() -> dict[str, str]:
    """Return a fixed body for the guard probe routes."""
    return {"ok": "true"}


@pytest.fixture()
def rbac_client(engine: Engine) -> Iterator[TestClient]:
    """Yield a client exposing one probe route per role guard.

    Accounts are seeded in the database, so these tests exercise the whole
    dependency chain including the session lookup.

    Args:
        engine: Test database engine.

    Yields:
        TestClient: Client for an app whose routes are guarded by
            :func:`require_viewer_or_above`, :func:`require_analyst_or_above`
            and :func:`require_admin`.
    """
    application = FastAPI()
    for path, guard in _guard_paths():
        application.get(path, dependencies=[Depends(guard)])(_probe_response)

    def override_get_db() -> Iterator[Session]:
        """Serve route handlers from the per-test database."""
        db = Session(bind=engine, expire_on_commit=False)
        try:
            yield db
        finally:
            db.close()

    application.dependency_overrides[get_db] = override_get_db
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture()
def role_probe_client() -> Iterator[Callable[[str], TestClient]]:
    """Yield a factory for clients whose token resolves to a chosen role.

    Bypasses the database entirely by overriding the authentication
    dependency, so any role string can be presented to the guards.

    Yields:
        Callable[[str], TestClient]: Factory taking the role to authenticate as.
    """

    def build(role: str) -> TestClient:
        """Build a client whose authenticated account holds ``role``.

        Args:
            role: Role the overridden dependency will report.

        Returns:
            TestClient: Client bound to the probe application.
        """
        application = FastAPI()
        for path, guard in _guard_paths():
            application.get(path, dependencies=[Depends(guard)])(_probe_response)

        def override_current_user() -> User:
            """Resolve every token to a fabricated account holding ``role``."""
            return fabricated_user(role)

        application.dependency_overrides[get_current_user] = override_current_user
        return TestClient(application)

    yield build


class TestRegistration:
    """``POST /api/v1/auth/register``."""

    def test_registration_returns_the_created_viewer_account(
        self, db_client: TestClient
    ) -> None:
        """A new account is created, returned in full, and issued the Viewer role."""
        response = db_client.post(REGISTER_URL, json=registration_payload())

        assert response.status_code == 201
        body = response.json()
        assert body["name"] == "Ada Lovelace"
        assert body["email"] == "ada@example.com"
        assert body["role"] == UserRole.VIEWER.value
        assert isinstance(body["id"], int)
        datetime.fromisoformat(body["created_at"])

    def test_registration_response_never_carries_the_password_hash(
        self, db_client: TestClient
    ) -> None:
        """The response schema exposes no secret material."""
        body = db_client.post(REGISTER_URL, json=registration_payload()).json()

        assert set(body) == {"id", "name", "email", "role", "created_at"}

    def test_registration_persists_a_verifiable_bcrypt_hash(
        self, db_client: TestClient, engine: Engine
    ) -> None:
        """The password is stored as a salted bcrypt digest at the configured cost."""
        db_client.post(REGISTER_URL, json=registration_payload())

        row = load_user_row(engine, "ada@example.com")
        stored = str(row["password_hash"])

        assert stored != VALID_PASSWORD
        assert not stored.startswith("crypt")
        assert stored.startswith("$2b$")
        assert stored.split("$")[2] == f"{get_settings().BCRYPT_ROUNDS:02d}"
        assert verify_password(VALID_PASSWORD, stored)
        assert not verify_password(VALID_PASSWORD + "x", stored)

    def test_registration_normalises_the_email_address(
        self, db_client: TestClient, engine: Engine
    ) -> None:
        """Surrounding whitespace and casing are stripped before storage."""
        body = db_client.post(
            REGISTER_URL, json=registration_payload(email="  Ada@Example.COM  ")
        ).json()

        assert body["email"] == "ada@example.com"
        assert load_user_row(engine, "ada@example.com")["name"] == "Ada Lovelace"

    def test_registration_trims_the_display_name(self, db_client: TestClient) -> None:
        """A name of only whitespace is rejected rather than stored as an empty string."""
        assert db_client.post(REGISTER_URL, json=registration_payload(name="   ")).status_code == 422

        response = db_client.post(REGISTER_URL, json=registration_payload(name="  Ada  "))
        assert response.json()["name"] == "Ada"

    def test_duplicate_registration_is_rejected(self, db_client: TestClient) -> None:
        """A second registration for the same address fails with 409."""
        db_client.post(REGISTER_URL, json=registration_payload())

        response = db_client.post(REGISTER_URL, json=registration_payload())

        assert response.status_code == 409
        assert response.json() == {
            "detail": "An account already exists for this email address"
        }

    def test_duplicate_registration_is_case_insensitive(self, db_client: TestClient) -> None:
        """Address normalisation prevents ``Ada@`` and ``ada@`` creating two accounts."""
        db_client.post(REGISTER_URL, json=registration_payload(email="ada@example.com"))

        response = db_client.post(
            REGISTER_URL, json=registration_payload(email="ADA@EXAMPLE.COM")
        )

        assert response.status_code == 409

    @pytest.mark.parametrize(
        "smuggled", [{"role": "Admin"}, {"role": UserRole.ADMIN.value}, {"is_admin": True}]
    )
    def test_registration_refuses_caller_supplied_privilege(
        self, db_client: TestClient, engine: Engine, smuggled: dict[str, object]
    ) -> None:
        """Unknown fields are rejected, so a registration cannot grant itself a role."""
        response = db_client.post(
            REGISTER_URL, json=registration_payload(**smuggled)
        )

        assert response.status_code == 422
        assert load_user_row(engine, "ada@example.com") == {}

    def test_registration_does_not_issue_a_token(self, db_client: TestClient) -> None:
        """Registration only creates the account; a separate login is required."""
        body = db_client.post(REGISTER_URL, json=registration_payload()).json()

        assert "access_token" not in body

    def test_registration_rejects_a_short_password(self, db_client: TestClient) -> None:
        """Passwords below PASSWORD_MIN_LENGTH never reach the hashing layer."""
        too_short = "a" * (get_settings().PASSWORD_MIN_LENGTH - 1)

        response = db_client.post(REGISTER_URL, json=registration_payload(password=too_short))

        assert response.status_code == 422
        assert "PASSWORD" in response.text or "Password" in response.text

    def test_registration_accepts_a_password_at_the_minimum_length(
        self, db_client: TestClient
    ) -> None:
        """A password of exactly PASSWORD_MIN_LENGTH is accepted."""
        at_limit = "a" * get_settings().PASSWORD_MIN_LENGTH

        response = db_client.post(REGISTER_URL, json=registration_payload(password=at_limit))

        assert response.status_code == 201

    def test_registration_rejects_a_password_over_the_bcrypt_byte_limit(
        self, db_client: TestClient
    ) -> None:
        """More than 72 bytes is a validation error, not a 500 from bcrypt."""
        too_long = "a" * (BCRYPT_MAX_PASSWORD_BYTES + 1)

        response = db_client.post(REGISTER_URL, json=registration_payload(password=too_long))

        assert response.status_code == 422
        assert str(BCRYPT_MAX_PASSWORD_BYTES) in response.text

    def test_registration_rejects_a_multibyte_password_over_the_byte_limit(
        self, db_client: TestClient
    ) -> None:
        """The limit is measured in UTF-8 bytes, not characters."""
        assert len(MULTIBYTE_OVERFLOW_PASSWORD) < BCRYPT_MAX_PASSWORD_BYTES

        response = db_client.post(
            REGISTER_URL, json=registration_payload(password=MULTIBYTE_OVERFLOW_PASSWORD)
        )

        assert response.status_code == 422

    @pytest.mark.parametrize("email", ["not-an-email", "ada@", "@example.com", "ada@example", "a b@example.com"])
    def test_registration_rejects_malformed_emails(
        self, db_client: TestClient, email: str
    ) -> None:
        """Addresses that fail the shape check are refused."""
        assert db_client.post(REGISTER_URL, json=registration_payload(email=email)).status_code == 422

    def test_registration_requires_every_field(self, db_client: TestClient) -> None:
        """An incomplete body is rejected field by field."""
        for missing in ("name", "email", "password"):
            payload = registration_payload()
            payload.pop(missing)

            response = db_client.post(REGISTER_URL, json=payload)

            assert response.status_code == 422
            assert missing in response.json()["detail"][0]["loc"][-1]


class TestLogin:
    """``POST /api/v1/auth/login``."""

    def test_login_issues_a_bearer_token_for_valid_credentials(
        self, db_client: TestClient
    ) -> None:
        """Valid credentials yield an OAuth2-shaped token response."""
        db_client.post(REGISTER_URL, json=registration_payload())

        response = db_client.post(LOGIN_URL, json=login_payload())

        assert response.status_code == 200
        body = response.json()
        assert body["token_type"] == "bearer"
        assert body["expires_in"] == get_settings().ACCESS_TOKEN_EXPIRE_MINUTES * 60
        assert isinstance(body["access_token"], str)

    def test_issued_token_carries_only_the_account_id_as_subject(
        self, db_client: TestClient, engine: Engine
    ) -> None:
        """Authorization reads the role from the database, so the token omits it."""
        created = db_client.post(REGISTER_URL, json=registration_payload()).json()
        body = db_client.post(LOGIN_URL, json=login_payload()).json()

        payload = decode_access_token(body["access_token"])

        assert payload["sub"] == str(created["id"])
        assert "role" not in payload
        assert "password" not in payload
        assert load_user_row(engine, "ada@example.com")["role"] == UserRole.VIEWER.value

    def test_login_normalises_the_email_address(self, db_client: TestClient) -> None:
        """The registered address can be supplied in any casing."""
        db_client.post(REGISTER_URL, json=registration_payload(email="ada@example.com"))

        response = db_client.post(LOGIN_URL, json=login_payload(email="  ADA@Example.com "))

        assert response.status_code == 200

    def test_login_rejects_a_wrong_password(self, db_client: TestClient) -> None:
        """A wrong password yields 401 with a bearer challenge."""
        db_client.post(REGISTER_URL, json=registration_payload())

        response = db_client.post(LOGIN_URL, json=login_payload(password="wrong-password-entirely"))

        assert response.status_code == 401
        assert response.json() == {"detail": INVALID_CREDENTIALS_MESSAGE}
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_login_rejects_an_unknown_email(self, db_client: TestClient) -> None:
        """An unregistered address is indistinguishable from a wrong password."""
        db_client.post(REGISTER_URL, json=registration_payload())

        response = db_client.post(LOGIN_URL, json=login_payload(email="nobody@example.com"))

        assert response.status_code == 401
        assert response.json() == {"detail": INVALID_CREDENTIALS_MESSAGE}
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_login_rejects_an_unrecognised_role(
        self, db_client: TestClient, engine: Engine
    ) -> None:
        """A stored role this build does not recognise cannot buy a credential."""
        seed_user(engine, role=UNRECOGNISED_ROLE, email="overlord@example.com")

        response = db_client.post(
            LOGIN_URL, json=login_payload(email="overlord@example.com")
        )

        assert response.status_code == 401

    def test_login_rejects_malformed_input(self, db_client: TestClient) -> None:
        """Empty passwords and malformed addresses never reach the service."""
        assert db_client.post(LOGIN_URL, json=login_payload(password="")).status_code == 422
        assert db_client.post(LOGIN_URL, json=login_payload(email="nope")).status_code == 422
        assert (
            db_client.post(
                LOGIN_URL, json=login_payload(password="a" * (BCRYPT_MAX_PASSWORD_BYTES + 1))
            ).status_code
            == 422
        )
        assert db_client.post(LOGIN_URL, json=login_payload(role="Admin")).status_code == 422

    def test_login_requires_both_fields(self, db_client: TestClient) -> None:
        """Neither field is optional."""
        assert db_client.post(LOGIN_URL, json={}).status_code == 422

    def test_registration_then_login_reaches_the_protected_endpoint(
        self, db_client: TestClient
    ) -> None:
        """The token from a fresh registration opens a protected route."""
        created = db_client.post(REGISTER_URL, json=registration_payload()).json()
        body = db_client.post(LOGIN_URL, json=login_payload()).json()

        response = db_client.get(
            "/api/v1/users/me", headers={"Authorization": f"Bearer {body['access_token']}"}
        )

        assert response.status_code == 200
        assert response.json()["id"] == created["id"]


class TestAuthService:
    """Service-level behaviour that is awkward to reach over HTTP."""

    def test_registration_is_idempotent_only_for_distinct_addresses(
        self, engine: Engine
    ) -> None:
        """The service raises its typed error instead of an HTTP exception."""
        with Session(bind=engine) as db:
            register_user(db, name="Ada", email="ada@example.com", password=VALID_PASSWORD)
            with pytest.raises(EmailAlreadyRegisteredError) as excinfo:
                register_user(db, name="Ada", email="ada@example.com", password=VALID_PASSWORD)

        assert excinfo.value.email == "ada@example.com"

    def test_authentication_reports_one_error_for_both_failure_modes(
        self, engine: Engine
    ) -> None:
        """Unknown address and wrong password are the same error to the caller."""
        with Session(bind=engine) as db:
            register_user(db, name="Ada", email="ada@example.com", password=VALID_PASSWORD)

            with pytest.raises(InvalidCredentialsError) as unknown:
                authenticate_user(db, email="nobody@example.com", password=VALID_PASSWORD)
            with pytest.raises(InvalidCredentialsError) as wrong:
                authenticate_user(db, email="ada@example.com", password="nope-nope-nope")

        assert str(unknown.value) == str(wrong.value) == INVALID_CREDENTIALS_MESSAGE

    def test_authentication_returns_the_account(self, engine: Engine) -> None:
        """Correct credentials resolve to the stored user."""
        with Session(bind=engine) as db:
            register_user(db, name="Ada", email="ada@example.com", password=VALID_PASSWORD)
            db.expire_all()

            user = authenticate_user(db, email="ada@example.com", password=VALID_PASSWORD)

        assert user.email == "ada@example.com"
        assert user.role == UserRole.VIEWER.value

    def test_authentication_refuses_an_unrecognised_role(self, engine: Engine) -> None:
        """The service reports the same failure as a wrong password.

        Refusing here stops a credential being issued to an account whose
        authority is undefined. Without it the request would succeed at login and
        then be rejected by every protected route, which is a confusing place to
        discover the problem.
        """
        seed_user(engine, role=UNRECOGNISED_ROLE, email="overlord@example.com")

        with Session(bind=engine) as db:
            with pytest.raises(InvalidCredentialsError):
                authenticate_user(
                    db, email="overlord@example.com", password=VALID_PASSWORD
                )


class TestRoleEnforcement:
    """The role guards in :mod:`app.core.dependencies`."""

    @pytest.mark.parametrize("role", [role.value for role in UserRole])
    def test_viewer_guard_admits_every_role(
        self, rbac_client: TestClient, engine: Engine, role: str
    ) -> None:
        """Any recognised role passes the lowest guard."""
        user_id = seed_user(engine, role=role, email=f"{role}@example.com")

        response = rbac_client.get("/viewer", headers=auth_headers(user_id))

        assert response.status_code == 200
        assert response.json() == {"ok": "true"}

    def test_analyst_guard_rejects_a_viewer(
        self, rbac_client: TestClient, engine: Engine
    ) -> None:
        """A Viewer cannot reach analyst-level routes."""
        user_id = seed_user(engine, role=UserRole.VIEWER.value)

        response = rbac_client.get("/analyst", headers=auth_headers(user_id))

        assert response.status_code == 403
        assert response.json() == {
            "detail": "Insufficient permissions to access this resource"
        }

    @pytest.mark.parametrize("role", [UserRole.ANALYST.value, UserRole.ADMIN.value])
    def test_analyst_guard_admits_analyst_and_admin(
        self, rbac_client: TestClient, engine: Engine, role: str
    ) -> None:
        """Analyst and Admin both clear the analyst guard."""
        user_id = seed_user(engine, role=role, email=f"{role}@example.com")

        assert rbac_client.get("/analyst", headers=auth_headers(user_id)).status_code == 200

    @pytest.mark.parametrize("role", [UserRole.VIEWER.value, UserRole.ANALYST.value])
    def test_admin_guard_rejects_every_lower_role(
        self, rbac_client: TestClient, engine: Engine, role: str
    ) -> None:
        """Only Admin reaches administrative routes."""
        user_id = seed_user(engine, role=role, email=f"{role}@example.com")

        response = rbac_client.get("/admin", headers=auth_headers(user_id))

        assert response.status_code == 403

    def test_admin_guard_admits_admin(self, rbac_client: TestClient, engine: Engine) -> None:
        """An Admin clears the administrative guard."""
        user_id = seed_user(engine, role=UserRole.ADMIN.value)

        assert rbac_client.get("/admin", headers=auth_headers(user_id)).status_code == 200

    def test_unrecognised_role_is_refused_by_every_guard(
        self, role_probe_client: Callable[[str], TestClient]
    ) -> None:
        """A role outside the enum cannot borrow access from a permissive guard."""
        test_client = role_probe_client(UNRECOGNISED_ROLE)

        for path, _ in _guard_paths():
            response = test_client.get(path)

            assert response.status_code == 403
            assert response.json() == {
                "detail": "Account role is not permitted to access this resource"
            }

    @pytest.mark.parametrize(
        ("role", "permitted"),
        [
            (UserRole.VIEWER.value, {"/viewer"}),
            (UserRole.ANALYST.value, {"/viewer", "/analyst"}),
            (UserRole.ADMIN.value, {"/viewer", "/analyst", "/admin"}),
        ],
    )
    def test_guard_matrix_matches_the_role_hierarchy(
        self, role_probe_client: Callable[[str], TestClient],
        role: str,
        permitted: set[str],
    ) -> None:
        """Each role reaches exactly the guards it should, and no others."""
        test_client = role_probe_client(role)

        for path, _ in _guard_paths():
            expected = 200 if path in permitted else 403

            assert test_client.get(path).status_code == expected, f"{role} on {path}"

    def test_guards_reject_anonymous_callers(self, rbac_client: TestClient) -> None:
        """No guard can be satisfied without a bearer token."""
        for path in ("/viewer", "/analyst", "/admin"):
            response = rbac_client.get(path)

            assert response.status_code == 401
            assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_require_roles_needs_at_least_one_role(self) -> None:
        """A guard with an empty role set would admit nobody, so it is refused."""
        with pytest.raises(ValueError, match="at least one role"):
            require_roles()

    def test_require_roles_builds_a_custom_guard(self, engine: Engine) -> None:
        """Arbitrary role sets are supported for routes outside the shipped guards."""
        application = FastAPI()

        @application.get("/custom", dependencies=[Depends(require_roles(UserRole.ANALYST))])
        def probe() -> dict[str, str]:
            """Answer once the analyst-only guard has admitted the caller."""
            return {"ok": "true"}

        def override_get_db() -> Iterator[Session]:
            """Serve the probe route from the per-test database."""
            db = Session(bind=engine, expire_on_commit=False)
            try:
                yield db
            finally:
                db.close()

        application.dependency_overrides[get_db] = override_get_db
        with TestClient(application) as test_client:
            analyst = seed_user(engine, role=UserRole.ANALYST.value, email="a@example.com")
            admin = seed_user(engine, role=UserRole.ADMIN.value, email="b@example.com")

            assert test_client.get("/custom", headers=auth_headers(analyst)).status_code == 200
            assert test_client.get("/custom", headers=auth_headers(admin)).status_code == 403


class TestAuthSurface:
    """The published OpenAPI contract for the auth slice."""

    def test_openapi_documents_the_auth_routes(self, client: TestClient) -> None:
        """All three routes appear under the versioned prefix."""
        paths = client.get("/openapi.json").json()["paths"]

        assert REGISTER_URL in paths
        assert LOGIN_URL in paths
        assert "/api/v1/users/me" in paths

    def test_openapi_marks_the_protected_route_as_bearer_authenticated(
        self, client: TestClient
    ) -> None:
        """The security scheme is advertised on protected routes only."""
        spec = client.get("/openapi.json").json()

        assert spec["components"]["securitySchemes"] == {
            "OAuth2PasswordBearer": {
                "type": "oauth2",
                "flows": {"password": {"scopes": {}, "tokenUrl": LOGIN_URL}},
            }
        }
        assert spec["paths"]["/api/v1/users/me"]["get"]["security"] == [
            {"OAuth2PasswordBearer": []}
        ]
        assert "security" not in spec["paths"][LOGIN_URL]["post"]

    def test_expired_token_is_rejected(self, db_client: TestClient) -> None:
        """A token past its expiry no longer opens a protected route."""
        expired = create_access_token(subject="1", expires_delta=timedelta(seconds=-1))

        response = db_client.get(
            "/api/v1/users/me", headers={"Authorization": f"Bearer {expired}"}
        )

        assert response.status_code == 401

    def test_token_signed_with_another_key_is_rejected(
        self, db_client: TestClient
    ) -> None:
        """A forged signature fails signature verification."""
        now = datetime.now(timezone.utc)
        forged = jwt.encode(
            {
                "sub": "1",
                "iat": now,
                "exp": now + timedelta(minutes=5),
                "jti": "forged",
                "iss": JWT_ISSUER,
            },
            "a-completely-different-signing-key",
            algorithm=get_settings().JWT_ALGORITHM,
        )

        response = db_client.get(
            "/api/v1/users/me", headers={"Authorization": f"Bearer {forged}"}
        )

        assert response.status_code == 401
