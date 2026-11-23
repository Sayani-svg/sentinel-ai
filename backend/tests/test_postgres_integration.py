"""Integration tests that run only against a migrated disposable PostgreSQL database.

The tests are disabled unless ``SENTINEL_RUN_POSTGRES_TESTS=1``. When enabled,
``SENTINEL_TEST_DATABASE_URL`` must point to a loopback host, user ``postgres``,
database ``sentinel_ci_test`` with ``ENVIRONMENT=testing`` and no query string or
fragment. These constraints keep the destructive per-test cleanup away from
developer and production databases.

The schema is created by Alembic before pytest starts. These tests never create,
drop or migrate tables; they verify what the migration produced and clean up the
rows they write.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Final
from urllib.parse import unquote, urlparse
from uuid import uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from app.core.config import Settings, get_settings
from app.core.dependencies import get_db
from app.core.security import hash_password
from app.database.base import Base
from app.main import create_app
from app.models import (
    AuditLog,
    ModelVersion,
    Prediction,
    Report,
    UploadedLog,
    User,
)
from app.schemas.audit import VALID_AUDIT_ACTION_VALUES
from app.schemas.prediction import AttackType
from app.schemas.user import UserRole

PG_URL_ENV: Final[str] = "SENTINEL_TEST_DATABASE_URL"
PG_OPT_IN_ENV: Final[str] = "SENTINEL_RUN_POSTGRES_TESTS"
PG_ENVIRONMENT_ENV: Final[str] = "ENVIRONMENT"
PG_OPT_IN_VALUE: Final[str] = "1"
PG_TEST_DATABASE: Final[str] = "sentinel_ci_test"
PG_TEST_USER: Final[str] = "postgres"
PG_TEST_ENVIRONMENT: Final[str] = "testing"
PG_ALLOWED_HOSTS: Final[frozenset[str]] = frozenset(
    {"localhost", "127.0.0.1", "::1"}
)
PG_ALLOWED_SCHEMES: Final[frozenset[str]] = frozenset(
    {"postgresql", "postgresql+psycopg2"}
)
HTTPX_REQUIREMENT: Final[str] = "starlette TestClient requires the httpx package"
BACKEND_DIR: Final[Path] = Path(__file__).resolve().parents[1]
ALEMBIC_INI: Final[Path] = BACKEND_DIR / "alembic.ini"

_EXPECTED_FOREIGN_KEYS: Final[frozenset[tuple[str, tuple[str, ...], str, tuple[str, ...]]]] = frozenset(
    {
        ("uploaded_logs", ("uploaded_by",), "users", ("id",)),
        ("predictions", ("upload_id",), "uploaded_logs", ("id",)),
        ("reports", ("prediction_id",), "predictions", ("id",)),
        ("audit_logs", ("user_id",), "users", ("id",)),
    }
)
_CLEANUP_ORDER: Final[tuple[type, ...]] = (
    Report,
    Prediction,
    UploadedLog,
    AuditLog,
    User,
    ModelVersion,
)


class PostgresTargetError(ValueError):
    pass


def _postgres_opt_in_enabled() -> bool:
    return os.environ.get(PG_OPT_IN_ENV, "").strip() == PG_OPT_IN_VALUE


def _parse_postgres_target(raw_url: str) -> tuple[str, str, int | None]:
    parsed = urlparse(raw_url.strip())
    if parsed.query or parsed.fragment:
        raise PostgresTargetError(
            f"{PG_URL_ENV} must not contain query parameters or fragments"
        )
    if parsed.scheme not in PG_ALLOWED_SCHEMES:
        raise PostgresTargetError(
            f"{PG_URL_ENV} must use one of: {', '.join(sorted(PG_ALLOWED_SCHEMES))}"
        )
    if not parsed.hostname or not parsed.path or parsed.path == "/":
        raise PostgresTargetError(f"{PG_URL_ENV} must include host and database name")

    database = unquote(parsed.path.lstrip("/"))
    if not database or "/" in database:
        raise PostgresTargetError(f"{PG_URL_ENV} must target exactly one database")
    return parsed.hostname.strip().lower(), database, parsed.port


def _validate_postgres_target(raw_url: str) -> None:
    parsed = urlparse(raw_url.strip())
    host, database, port = _parse_postgres_target(raw_url)

    if database.lower() == "sentinel_ai":
        raise PostgresTargetError("Refusing to target the local sentinel_ai database")
    if database != PG_TEST_DATABASE:
        raise PostgresTargetError(
            f"Refusing database {database!r}; only {PG_TEST_DATABASE!r} is allowed"
        )
    if host not in PG_ALLOWED_HOSTS:
        raise PostgresTargetError(
            f"Refusing host {host!r}; only loopback test hosts are allowed"
        )
    if port not in (None, 5432):
        raise PostgresTargetError(f"Refusing port {port!r}; expected 5432")
    if parsed.username != PG_TEST_USER:
        raise PostgresTargetError(
            f"Refusing user {parsed.username!r}; expected {PG_TEST_USER!r}"
        )

    environment = os.environ.get(PG_ENVIRONMENT_ENV, "").strip().lower()
    if environment != PG_TEST_ENVIRONMENT:
        raise PostgresTargetError(
            f"Refusing ENVIRONMENT={environment!r}; expected {PG_TEST_ENVIRONMENT!r}"
        )


def _assert_cleanup_permitted(raw_url: str) -> None:
    try:
        _validate_postgres_target(raw_url)
    except PostgresTargetError as exc:
        raise RuntimeError(f"PostgreSQL cleanup refused: {exc}") from exc


def _naive_utcnow() -> datetime:
    """Return the current UTC instant in the application's naive-datetime form."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _make_user(email: str, *, role: UserRole = UserRole.VIEWER) -> User:
    """Build a user row with the given email and role."""
    return User(
        name="PostgreSQL test account",
        email=email,
        password_hash=hash_password("CivicLedger-42!"),
        role=role.value,
        created_at=_naive_utcnow(),
    )


def _make_upload(user_id: int) -> UploadedLog:
    """Build an uploaded-log row owned by ``user_id``."""
    return UploadedLog(
        filename="events.csv",
        file_path="/unused/events.csv",
        upload_status="completed",
        uploaded_by=user_id,
        created_at=_naive_utcnow(),
    )


def _make_prediction(upload_id: int) -> Prediction:
    """Build a prediction row linked to ``upload_id``."""
    return Prediction(
        upload_id=upload_id,
        attack_type=AttackType.DOS.value,
        confidence=0.95,
        severity="High",
        created_at=_naive_utcnow(),
    )


def _make_report(prediction_id: int) -> Report:
    """Build a report row linked to ``prediction_id``."""
    return Report(
        prediction_id=prediction_id,
        report_path="/unused/report.json",
        generated_at=_naive_utcnow(),
    )


def _make_audit(user_id: int, action: str) -> AuditLog:
    """Build an audit row for ``user_id``."""
    return AuditLog(
        user_id=user_id,
        action_performed=action,
        timestamp=_naive_utcnow(),
        ip_address="testclient",
    )


def _unique_email(prefix: str) -> str:
    """Return an address that cannot collide with another test run."""
    return f"{prefix}-{uuid4().hex}@example.test"


@pytest.fixture(scope="module")
def postgres_url() -> str:
    if not _postgres_opt_in_enabled():
        pytest.skip(
            f"PostgreSQL integration tests require {PG_OPT_IN_ENV}={PG_OPT_IN_VALUE}"
        )

    raw_url = os.environ.get(PG_URL_ENV)
    if not raw_url:
        pytest.fail(f"{PG_URL_ENV} must be set when {PG_OPT_IN_ENV}={PG_OPT_IN_VALUE}")

    try:
        _validate_postgres_target(raw_url)
    except PostgresTargetError as exc:
        pytest.fail(str(exc))
    return raw_url


@pytest.fixture(scope="module")
def postgres_engine(postgres_url: str) -> Iterator[Engine]:
    """Yield an engine bound to the migrated PostgreSQL test database."""
    engine = create_engine(postgres_url, poolclass=NullPool, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        yield engine
    finally:
        engine.dispose()


@pytest.fixture()
def postgres_session(postgres_engine: Engine, postgres_url: str) -> Iterator[Session]:
    """Yield a session whose rows are removed after the test."""
    db = Session(bind=postgres_engine, expire_on_commit=False)
    try:
        yield db
    finally:
        db.rollback()
        try:
            _assert_cleanup_permitted(postgres_url)
            for model in _CLEANUP_ORDER:
                db.query(model).delete(synchronize_session=False)
            db.commit()
        finally:
            db.close()


@pytest.fixture()
def postgres_client(
    postgres_engine: Engine,
    postgres_url: str,
    tmp_path: Path,
) -> Iterator[TestClient]:
    """Yield an API client whose routes use the PostgreSQL test database."""
    pytest.importorskip("httpx", reason=HTTPX_REQUIREMENT)

    upload_dir = tmp_path / "uploads"
    report_dir = tmp_path / "reports"
    log_dir = tmp_path / "logs"
    for directory in (upload_dir, report_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    test_settings = get_settings().model_copy(
        update={
            "DATABASE_URL": postgres_url,
            "UPLOAD_DIR": upload_dir,
            "REPORT_DIR": report_dir,
            "LOG_DIR": log_dir,
            "BCRYPT_ROUNDS": 4,
            "ENABLE_FILE_LOGGING": False,
        }
    )

    def override_get_settings() -> Settings:
        """Serve routes with scratch directories and the test database URL."""
        return test_settings

    def override_get_db() -> Iterator[Session]:
        """Serve each request from the shared PostgreSQL test engine."""
        db = Session(bind=postgres_engine, expire_on_commit=False)
        try:
            yield db
        finally:
            db.close()

    application: FastAPI = create_app()
    application.dependency_overrides[get_settings] = override_get_settings
    application.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(application) as test_client:
            yield test_client
    finally:
        application.dependency_overrides.clear()


def test_postgres_tests_skip_without_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CI", "true")
    monkeypatch.delenv(PG_OPT_IN_ENV, raising=False)
    monkeypatch.delenv(PG_URL_ENV, raising=False)

    assert _postgres_opt_in_enabled() is False
    with pytest.raises(pytest.skip.Exception):
        postgres_url.__wrapped__()


def test_postgres_tests_skip_for_non_one_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv(PG_OPT_IN_ENV, "true")
    monkeypatch.setenv(
        PG_URL_ENV,
        "postgresql://postgres:secret@localhost:5432/sentinel_ci_test",
    )

    assert _postgres_opt_in_enabled() is False
    with pytest.raises(pytest.skip.Exception):
        postgres_url.__wrapped__()


def test_postgres_tests_fail_without_url_when_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PG_OPT_IN_ENV, PG_OPT_IN_VALUE)
    monkeypatch.setenv(PG_ENVIRONMENT_ENV, PG_TEST_ENVIRONMENT)
    monkeypatch.delenv(PG_URL_ENV, raising=False)

    with pytest.raises(pytest.fail.Exception):
        postgres_url.__wrapped__()


def test_postgres_target_accepts_disposable_ci_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PG_ENVIRONMENT_ENV, PG_TEST_ENVIRONMENT)
    _validate_postgres_target(
        "postgresql://postgres:secret@localhost:5432/sentinel_ci_test"
    )


def test_postgres_target_rejects_host_query_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PG_ENVIRONMENT_ENV, PG_TEST_ENVIRONMENT)
    with pytest.raises(PostgresTargetError, match="query parameters or fragments"):
        _validate_postgres_target(
            "postgresql://postgres:secret@localhost:5432/"
            "sentinel_ci_test?host=remote.example.com"
        )


def test_postgres_target_rejects_database_query_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PG_ENVIRONMENT_ENV, PG_TEST_ENVIRONMENT)
    with pytest.raises(PostgresTargetError, match="query parameters or fragments"):
        _validate_postgres_target(
            "postgresql://postgres:secret@localhost:5432/"
            "sentinel_ci_test?database=sentinel_ai"
        )


def test_postgres_target_rejects_multiple_query_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PG_ENVIRONMENT_ENV, PG_TEST_ENVIRONMENT)
    with pytest.raises(PostgresTargetError, match="query parameters or fragments"):
        _validate_postgres_target(
            "postgresql://postgres:secret@localhost:5432/"
            "sentinel_ci_test?host=remote.example.com"
            "&database=sentinel_ai&port=5433"
        )


def test_postgres_target_rejects_url_encoded_query_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PG_ENVIRONMENT_ENV, PG_TEST_ENVIRONMENT)
    with pytest.raises(PostgresTargetError, match="query parameters or fragments"):
        _validate_postgres_target(
            "postgresql://postgres:secret@localhost:5432/"
            "sentinel_ci_test?%68ost=remote.example.com"
            "&database=%73entinel_ai"
        )


def test_postgres_target_rejects_url_fragment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PG_ENVIRONMENT_ENV, PG_TEST_ENVIRONMENT)
    with pytest.raises(PostgresTargetError, match="query parameters or fragments"):
        _validate_postgres_target(
            "postgresql://postgres:secret@localhost:5432/"
            "sentinel_ci_test#remote.example.com"
        )


def test_postgres_target_accepts_url_without_query_or_fragment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PG_ENVIRONMENT_ENV, PG_TEST_ENVIRONMENT)
    _validate_postgres_target(
        "postgresql+psycopg2://postgres:secret@localhost:5432/sentinel_ci_test"
    )


def test_postgres_target_rejects_remote_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PG_ENVIRONMENT_ENV, PG_TEST_ENVIRONMENT)
    with pytest.raises(PostgresTargetError, match="host"):
        _validate_postgres_target(
            "postgresql://postgres:secret@db.example.com:5432/sentinel_ci_test"
        )


def test_postgres_target_rejects_unapproved_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PG_ENVIRONMENT_ENV, PG_TEST_ENVIRONMENT)
    with pytest.raises(PostgresTargetError, match="database"):
        _validate_postgres_target(
            "postgresql://postgres:secret@localhost:5432/sentinel_analytics_test"
        )


def test_postgres_target_rejects_sentinel_ai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PG_ENVIRONMENT_ENV, PG_TEST_ENVIRONMENT)
    with pytest.raises(PostgresTargetError, match="sentinel_ai"):
        _validate_postgres_target(
            "postgresql://postgres:secret@localhost:5432/sentinel_ai"
        )


def test_cleanup_is_refused_for_sentinel_ai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PG_ENVIRONMENT_ENV, PG_TEST_ENVIRONMENT)
    with pytest.raises(RuntimeError, match="cleanup refused"):
        _assert_cleanup_permitted(
            "postgresql://postgres:secret@localhost:5432/sentinel_ai"
        )


def test_migrated_schema_matches_models(postgres_engine: Engine) -> None:
    """The Alembic head must have created exactly the schema the ORM declares."""
    inspector = inspect(postgres_engine)
    with postgres_engine.connect() as connection:
        revision = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()
        expected_head = ScriptDirectory.from_config(
            Config(str(ALEMBIC_INI))
        ).get_current_head()
        assert revision == expected_head

        context = MigrationContext.configure(connection)
        assert compare_metadata(context, Base.metadata) == []

    assert set(inspector.get_table_names()) == set(Base.metadata.tables) | {
        "alembic_version"
    }
    for table_name, table in Base.metadata.tables.items():
        actual_columns = {
            column["name"] for column in inspector.get_columns(table_name)
        }
        assert actual_columns == set(table.columns.keys())

    actual_foreign_keys = set()
    for table_name in Base.metadata.tables:
        for foreign_key in inspector.get_foreign_keys(table_name):
            actual_foreign_keys.add(
                (
                    table_name,
                    tuple(foreign_key["constrained_columns"]),
                    foreign_key["referred_table"],
                    tuple(foreign_key["referred_columns"]),
                )
            )
    assert actual_foreign_keys == _EXPECTED_FOREIGN_KEYS

    unique_constraints = [
        set(constraint["column_names"])
        for constraint in inspector.get_unique_constraints("users")
    ]
    assert {"email"} in unique_constraints


def test_unique_email_is_enforced_by_postgresql(
    postgres_session: Session,
) -> None:
    """Two users cannot share an email address, even outside the API."""
    email = _unique_email("unique")
    postgres_session.add(_make_user(email))
    postgres_session.commit()

    postgres_session.add(_make_user(email))
    with pytest.raises(IntegrityError):
        postgres_session.commit()
    postgres_session.rollback()

    stored = postgres_session.execute(
        select(User).where(User.email == email)
    ).all()
    assert len(stored) == 1


def test_foreign_keys_are_enforced_by_postgresql(
    postgres_session: Session,
) -> None:
    """Every child row must reference a real parent row."""
    user = _make_user(_unique_email("fk-user"))
    postgres_session.add(user)
    postgres_session.commit()
    upload = _make_upload(user.id)
    postgres_session.add(upload)
    postgres_session.commit()
    prediction = _make_prediction(upload.id)
    postgres_session.add(prediction)
    postgres_session.commit()

    invalid_rows = (
        _make_upload(999_999_999),
        _make_prediction(999_999_999),
        _make_report(999_999_999),
        _make_audit(999_999_999, "report.retrieved"),
    )
    for row in invalid_rows:
        postgres_session.add(row)
        with pytest.raises(IntegrityError):
            postgres_session.commit()
        postgres_session.rollback()

    assert postgres_session.execute(select(UploadedLog)).all()
    assert postgres_session.execute(select(AuditLog)).all() == []


def test_transaction_rollback_leaves_no_postgresql_rows(
    postgres_session: Session,
) -> None:
    """A rolled-back unit of work cannot leak partially written rows."""
    kept_email = _unique_email("kept")
    discarded_email = _unique_email("discarded")
    kept = _make_user(kept_email)
    postgres_session.add(kept)
    postgres_session.commit()
    kept_id = kept.id

    postgres_session.add(_make_user(discarded_email))
    postgres_session.flush()
    assert postgres_session.execute(
        select(User.id).where(User.email == discarded_email)
    ).all()
    postgres_session.rollback()

    assert postgres_session.get(User, kept_id) is not None
    assert (
        postgres_session.execute(
            select(User.id).where(User.email == discarded_email)
        ).all()
        == []
    )


def test_fk_chain_joins_across_all_related_tables(
    postgres_session: Session,
) -> None:
    """The upload, prediction, report and audit records join through their FKs."""
    email = _unique_email("joins")
    user = _make_user(email)
    postgres_session.add(user)
    postgres_session.commit()

    upload = _make_upload(user.id)
    postgres_session.add(upload)
    postgres_session.commit()
    prediction = _make_prediction(upload.id)
    postgres_session.add(prediction)
    postgres_session.commit()
    report = _make_report(prediction.id)
    audit = _make_audit(user.id, "report.generated")
    postgres_session.add_all((report, audit))
    postgres_session.commit()

    row = postgres_session.execute(
        select(User.email, UploadedLog.filename, Prediction.attack_type, Report.id)
        .join(UploadedLog, UploadedLog.uploaded_by == User.id)
        .join(Prediction, Prediction.upload_id == UploadedLog.id)
        .join(Report, Report.prediction_id == Prediction.id)
        .where(User.email == email)
    ).one()
    assert row == (email, "events.csv", AttackType.DOS.value, report.id)

    audit_email = postgres_session.execute(
        select(User.email, AuditLog.action_performed)
        .join(AuditLog, AuditLog.user_id == User.id)
        .where(AuditLog.id == audit.id)
    ).one()
    assert audit_email == (email, "report.generated")


def test_core_api_flow_against_postgresql(
    postgres_client: TestClient,
    postgres_session: Session,
) -> None:
    """Registration, RBAC, upload, prediction, reports and audit work on PG."""
    viewer_email = _unique_email("viewer")
    password = "CivicLedger-42!"

    response = postgres_client.post(
        "/api/v1/auth/register",
        json={
            "name": "PostgreSQL Viewer",
            "email": viewer_email,
            "password": password,
        },
    )
    assert response.status_code == 201

    response = postgres_client.post(
        "/api/v1/auth/login",
        json={"email": viewer_email, "password": "wrong-password"},
    )
    assert response.status_code == 401

    response = postgres_client.post(
        "/api/v1/auth/login",
        json={"email": viewer_email, "password": password},
    )
    assert response.status_code == 200
    viewer_headers = {
        "Authorization": f"Bearer {response.json()['access_token']}"
    }

    response = postgres_client.get("/api/v1/users/me", headers=viewer_headers)
    assert response.status_code == 200
    assert response.json()["email"] == viewer_email
    viewer_id = response.json()["id"]

    csv_bytes = (
        b"Flow Duration,Total Bytes,Total Packets,Avg Packet Size,Packet Rate\n"
        b"1000,2000000,3000,64,2000\n"
        b"1200,2100000,3100,64,2100\n"
    )
    response = postgres_client.post(
        "/api/v1/upload/logs",
        headers=viewer_headers,
        files={"file": ("events.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 201
    upload_id = response.json()["id"]
    assert response.json()["upload_status"] == "completed"

    response = postgres_client.post(
        "/api/v1/predict",
        headers=viewer_headers,
        json={"upload_id": upload_id},
    )
    assert response.status_code == 403

    analyst = _make_user(_unique_email("analyst"), role=UserRole.ANALYST)
    postgres_session.add(analyst)
    postgres_session.commit()

    response = postgres_client.post(
        "/api/v1/auth/login",
        json={"email": analyst.email, "password": "CivicLedger-42!"},
    )
    assert response.status_code == 200
    analyst_headers = {
        "Authorization": f"Bearer {response.json()['access_token']}"
    }

    response = postgres_client.post(
        "/api/v1/predict",
        headers=analyst_headers,
        json={"upload_id": upload_id},
    )
    assert response.status_code == 201
    prediction_id = response.json()["id"]
    assert response.json()["upload_id"] == upload_id
    assert response.json()["attack_type"] in {
        attack_type.value for attack_type in AttackType
    }

    response = postgres_client.post(
        "/api/v1/reports",
        headers=analyst_headers,
        json={"prediction_id": prediction_id},
    )
    assert response.status_code == 201
    report_id = response.json()["id"]
    assert response.json()["prediction_id"] == prediction_id
    assert response.json()["document"]["recommendations"]

    response = postgres_client.get(
        f"/api/v1/reports/{report_id}",
        headers=viewer_headers,
        params={"format": "json"},
    )
    assert response.status_code == 200
    assert response.json()["id"] == report_id

    response = postgres_client.get(
        f"/api/v1/reports/{report_id}",
        headers=viewer_headers,
        params={"format": "csv"},
    )
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]

    postgres_session.expire_all()
    analyst_actions = sorted(
        row[0]
        for row in postgres_session.execute(
            select(AuditLog.action_performed).where(AuditLog.user_id == analyst.id)
        ).all()
    )
    viewer_actions = sorted(
        row[0]
        for row in postgres_session.execute(
            select(AuditLog.action_performed).where(AuditLog.user_id == viewer_id)
        ).all()
    )
    assert analyst_actions == ["report.generated"]
    assert viewer_actions == ["report.downloaded", "report.retrieved"]
    assert set(analyst_actions + viewer_actions) == VALID_AUDIT_ACTION_VALUES

    joined = postgres_session.execute(
        select(UploadedLog.filename, Prediction.attack_type, Report.id)
        .join(Prediction, Prediction.upload_id == UploadedLog.id)
        .join(Report, Report.prediction_id == Prediction.id)
        .where(UploadedLog.id == upload_id)
    ).one()
    assert joined[0] == "events.csv"
    assert joined[1] in {attack_type.value for attack_type in AttackType}
    assert joined[2] == report_id
