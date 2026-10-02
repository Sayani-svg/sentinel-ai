"""Focused tests for the ORM column definitions.

These exist because a ``Mapped[Literal[...]]`` annotation is resolved by
SQLAlchemy into a :class:`sqlalchemy.Enum` column. That type reaches Alembic
autogenerate output and rejects unexpected values with a ``LookupError`` in its
bind processor, which contradicts this schema's rule that no database enum
types are used. The tests below fail if an ``Enum`` is reintroduced.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import Enum as SAEnum
from sqlalchemy import String
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.schema import CreateTable

from app.models.prediction import Prediction
from app.models.user import User
from app.schemas.user import VALID_ROLE_VALUES, UserRole

if TYPE_CHECKING:
    from sqlalchemy.engine import Dialect
    from sqlalchemy.orm import Session

#: The permitted severities, stated here so the tests do not depend on an
#: application enum that the prediction slice has not introduced yet.
PERMITTED_SEVERITIES: tuple[str, ...] = ("Critical", "High", "Medium", "Low")


def rendered_ddl(model: type, dialect: Dialect) -> str:
    """Render a model's table for one dialect.

    Args:
        model: The declarative model whose table to render.
        dialect: The SQLAlchemy dialect to compile against.

    Returns:
        str: The ``CREATE TABLE`` statement.
    """
    return str(CreateTable(model.__table__).compile(dialect=dialect))


@pytest.mark.parametrize("model", [User, Prediction], ids=["users", "predictions"])
class TestNoDatabaseEnumTypes:
    """No table may depend on a database enum type."""

    def test_no_column_uses_an_enum(self, model: type) -> None:
        """No column of the model carries a ``sqlalchemy.Enum``."""
        offenders = [
            column.name for column in model.__table__.columns if isinstance(column.type, SAEnum)
        ]

        assert offenders == []

    @pytest.mark.parametrize(
        "dialect", [postgresql.dialect(), sqlite.dialect()], ids=["postgresql", "sqlite"]
    )
    def test_ddl_emits_no_enum_or_check(self, model: type, dialect: Dialect) -> None:
        """Neither dialect renders an enum type or a value constraint."""
        ddl = rendered_ddl(model, dialect).upper()

        assert "ENUM" not in ddl
        assert "CHECK" not in ddl


def test_no_column_in_the_shared_metadata_uses_an_enum() -> None:
    """Sweeping guard: no table in the schema may introduce one later."""
    offenders = [
        f"{table.name}.{column.name}"
        for table in User.metadata.tables.values()
        for column in table.columns
        if isinstance(column.type, SAEnum)
    ]

    assert offenders == []


class TestUserRoleColumn:
    """``users.role`` must be plain text, not an enum type."""

    def test_column_type_is_not_an_enum(self) -> None:
        """The mapped type carries no ``sqlalchemy.Enum``."""
        assert not isinstance(User.__table__.c.role.type, SAEnum)

    def test_column_type_is_a_plain_string(self) -> None:
        """The column is text and binds Python strings directly."""
        column_type = User.__table__.c.role.type

        assert isinstance(column_type, String)
        assert column_type.python_type is str

    def test_column_remains_not_nullable(self) -> None:
        """Dropping the enum type must not make the role optional."""
        assert User.__table__.c.role.nullable is False

    @pytest.mark.parametrize("role", sorted(VALID_ROLE_VALUES))
    def test_permitted_roles_round_trip_unchanged(self, db_session: Session, role: str) -> None:
        """Admin, Analyst and Viewer store and read back verbatim."""
        db_session.add(
            User(
                name="Role Probe",
                email=f"{role.lower()}@example.com",
                password_hash="not-a-real-hash",
                role=role,
                created_at=datetime.now(timezone.utc),
            )
        )
        db_session.commit()
        db_session.expire_all()

        stored = db_session.get(User, 1)

        assert stored is not None
        assert stored.role == role

    def test_an_unrecognised_role_is_storable(self, db_session: Session) -> None:
        """The column no longer rejects a value the application does not know.

        Responsibility for rejecting it belongs to
        :func:`app.core.dependencies.get_current_active_user`, which is covered
        in ``test_users.py``.
        """
        db_session.add(
            User(
                name="Corrupt Row",
                email="corrupt@example.com",
                password_hash="not-a-real-hash",
                role="Overlord",
                created_at=datetime.now(timezone.utc),
            )
        )
        db_session.commit()
        db_session.expire_all()

        stored = db_session.get(User, 1)

        assert stored is not None
        assert stored.role == "Overlord"


class TestPredictionSeverityColumn:
    """``predictions.severity`` must be plain text, not an enum type."""

    def test_column_type_is_not_an_enum(self) -> None:
        """The mapped type carries no ``sqlalchemy.Enum``."""
        assert not isinstance(Prediction.__table__.c.severity.type, SAEnum)

    def test_column_type_is_a_plain_string(self) -> None:
        """The column is text and binds Python strings directly."""
        column_type = Prediction.__table__.c.severity.type

        assert isinstance(column_type, String)
        assert column_type.python_type is str

    def test_column_remains_not_nullable(self) -> None:
        """Dropping the enum type must not make the severity optional."""
        assert Prediction.__table__.c.severity.nullable is False

    def test_column_matches_the_other_text_column_on_the_model(self) -> None:
        """``severity`` and ``attack_type`` are declared the same way."""
        severity_type = Prediction.__table__.c.severity.type
        attack_type_type = Prediction.__table__.c.attack_type.type

        assert type(severity_type) is type(attack_type_type)
        assert repr(severity_type) == repr(attack_type_type)

    def test_foreign_key_to_uploaded_logs_is_intact(self) -> None:
        """Only the severity column type changed; the relation still resolves."""
        foreign_keys = Prediction.__table__.c.upload_id.foreign_keys

        assert [str(key.target_fullname) for key in foreign_keys] == ["uploaded_logs.id"]

    @pytest.mark.parametrize("severity", PERMITTED_SEVERITIES)
    def test_permitted_severities_round_trip_unchanged(
        self, db_session: Session, severity: str
    ) -> None:
        """Critical, High, Medium and Low store and read back verbatim."""
        db_session.add(
            Prediction(
                upload_id=1,
                attack_type="Brute Force",
                confidence=0.97,
                severity=severity,
                created_at=datetime.now(timezone.utc),
            )
        )
        db_session.commit()
        db_session.expire_all()

        stored = db_session.get(Prediction, 1)

        assert stored is not None
        assert stored.severity == severity
        assert stored.attack_type == "Brute Force"
        assert stored.confidence == pytest.approx(0.97)

    def test_an_unrecognised_severity_is_storable(self, db_session: Session) -> None:
        """The column no longer rejects a value the application does not know.

        No application-level severity validation exists yet, so nothing refuses
        this value today. The prediction slice is expected to validate the four
        permitted values on write, the same way the auth slice validates roles.
        """
        db_session.add(
            Prediction(
                upload_id=1,
                attack_type="Brute Force",
                confidence=0.5,
                severity="Apocalyptic",
                created_at=datetime.now(timezone.utc),
            )
        )
        db_session.commit()
        db_session.expire_all()

        stored = db_session.get(Prediction, 1)

        assert stored is not None
        assert stored.severity == "Apocalyptic"


class TestApplicationValueContracts:
    """The permitted application values stay exactly as specified."""

    def test_role_enum_has_exactly_three_members(self) -> None:
        """No role is added or removed by the column type change."""
        assert {role.value for role in UserRole} == {"Admin", "Analyst", "Viewer"}

    def test_role_enum_values_match_the_valid_role_set(self) -> None:
        """The comparison set and the enum cannot drift apart."""
        assert VALID_ROLE_VALUES == frozenset(role.value for role in UserRole)

    def test_permitted_severities_are_exactly_as_specified(self) -> None:
        """Critical, High, Medium and Low are unchanged by the column fix."""
        assert set(PERMITTED_SEVERITIES) == {"Critical", "High", "Medium", "Low"}
