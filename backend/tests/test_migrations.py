"""Focused tests for the initial Alembic revision.

The revision is generated from ``Base.metadata``, so these tests pin both sides
of that contract: the metadata the autogenerator sees, and the schema an actual
``upgrade head`` produces on a throwaway SQLite database. SQLite is used only as
a local stand-in for the migration runner; the DDL itself is generic SQLAlchemy
and contains no SQLite-specific operations. No configured database is opened.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Inspector

from app import models
from app.core import config as config_module
from app.core.config import Settings, get_settings
from app.database.base import Base

BACKEND_DIR: Path = Path(__file__).resolve().parents[1]
INITIAL_REVISION: str = "3b8caabe1c5f"


class TableContract(NamedTuple):
    """Expected structural shape of one migrated table."""

    columns: tuple[str, ...]
    primary_key: tuple[str, ...]
    foreign_keys: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...]
    unique: tuple[tuple[str, ...], ...]


EXPECTED_SCHEMA: dict[str, TableContract] = {
    "users": TableContract(
        ("id", "name", "email", "password_hash", "role", "created_at"),
        ("id",),
        (),
        (("email",),),
    ),
    "uploaded_logs": TableContract(
        ("id", "filename", "file_path", "upload_status", "uploaded_by", "created_at"),
        ("id",),
        ((("uploaded_by",), "users", ("id",)),),
        (),
    ),
    "predictions": TableContract(
        ("id", "upload_id", "attack_type", "confidence", "severity", "created_at"),
        ("id",),
        ((("upload_id",), "uploaded_logs", ("id",)),),
        (),
    ),
    "reports": TableContract(
        ("id", "prediction_id", "report_path", "generated_at"),
        ("id",),
        ((("prediction_id",), "predictions", ("id",)),),
        (),
    ),
    "audit_logs": TableContract(
        ("id", "user_id", "action_performed", "timestamp", "ip_address"),
        ("id",),
        ((("user_id",), "users", ("id",)),),
        (),
    ),
    "model_versions": TableContract(
        (
            "id",
            "version_tag",
            "model_type",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "is_active",
            "created_at",
        ),
        ("id",),
        (),
        (),
    ),
}


def _metadata_contract(table_name: str) -> TableContract:
    """Read one table's structure from the shared declarative metadata.

    Args:
        table_name: Name of the mapped table to inspect.

    Returns:
        TableContract: Structural contract derived from the ORM metadata.
    """
    table = Base.metadata.tables[table_name]
    foreign_keys = tuple(
        (
            (foreign_key.parent.name,),
            foreign_key.target_fullname.split(".")[0],
            (foreign_key.target_fullname.split(".")[1],),
        )
        for foreign_key in table.foreign_keys
    )
    return TableContract(
        tuple(column.name for column in table.columns),
        tuple(column.name for column in table.primary_key.columns),
        foreign_keys,
        tuple(
            tuple(column.name for column in constraint.columns)
            for constraint in table.constraints
            if constraint.__class__.__name__ == "UniqueConstraint"
        ),
    )


def _migrated_contract(inspector: Inspector, table_name: str) -> TableContract:
    """Read one table's structure back from a migrated database.

    Args:
        inspector: SQLAlchemy ``Inspector`` bound to the temporary database.
        table_name: Name of the table to inspect.

    Returns:
        TableContract: Structural contract reflected from the database.
    """
    foreign_keys = tuple(
        (
            tuple(key["constrained_columns"]),
            key["referred_table"],
            tuple(key["referred_columns"]),
        )
        for key in inspector.get_foreign_keys(table_name)
    )
    return TableContract(
        tuple(column["name"] for column in inspector.get_columns(table_name)),
        tuple(inspector.get_pk_constraint(table_name)["constrained_columns"]),
        foreign_keys,
        tuple(
            tuple(unique["column_names"])
            for unique in inspector.get_unique_constraints(table_name)
        ),
    )


@pytest.fixture()
def migration_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Config, str]:
    """Build an Alembic config bound to a database inside the pytest temp dir.

    ``Settings`` accepts PostgreSQL URLs only, so the copied settings object
    changes its URL after validation. ``env.py`` imports ``get_settings`` when a
    command runs, which lets the monkeypatched getter supply this private URL
    without touching ``.env``, the configured database or the cached settings
    used by the other tests. The timezone key is removed in memory because this
    Windows environment has no ``UTC`` zoneinfo database.

    Args:
        tmp_path: Pytest-provided temporary directory for this test.
        monkeypatch: Pytest-provided patcher, reverted after the test.

    Returns:
        tuple[Config, str]: Alembic config and the temporary SQLite URL.
    """
    url: str = "sqlite+pysqlite:///" + (tmp_path / "migration.db").as_posix()
    test_settings: Settings = get_settings().model_copy(update={"DATABASE_URL": url})
    monkeypatch.setattr(config_module, "get_settings", lambda: test_settings)

    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.file_config.remove_option(config.config_ini_section, "timezone")
    return config, url


@pytest.mark.parametrize("table_name", sorted(EXPECTED_SCHEMA))
def test_revision_input_metadata_matches_expected_schema(table_name: str) -> None:
    """The autogenerated revision may only see the six declared tables.

    Args:
        table_name: Table whose metadata shape is checked.
    """
    assert set(Base.metadata.tables) == set(EXPECTED_SCHEMA)
    assert models.User.__tablename__ in EXPECTED_SCHEMA
    assert _metadata_contract(table_name) == EXPECTED_SCHEMA[table_name]


def test_initial_revision_is_the_only_head(
    migration_environment: tuple[Config, str],
) -> None:
    """The migration chain consists of one root revision.

    Args:
        migration_environment: Temporary Alembic configuration and database URL.
    """
    config, _ = migration_environment
    scripts = ScriptDirectory.from_config(config)
    assert scripts.get_current_head() == INITIAL_REVISION
    assert scripts.get_revisions(INITIAL_REVISION)[0].down_revision is None


def test_upgrade_creates_the_declared_schema(
    migration_environment: tuple[Config, str],
) -> None:
    """``upgrade head`` creates exactly the six tables with the expected shape.

    Args:
        migration_environment: Temporary Alembic configuration and database URL.
    """
    config, url = migration_environment
    command.upgrade(config, "head")

    engine = create_engine(url)
    try:
        inspector = inspect(engine)
        migrated = {
            name
            for name in inspector.get_table_names()
            if name != "alembic_version"
        }
        assert migrated == set(EXPECTED_SCHEMA)
        for table_name in EXPECTED_SCHEMA:
            assert (
                _migrated_contract(inspector, table_name) == EXPECTED_SCHEMA[table_name]
            )
        with engine.connect() as connection:
            version = connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar()
            assert version == INITIAL_REVISION
            differences = compare_metadata(
                MigrationContext.configure(connection), Base.metadata
            )
            assert differences == []
    finally:
        engine.dispose()


def test_downgrade_removes_tables_and_upgrade_is_repeatable(
    migration_environment: tuple[Config, str],
) -> None:
    """``downgrade base`` undoes the revision and a second upgrade still works.

    Args:
        migration_environment: Temporary Alembic configuration and database URL.
    """
    config, url = migration_environment
    command.upgrade(config, "head")
    command.downgrade(config, "base")

    engine = create_engine(url)
    try:
        assert inspect(engine).get_table_names() == ["alembic_version"]
    finally:
        engine.dispose()

    command.upgrade(config, "head")
    engine = create_engine(url)
    try:
        tables = {
            name
            for name in inspect(engine).get_table_names()
            if name != "alembic_version"
        }
        assert tables == set(EXPECTED_SCHEMA)
    finally:
        engine.dispose()
