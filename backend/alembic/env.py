"""Alembic migration environment for Sentinel AI.

The database URL is always sourced from application settings
(``app.core.config.Settings.DATABASE_URL``); the ``sqlalchemy.url`` key in
``alembic.ini`` is therefore left unset. ``app.models`` is imported so every
ORM model is mapped onto ``app.database.base.Base.metadata`` before Alembic
binds to it, which is what lets ``alembic revision --autogenerate`` see all
project tables.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app.models  # noqa: E402,F401  (registers all ORM models on Base.metadata)
from app.core.config import get_settings  # noqa: E402
from app.database.base import Base  # noqa: E402


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

settings = get_settings()


def _database_url() -> str:
    """Return the live database URL from application settings."""
    return settings.DATABASE_URL


def run_migrations_offline() -> None:
    """Run migrations in offline (SQL script) mode."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in online mode against the configured database."""
    config_section = config.get_section(config.config_ini_section, {})
    config_section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(
        config_section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()