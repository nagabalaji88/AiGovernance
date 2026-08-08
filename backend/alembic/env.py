"""Alembic environment.

Two things here are not boilerplate:

1. **The URL comes from the environment**, never from alembic.ini. The same
   migration set must run against local, staging and production with no file
   edits, and a committed connection string is a credential leak waiting to
   happen.

2. **Partitioned tables are excluded from autogenerate.** Alembic's
   autogenerate does not understand declarative partitioning: it sees the
   monthly child partitions as unmanaged tables and helpfully proposes
   `DROP TABLE usage_events_2026_03`. Excluding them means partition
   maintenance stays with the `ensure_partitions` job, which is the only thing
   that should own it.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

#: Tables whose physical children are managed by the partition job.
PARTITIONED_TABLES = {"usage_events", "usage_costs"}


def _database_url() -> str:
    url = os.getenv("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL must be set to run migrations")
    # Alembic runs synchronously; strip the async driver if the application's
    # URL is used verbatim.
    return url.replace("+asyncpg", "+psycopg2")


def include_object(obj, name: str, type_: str, reflected: bool, compare_to) -> bool:  # type: ignore[no-untyped-def]
    """Keep partition children out of autogenerate diffs."""
    if type_ == "table" and reflected:
        for parent in PARTITIONED_TABLES:
            # Children are named `<parent>_YYYY_MM`.
            if name.startswith(f"{parent}_"):
                return False
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        include_object=include_object,
        compare_type=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            # compare_type catches column type drift, which is otherwise
            # invisible until a value silently truncates in production.
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
