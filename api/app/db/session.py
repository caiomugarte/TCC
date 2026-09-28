from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

_API_ROOT = Path(__file__).resolve().parents[2]


def _normalize_database_url(database_url: str) -> str:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.database:
        return database_url

    database = url.database
    if database == ":memory:" or database.startswith("file:"):
        return database_url

    database_path = Path(database)
    if database_path.is_absolute():
        return database_url

    normalized_path = (_API_ROOT / database_path).resolve()
    return url.set(database=str(normalized_path)).render_as_string(hide_password=False)


DATABASE_URL = _normalize_database_url(os.getenv("DATABASE_URL", "sqlite:///./prumo-dev.db"))
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session() -> Iterator[Session]:
    with SessionLocal() as session:
        yield session
