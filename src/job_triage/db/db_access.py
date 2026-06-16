from functools import cache
from os import getenv

from dotenv import load_dotenv
from sqlalchemy import Engine, MetaData, create_engine
from sqlalchemy.orm import Session

from job_triage._helpers import ROOT_DIR

convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=convention)

_DOTENV_PATH = ROOT_DIR / ".env"


def get_session() -> Session:
    """Return a fresh SQLAlchemy session bound to the application database."""
    return Session(_get_engine())


@cache
def _get_engine() -> Engine:
    """Return the cached process-wide SQLAlchemy engine for the SQLite database."""
    return create_engine(_build_db_url())


def _build_db_url() -> str:
    """Build the SQLite database URL from ``SQLITE_DB_PATH``.

    Raises:
        ValueError: If the required database path variable is missing.
    """
    load_dotenv(dotenv_path=_DOTENV_PATH, override=False)

    db_path = getenv("SQLITE_DB_PATH")

    if db_path is None:
        raise ValueError(".env value for SQLITE_DB_PATH is not being read correctly.")

    return f"sqlite:///{ROOT_DIR}/{db_path}"
