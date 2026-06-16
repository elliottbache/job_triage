from unittest.mock import MagicMock

import pytest

from job_triage.db import db_access


class TestBuildDbUrl:
    def test_builds_sqlite_url_from_environment(self, monkeypatch) -> None:
        monkeypatch.setenv("SQLITE_DB_PATH", "private/job_triage.sqlite3")

        result = db_access._build_db_url()

        assert result == f"sqlite:///{db_access.ROOT_DIR}/private/job_triage.sqlite3"

    def test_raises_when_sqlite_db_path_is_missing(self, monkeypatch) -> None:
        monkeypatch.delenv("SQLITE_DB_PATH", raising=False)
        monkeypatch.setattr(db_access, "load_dotenv", MagicMock())

        with pytest.raises(ValueError, match="SQLITE_DB_PATH"):
            db_access._build_db_url()


class TestGetEngine:
    def test_returns_cached_engine(self, monkeypatch) -> None:
        db_access._get_engine.cache_clear()
        monkeypatch.setenv("SQLITE_DB_PATH", ":memory:")

        first = db_access._get_engine()
        second = db_access._get_engine()

        assert first is second
        db_access._get_engine.cache_clear()


class TestGetSession:
    def test_returns_session_bound_to_cached_engine(self, monkeypatch) -> None:
        engine = MagicMock()
        session = MagicMock()

        monkeypatch.setattr(db_access, "_get_engine", MagicMock(return_value=engine))
        monkeypatch.setattr(db_access, "Session", MagicMock(return_value=session))

        result = db_access.get_session()

        assert result is session
        db_access._get_engine.assert_called_once_with()
        db_access.Session.assert_called_once_with(engine)
