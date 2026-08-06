from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.infrastructure.db.session import SessionLocal, engine


def test_engine_is_created() -> None:
    assert engine is not None
    assert isinstance(engine, Engine)


def test_engine_url_uses_postgresql() -> None:
    assert str(engine.url).startswith("postgresql")


def test_engine_url_matches_settings() -> None:
    assert settings.database_url.startswith(str(engine.url).split("://")[0])


def test_session_local_is_configured() -> None:
    assert SessionLocal is not None
    assert isinstance(SessionLocal, sessionmaker)


def test_session_local_creates_session() -> None:
    session = SessionLocal()
    assert session is not None
    session.close()
