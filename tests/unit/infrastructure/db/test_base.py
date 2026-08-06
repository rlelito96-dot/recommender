from sqlalchemy.orm import DeclarativeBase

from app.infrastructure.db.base import Base


def test_base_is_declarative_base_subclass() -> None:
    assert issubclass(Base, DeclarativeBase)
