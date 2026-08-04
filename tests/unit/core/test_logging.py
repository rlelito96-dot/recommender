import logging

from app.core.logging import configure_logging


def test_configure_logging_sets_info_level() -> None:
    configure_logging()
    assert logging.getLogger().level == logging.INFO
