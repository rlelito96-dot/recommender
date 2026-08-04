from app.core.config import settings


def test_app_name_is_set_correctly() -> None:
    assert settings.app_name == "recommender-system"
