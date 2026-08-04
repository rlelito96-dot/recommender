from enum import Enum

from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    app_name: str = "recommender-system"
    secret_key: str
    access_token_expire_minutes: int
    algorithm: str
    debug: bool = False
    environment: Environment = Environment.DEVELOPMENT
    postgres_user: str
    postgres_password: str
    postgres_db: str
    database_url: str
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    model_path: str
    model_version: str


settings: Settings = Settings()
