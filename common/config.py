"""Centralised configuration management (12-factor style): every tunable
comes from the environment, with sane local defaults, never hardcoded in
service code."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Service URLs
    redis_url: str = "redis://localhost:6379/0"
    scoring_service_url: str = "http://localhost:8001"

    # MLflow / model registry
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    model_name: str = "fraud-triage-model"
    model_alias: str = "champion"
    thresholds_path: str = "data/thresholds.json"

    # Rule engine (FR2)
    hard_block_amount: float = 50_000_000.0
    blocklisted_accounts: set[str] = set()

    # Case service (FR5)
    stepup_case_min_amount: float = 10_000.0

    # Heartbeat / fallback (FR9)
    scoring_timeout_seconds: float = 0.1
    heartbeat_interval_seconds: float = 2.0
    heartbeat_failure_threshold: int = 3

    # Monitoring (FR8)
    psi_alert_threshold: float = 0.2
    latency_p95_alert_ms: float = 150.0

    # Event bus
    stream_payment_decided: str = "payment.decided"


settings = Settings()
