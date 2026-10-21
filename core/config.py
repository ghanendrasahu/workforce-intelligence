"""Central application settings loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Identity
    app_name: str = "Workforce Intelligence API"
    version: str = "0.1.0"
    seed: int = 42

    # Observability
    log_json: bool = False
    log_level: str = "INFO"

    # Artifact paths
    data_dir: Path = Path("artifacts/data")
    models_dir: Path = Path("artifacts/models")
    metrics_dir: Path = Path("artifacts/metrics")

    # Business constants
    currency: str = "INR"
    replacement_cost_multiplier: float = 0.75
    training_cost_per_employee: float = 50_000.0
    coaching_cost_per_employee: float = 15_000.0

    # Azure OpenAI (AI Analyst)
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_api_version: str = "2024-07-18"
    azure_use_cli_credential: bool = False

    @property
    def agent_configured(self) -> bool:
        endpoint_ok = bool(self.azure_openai_endpoint)
        creds_ok = bool(self.azure_openai_api_key) or self.azure_use_cli_credential
        return endpoint_ok and creds_ok and bool(self.azure_openai_deployment)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
