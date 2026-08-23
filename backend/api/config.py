"""HTTP API configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    environment: Literal["development", "test", "production"] = Field(
        default="development",
        validation_alias="MAGIFF_ENV",
    )
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias="MAGIFF_API_KEY",
    )
    cors_origins_raw: str | None = Field(
        default=None,
        validation_alias="MAGIFF_CORS_ORIGINS",
    )
    agent_model: str = Field(
        default="gpt-5.6-terra",
        validation_alias="OPENAI_AGENT_MODEL",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )
    supabase_url: str | None = Field(
        default=None,
        validation_alias="SUPABASE_URL",
    )
    supabase_secret_key: SecretStr | None = Field(
        default=None,
        validation_alias="SUPABASE_SECRET_KEY",
    )

    @property
    def api_key_value(self) -> str | None:
        return self.api_key.get_secret_value() if self.api_key else None

    @property
    def cors_origins(self) -> list[str]:
        if self.cors_origins_raw is None:
            return (
                ["http://localhost:5173"]
                if self.environment != "production"
                else []
            )
        return [
            value.strip().rstrip("/")
            for value in self.cors_origins_raw.split(",")
            if value.strip()
        ]

    def dependency_status(self) -> dict[str, bool]:
        return {
            "openai": self.openai_api_key is not None,
            "supabase_url": bool(self.supabase_url),
            "supabase_secret": self.supabase_secret_key is not None,
            "api_auth": bool(self.api_key_value),
        }

    def validate_runtime(self) -> None:
        if "*" in self.cors_origins:
            raise RuntimeError(
                "MAGIFF_CORS_ORIGINS must list explicit origins; '*' is not allowed"
            )
        if self.environment != "production":
            return
        missing = [
            name
            for name, present in self.dependency_status().items()
            if not present
        ]
        if missing:
            raise RuntimeError(
                "Missing production configuration: " + ", ".join(missing)
            )
