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
    discord_application_id: str | None = Field(
        default=None,
        validation_alias="DISCORD_APPLICATION_ID",
    )
    discord_public_key: SecretStr | None = Field(
        default=None,
        validation_alias="DISCORD_PUBLIC_KEY",
    )
    discord_test_guild_id: str | None = Field(
        default=None,
        validation_alias="DISCORD_TEST_GUILD_ID",
    )
    discord_uai_guild_id: str | None = Field(
        default=None,
        validation_alias="DISCORD_UAI_GUILD_ID",
    )
    discord_uai_enabled: bool = Field(
        default=False,
        validation_alias="DISCORD_UAI_ENABLED",
    )

    @property
    def api_key_value(self) -> str | None:
        return self.api_key.get_secret_value() if self.api_key else None

    @property
    def discord_public_key_value(self) -> str | None:
        return (
            self.discord_public_key.get_secret_value()
            if self.discord_public_key
            else None
        )

    @property
    def discord_configured(self) -> bool:
        return all(
            (
                self.discord_application_id,
                self.discord_public_key_value,
                self.discord_test_guild_id,
            )
        )

    @property
    def discord_partially_configured(self) -> bool:
        values = (
            self.discord_application_id,
            self.discord_public_key_value,
            self.discord_test_guild_id,
        )
        return any(values) and not all(values)

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
        dependencies = {
            "openai": self.openai_api_key is not None,
            "supabase_url": bool(self.supabase_url),
            "supabase_secret": self.supabase_secret_key is not None,
            "api_auth": bool(self.api_key_value),
        }
        if self.discord_configured or self.discord_partially_configured:
            dependencies["discord"] = self.discord_configured
        return dependencies

    def validate_runtime(self) -> None:
        if "*" in self.cors_origins:
            raise RuntimeError(
                "MAGIFF_CORS_ORIGINS must list explicit origins; '*' is not allowed"
            )
        if self.discord_partially_configured:
            raise RuntimeError(
                "Discord requires DISCORD_APPLICATION_ID, DISCORD_PUBLIC_KEY, "
                "and DISCORD_TEST_GUILD_ID together"
            )
        if (
            (self.discord_uai_guild_id or self.discord_uai_enabled)
            and not self.discord_configured
        ):
            raise RuntimeError(
                "Discord UAI configuration requires the application, public "
                "key, and test guild configuration"
            )
        if self.discord_configured:
            if not self.discord_application_id.isdigit():
                raise RuntimeError("DISCORD_APPLICATION_ID must be numeric")
            if not self.discord_test_guild_id.isdigit():
                raise RuntimeError("DISCORD_TEST_GUILD_ID must be numeric")
            if (
                self.discord_uai_guild_id is not None
                and not self.discord_uai_guild_id.isdigit()
            ):
                raise RuntimeError("DISCORD_UAI_GUILD_ID must be numeric")
            if self.discord_uai_enabled and not self.discord_uai_guild_id:
                raise RuntimeError(
                    "DISCORD_UAI_ENABLED requires DISCORD_UAI_GUILD_ID"
                )
            if self.discord_uai_guild_id == self.discord_test_guild_id:
                raise RuntimeError(
                    "Discord test and UAI guild IDs must be different"
                )
            try:
                public_key = bytes.fromhex(self.discord_public_key_value or "")
            except ValueError as error:
                raise RuntimeError(
                    "DISCORD_PUBLIC_KEY must be a hexadecimal Ed25519 public key"
                ) from error
            if len(public_key) != 32:
                raise RuntimeError(
                    "DISCORD_PUBLIC_KEY must contain exactly 32 bytes"
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

    def discord_guild_profile(self, guild_id: str) -> Literal["test", "uai"] | None:
        """Return the enabled feature profile for one Discord guild."""
        if guild_id == self.discord_test_guild_id:
            return "test"
        if (
            self.discord_uai_enabled
            and self.discord_uai_guild_id is not None
            and guild_id == self.discord_uai_guild_id
        ):
            return "uai"
        return None
