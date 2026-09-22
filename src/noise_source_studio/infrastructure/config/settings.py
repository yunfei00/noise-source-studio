"""Typed application settings and JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from noise_source_studio.common.exceptions import ConfigurationError
from noise_source_studio.common.paths import ApplicationPaths
from noise_source_studio.version import APPLICATION_NAME, __version__


class AppSettings(BaseSettings):
    """Runtime settings with optional ``NSS_`` environment overrides."""

    model_config = SettingsConfigDict(env_prefix="NSS_", extra="ignore")

    application_name: str = APPLICATION_NAME
    application_version: str = __version__
    window_width: int = Field(default=1440, ge=1180)
    window_height: int = Field(default=900, ge=720)
    output_directory: Path
    model_directory: Path
    log_directory: Path
    log_level: str = "INFO"
    theme: str = "light"
    device_preference: str = "auto"
    allow_cpu_fallback: bool = True

    @field_validator("device_preference")
    @classmethod
    def validate_device_preference(cls, value: str) -> str:
        """Accept automatic, CPU, or a concrete non-negative CUDA index."""
        normalized = value.strip().lower()
        if normalized in {"auto", "cpu"}:
            return normalized
        if normalized.startswith("cuda:") and normalized[5:].isdigit():
            return f"cuda:{int(normalized[5:])}"
        raise ValueError("device_preference 必须是 auto、cpu 或 cuda:N")


class SettingsManager:
    """Load and persist non-sensitive application settings."""

    def __init__(self, paths: ApplicationPaths) -> None:
        self.paths = paths

    def default_settings(self) -> AppSettings:
        """Return defaults derived from the active platform paths."""
        return AppSettings(
            output_directory=self.paths.output_directory,
            model_directory=self.paths.model_directory,
            log_directory=self.paths.log_directory,
        )

    def load(self) -> AppSettings:
        """Load settings, creating the initial configuration when necessary."""
        self.paths.ensure_directories()
        defaults = self.default_settings()
        if not self.paths.config_file.exists():
            self.save(defaults)
            return defaults

        try:
            raw_data: dict[str, Any] = json.loads(
                self.paths.config_file.read_text(encoding="utf-8")
            )
            if (
                "device_preference" not in raw_data
                and "default_device" in raw_data
            ):
                raw_data["device_preference"] = raw_data.pop("default_device")
            merged = defaults.model_dump()
            merged.update(raw_data)
            settings = AppSettings.model_validate(merged)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise ConfigurationError("应用配置无法读取，请检查设置文件。") from exc

        self._ensure_runtime_directories(settings)
        return settings

    def save(self, settings: AppSettings) -> None:
        """Persist settings as readable UTF-8 JSON."""
        self.paths.ensure_directories()
        self._ensure_runtime_directories(settings)
        try:
            self.paths.config_file.write_text(
                settings.model_dump_json(indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            raise ConfigurationError("应用配置无法保存，请检查目录权限。") from exc

    @staticmethod
    def _ensure_runtime_directories(settings: AppSettings) -> None:
        for directory in (
            settings.output_directory,
            settings.model_directory,
            settings.log_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)
