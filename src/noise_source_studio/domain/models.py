"""GUI-facing models that isolate presentation code from the runtime package."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelRecord:
    """One installed model-package registration."""

    model_name: str
    model_version: str
    package_path: Path
    manifest: dict[str, Any]
    installed_at: str
    is_active: bool
    integrity_status: str

    @property
    def identifier(self) -> str:
        """Return a stable registry identifier."""
        return f"{self.model_name}@{self.model_version}"

    @property
    def display_name(self) -> str:
        """Return the compact name shown in status areas."""
        return f"{self.model_name} {self.model_version}"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelRecord:
        """Build a record from persisted JSON."""
        return cls(
            model_name=str(payload["model_name"]),
            model_version=str(payload["model_version"]),
            package_path=Path(payload["package_path"]),
            manifest=dict(payload["manifest"]),
            installed_at=str(payload["installed_at"]),
            is_active=bool(payload.get("is_active", False)),
            integrity_status=str(payload.get("integrity_status", "unknown")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        payload = asdict(self)
        payload["package_path"] = str(self.package_path)
        return payload


@dataclass(frozen=True, slots=True)
class PackageInspection:
    """Verified model-package metadata returned before import."""

    package_path: Path
    manifest: dict[str, Any]
    verification: dict[str, Any]
    runtime_version: str


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """Runtime details for the session currently retained by the application."""

    record: ModelRecord
    runtime_version: str
    device: str
    prediction_mode: str
    labels: tuple[str, ...]
    inspection: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SignalPreview:
    """Runtime-parsed raw signal plus display-safe diagnostics."""

    source_path: Path
    display_values: tuple[float, ...]
    original_point_count: int
    raw_minimum: float
    raw_maximum: float
    raw_mean: float
    raw_standard_deviation: float
    parser_mode: str
    data_start_line: int
    selected_columns: tuple[int, ...]
    encoding: str
    delimiter: str


@dataclass(frozen=True, slots=True)
class PredictionOutcome:
    """Prediction result paired with task and locked-model metadata."""

    task_id: str
    source_path: Path
    model: ModelRecord
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    result: Any
