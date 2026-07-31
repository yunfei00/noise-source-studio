"""Inference service boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from noise_source_studio.domain.models import SignalPreview


@runtime_checkable
class InferenceEngine(Protocol):
    """Contract implemented by future model-specific inference adapters."""

    @property
    def runtime_version(self) -> str:
        """Return the installed runtime version."""
        ...

    def verify_model_package(self, package_path: Path) -> dict[str, Any]:
        """Validate a complete runtime model package."""
        ...

    def load_model(self, package_path: Path, *, device: str = "auto") -> dict[str, Any]:
        """Load and retain a compatible model session."""
        ...

    def inspect_model(self) -> dict[str, Any]:
        """Inspect the retained model session."""
        ...

    def preview_file(self, file_path: Path) -> SignalPreview:
        """Parse a CSV through the runtime parser for GUI preview."""
        ...

    def predict_file(self, file_path: Path) -> Any:
        """Predict labels for one input file."""
        ...

    def export_result(
        self,
        result: Any,
        json_path: Path,
        *,
        contract_path: Path | None = None,
    ) -> tuple[Path, Path | None]:
        """Export a result through the runtime reporting API."""
        ...

    def close(self) -> None:
        """Release engine resources."""
        ...
