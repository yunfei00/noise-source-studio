"""Inference service boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class InferenceEngine(Protocol):
    """Contract implemented by future model-specific inference adapters."""

    def load_model(self, model_path: Path) -> None:
        """Load and validate a compatible model."""
        ...

    def predict_file(self, file_path: Path) -> object:
        """Predict labels for one input file."""
        ...

    def close(self) -> None:
        """Release engine resources."""
        ...
