"""Shared test fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from noise_source_studio.common.paths import ApplicationPaths  # noqa: E402


@pytest.fixture
def application_paths(tmp_path: Path) -> ApplicationPaths:
    """Return isolated platform paths so tests never use real user data."""
    return ApplicationPaths(
        data_directory=tmp_path / "data",
        config_directory=tmp_path / "config",
        log_directory=tmp_path / "logs",
        model_directory=tmp_path / "data" / "models",
        output_directory=tmp_path / "data" / "outputs",
        resource_directory=tmp_path / "resources",
    )
