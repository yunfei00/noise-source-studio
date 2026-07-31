"""Runtime adapter lifecycle tests without importing the real wheel."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from noise_source_studio.common.paths import ApplicationPaths
from noise_source_studio.infrastructure.config import SettingsManager
from noise_source_studio.infrastructure.inference import RuntimeAdapter
from noise_source_studio.infrastructure.logging import configure_logging
from noise_source_studio.presentation.main_window import MainWindow
from noise_source_studio.services import ModelService, PredictionService


class FakeSession:
    """Track close calls across model switches."""

    created: list[FakeSession] = []

    def __init__(self, checkpoint: Path, device: str) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.closed = False
        self.created.append(self)

    @classmethod
    def load_model(cls, checkpoint: Path, *, device: str) -> FakeSession:
        return cls(Path(checkpoint), device)

    def inspect_model(self) -> dict[str, Any]:
        return {
            "runtime_version": "1.0.0",
            "device": "cpu",
            "prediction_mode": "multilabel",
            "labels": ["one", "two"],
        }

    def predict_file(self, file_path: Path) -> object:
        return SimpleNamespace(csv_path=str(file_path))

    def close(self) -> None:
        self.closed = True


def _fake_runtime() -> SimpleNamespace:
    return SimpleNamespace(
        RUNTIME_VERSION="1.0.0",
        InferenceSession=FakeSession,
        verify_model_package=lambda package: {"valid": True, "package": str(package)},
        write_prediction_json=lambda result, path: path,
        write_inference_contract=lambda result, session, path: path,
    )


def test_switching_model_closes_previous_session(tmp_path: Path) -> None:
    FakeSession.created.clear()
    adapter = RuntimeAdapter(_fake_runtime())

    adapter.load_model(tmp_path / "first")
    first = FakeSession.created[-1]
    adapter.load_model(tmp_path / "second")

    assert first.closed
    assert adapter.has_session
    assert not FakeSession.created[-1].closed


def test_close_releases_retained_session(tmp_path: Path) -> None:
    FakeSession.created.clear()
    adapter = RuntimeAdapter(_fake_runtime())
    adapter.load_model(tmp_path / "model")
    session = FakeSession.created[-1]

    adapter.close()

    assert session.closed
    assert not adapter.has_session


def test_window_close_releases_runtime_session(
    qapp: QApplication,
    qtbot: QtBot,
    application_paths: ApplicationPaths,
    tmp_path: Path,
) -> None:
    FakeSession.created.clear()
    adapter = RuntimeAdapter(_fake_runtime())
    adapter.load_model(tmp_path / "model")
    session = FakeSession.created[-1]
    manager = SettingsManager(application_paths)
    settings = manager.load()
    model_service = ModelService(settings.model_directory, adapter)
    prediction_service = PredictionService(adapter, settings.output_directory)
    window = MainWindow(
        settings,
        manager,
        configure_logging(settings.log_directory),
        engine=adapter,
        model_service=model_service,
        prediction_service=prediction_service,
    )
    qtbot.addWidget(window)

    window.close()

    assert session.closed
    assert not adapter.has_session
