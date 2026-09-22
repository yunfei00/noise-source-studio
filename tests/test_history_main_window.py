"""Main-window history routing restores original pages without inference."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from noise_source_studio.common.paths import ApplicationPaths
from noise_source_studio.domain.batch import BatchPredictionTask
from noise_source_studio.domain.device import DeviceInfo, DeviceProbeReport
from noise_source_studio.domain.history import (
    HistoryStatus,
    IntegrityStatus,
    TaskHistoryRecord,
    TaskType,
)
from noise_source_studio.domain.models import ModelRecord, PredictionOutcome
from noise_source_studio.domain.validation import ValidationTask
from noise_source_studio.infrastructure.config import SettingsManager
from noise_source_studio.infrastructure.logging import configure_logging
from noise_source_studio.presentation import main_window as main_window_module
from noise_source_studio.presentation.main_window import MainWindow
from noise_source_studio.services import (
    DeviceService,
    HistoryService,
)


def _window(
    application_paths: ApplicationPaths,
    history: HistoryService,
) -> MainWindow:
    manager = SettingsManager(application_paths)
    settings = manager.load()
    report = DeviceProbeReport(
        devices=(DeviceInfo("cpu", "CPU", "cpu", True, True),),
        torch_cuda_available=False,
        cuda_runtime_version="",
        torch_version="test",
        cuda_device_count=0,
        error_message="",
    )
    return MainWindow(
        settings,
        manager,
        configure_logging(settings.log_directory),
        history_service=history,
        device_service=DeviceService(SimpleNamespace(probe_devices=lambda: report)),
    )


def _single_outcome(root: Path) -> PredictionOutcome:
    started = datetime.now(UTC) - timedelta(milliseconds=50)
    model = ModelRecord(
        "history-model",
        "1.0",
        root / "model",
        {"checkpoint_sha256": "abc"},
        started.isoformat(),
        True,
        "ok",
    )
    return PredictionOutcome(
        task_id="single-history-route",
        source_path=root / "source.csv",
        model=model,
        started_at=started,
        completed_at=datetime.now(UTC),
        duration_seconds=0.05,
        result={
            "runtime_version": "1.0",
            "device": "cpu",
            "decision_mode": "multilabel",
            "labels": ["fan"],
            "multilabel_probabilities": [0.9],
            "predicted_sources": ["fan"],
            "predicted_combination": "fan",
            "decoded_label_vector": [1],
        },
    )


def test_open_single_history_routes_without_prediction(
    qtbot,
    application_paths: ApplicationPaths,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    history = HistoryService(
        application_paths.history_database,
        application_paths.output_directory,
    )
    history.complete_single(_single_outcome(application_paths.data_directory))
    window = _window(application_paths, history)
    qtbot.addWidget(window)
    predict_calls: list[Path] = []
    monkeypatch.setattr(
        window.prediction_service,
        "predict_file",
        lambda path, *_args, **_kwargs: predict_calls.append(path),
    )

    window._open_history_task("single-history-route")
    qtbot.waitUntil(lambda: window.page_stack.currentIndex() == 1, timeout=3000)

    assert window._single_prediction_page.current_outcome is not None
    assert predict_calls == []


def test_open_batch_history_routes_without_prediction(
    qtbot,
    application_paths: ApplicationPaths,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    history = HistoryService(
        application_paths.history_database,
        application_paths.output_directory,
    )
    result_directory = application_paths.output_directory / "batch_existing"
    result_directory.mkdir(parents=True)
    record = TaskHistoryRecord(
        task_id="batch-history-route",
        task_type=TaskType.BATCH,
        task_name="Batch",
        status=HistoryStatus.COMPLETED,
        created_at=datetime.now(UTC).isoformat(),
        result_directory=str(result_directory),
        integrity_status=IntegrityStatus.OK,
    )
    history.repository.upsert_task(record)
    window = _window(application_paths, history)
    qtbot.addWidget(window)
    task = BatchPredictionTask("Restored batch", result_directory)
    calls: list[Path] = []

    def load(directory: Path) -> BatchPredictionTask:
        calls.append(directory)
        return task

    monkeypatch.setattr(window.batch_prediction_service, "load_history", load)
    window._open_history_task(record.task_id)
    qtbot.waitUntil(lambda: window.page_stack.currentIndex() == 2, timeout=3000)
    assert window.batch_task is task
    assert calls == [result_directory]


def test_open_validation_history_routes_without_prediction(
    qtbot,
    application_paths: ApplicationPaths,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    history = HistoryService(
        application_paths.history_database,
        application_paths.output_directory,
    )
    result_directory = application_paths.output_directory / "validation_existing"
    result_directory.mkdir(parents=True)
    record = TaskHistoryRecord(
        task_id="validation-history-route",
        task_type=TaskType.VALIDATION,
        task_name="Validation",
        status=HistoryStatus.COMPLETED,
        created_at=datetime.now(UTC).isoformat(),
        result_directory=str(result_directory),
        integrity_status=IntegrityStatus.OK,
    )
    history.repository.upsert_task(record)
    window = _window(application_paths, history)
    qtbot.addWidget(window)
    task = ValidationTask(
        manifest_path=application_paths.data_directory / "manifest.csv",
        data_root=application_paths.data_directory,
        output_directory=result_directory,
    )
    calls: list[Path] = []

    def load(directory: Path) -> ValidationTask:
        calls.append(directory)
        return task

    monkeypatch.setattr(window.validation_service, "load_history", load)
    window._open_history_task(record.task_id)
    qtbot.waitUntil(lambda: window.page_stack.currentIndex() == 3, timeout=3000)
    assert window.validation_task is task
    assert calls == [result_directory]


def test_history_initialization_failure_does_not_block_application(
    qtbot,
    application_paths: ApplicationPaths,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    manager = SettingsManager(application_paths)
    settings = manager.load()

    class BrokenHistoryService:
        def __init__(self, *_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            raise RuntimeError("database corrupt")

    monkeypatch.setattr(main_window_module, "HistoryService", BrokenHistoryService)
    report = DeviceProbeReport(
        devices=(DeviceInfo("cpu", "CPU", "cpu", True, True),),
        torch_cuda_available=False,
        cuda_runtime_version="",
        torch_version="test",
        cuda_device_count=0,
        error_message="",
    )
    window = MainWindow(
        settings,
        manager,
        configure_logging(settings.log_directory),
        device_service=DeviceService(SimpleNamespace(probe_devices=lambda: report)),
    )
    qtbot.addWidget(window)
    assert window.page_count == 8
    assert window.history_service is None
    assert not window._history_page.table.isEnabled()
    assert "database corrupt" in window._history_page.loading_label.text()
