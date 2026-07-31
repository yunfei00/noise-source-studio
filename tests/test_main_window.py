"""Main window, status and navigation tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from noise_source_studio.common.exceptions import ModelActivationError
from noise_source_studio.common.paths import ApplicationPaths
from noise_source_studio.domain.batch import BatchStatus
from noise_source_studio.domain.device import (
    DeviceInfo,
    DeviceProbeReport,
    DeviceResolution,
)
from noise_source_studio.domain.models import LoadedModel, ModelRecord
from noise_source_studio.domain.validation import ValidationStatus
from noise_source_studio.infrastructure.config import SettingsManager
from noise_source_studio.infrastructure.logging import configure_logging
from noise_source_studio.presentation.main_window import MainWindow
from noise_source_studio.presentation.pages.dashboard_page import DashboardPage
from noise_source_studio.services import DeviceService
from noise_source_studio.version import APPLICATION_TITLE


def _create_window(application_paths: ApplicationPaths) -> MainWindow:
    manager = SettingsManager(application_paths)
    settings = manager.load()
    log_file = configure_logging(settings.log_directory)
    report = _device_report()
    device_service = DeviceService(
        SimpleNamespace(probe_devices=lambda: report)
    )
    return MainWindow(
        settings,
        manager,
        log_file,
        device_service=device_service,
    )


def _device_report(*, cuda_available: bool = False) -> DeviceProbeReport:
    devices = [DeviceInfo("cpu", "CPU", "cpu", True, True)]
    if cuda_available:
        devices.append(
            DeviceInfo(
                "cuda:0",
                "CUDA:0 · Fake GPU",
                "cuda",
                True,
                True,
                total_memory=24 * 1024**3,
                index=0,
            )
        )
    return DeviceProbeReport(
        devices=tuple(devices),
        torch_cuda_available=cuda_available,
        cuda_runtime_version="13.0" if cuda_available else "",
        torch_version="2.9.0-test",
        cuda_device_count=1 if cuda_available else 0,
        error_message="" if cuda_available else "CUDA 不可用",
    )


def _loaded_model(device: str = "cpu") -> LoadedModel:
    record = ModelRecord(
        "noise-source-current",
        "0.1.0",
        Path("model"),
        {"prediction_mode": "multilabel"},
        "now",
        True,
        "valid",
    )
    return LoadedModel(
        record,
        "1.0.0",
        device,
        "multilabel",
        ("fan", "motor"),
        {"device": device},
    )


def test_main_window_can_be_created(
    qapp: QApplication,
    qtbot: QtBot,
    application_paths: ApplicationPaths,
) -> None:
    window = _create_window(application_paths)
    qtbot.addWidget(window)

    assert window.minimumWidth() == 1180
    assert window.minimumHeight() == 720
    assert window.page_count == 8
    assert window.windowTitle() == APPLICATION_TITLE


def test_all_eight_navigation_pages_can_be_switched(
    qapp: QApplication,
    qtbot: QtBot,
    application_paths: ApplicationPaths,
) -> None:
    window = _create_window(application_paths)
    qtbot.addWidget(window)

    for index in range(8):
        window.navigation.select_page(index)
        assert window.page_stack.currentIndex() == index
        assert window.navigation.list_widget.currentRow() == index
        assert not window.navigation.list_widget.item(index).icon().isNull()


def test_dashboard_quick_actions_navigate_to_expected_pages(
    qapp: QApplication,
    qtbot: QtBot,
    application_paths: ApplicationPaths,
) -> None:
    window = _create_window(application_paths)
    qtbot.addWidget(window)
    window.show()
    dashboard = window.pages[0]
    assert isinstance(dashboard, DashboardPage)

    for button, expected_index in zip(
        dashboard.quick_action_buttons,
        (1, 2, 3),
        strict=True,
    ):
        window.navigation.select_page(0)
        qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
        assert window.page_stack.currentIndex() == expected_index


def test_header_statuses_can_be_updated_through_shared_method(
    qapp: QApplication,
    qtbot: QtBot,
    application_paths: ApplicationPaths,
) -> None:
    window = _create_window(application_paths)
    qtbot.addWidget(window)

    window.update_header_status("device", "CPU", "active")

    assert window.header_statuses["device"].value_label.text() == "CPU"
    assert window.header_statuses["device"].value_label.property("state") == "active"


def test_unconfigured_model_status_is_explicit(
    qapp: QApplication,
    qtbot: QtBot,
    application_paths: ApplicationPaths,
) -> None:
    window = _create_window(application_paths)
    qtbot.addWidget(window)
    dashboard = window.pages[0]
    assert isinstance(dashboard, DashboardPage)

    assert window.header_statuses["model"].value_label.text() == "未配置"
    assert window.header_statuses["device"].value_label.text() == "待检测"
    assert window.header_statuses["mode"].value_label.text() == "本地"
    assert dashboard.status_cards["model"].value_label.text() == "未配置"
    assert dashboard.status_cards["device"].value_label.text() == "待检测"
    assert dashboard.status_cards["application"].value_label.text() == "正常"


def test_device_header_opens_system_settings(
    qapp: QApplication,
    qtbot: QtBot,
    application_paths: ApplicationPaths,
) -> None:
    window = _create_window(application_paths)
    qtbot.addWidget(window)

    qtbot.mouseClick(
        window.header_statuses["device"],
        Qt.MouseButton.LeftButton,
    )

    assert window.page_stack.currentIndex() == 7


def test_configured_policy_and_actual_session_device_are_separate(
    qapp: QApplication,
    qtbot: QtBot,
    application_paths: ApplicationPaths,
) -> None:
    window = _create_window(application_paths)
    qtbot.addWidget(window)
    report = _device_report()
    window.device_report = report
    window.device_resolution = DeviceResolution("auto", "cpu", report, "CUDA 不可用")

    window._apply_loaded_model(_loaded_model("cpu"))

    assert window.settings.device_preference == "auto"
    assert window.header_statuses["device"].value_label.text() == "CPU"
    assert window._settings_page.device_combo.currentData() == "auto"
    assert window._settings_page.actual_device_label.text() == "CPU"


def test_cuda_actual_device_uses_probed_name(
    qapp: QApplication,
    qtbot: QtBot,
    application_paths: ApplicationPaths,
) -> None:
    window = _create_window(application_paths)
    qtbot.addWidget(window)
    report = _device_report(cuda_available=True)
    window.device_report = report

    window._apply_loaded_model(_loaded_model("cuda:0"))

    assert (
        window.header_statuses["device"].value_label.text()
        == "CUDA:0 · Fake GPU"
    )
    assert window._settings_page.actual_device_label.text() == "CUDA:0 · Fake GPU"


def test_device_switch_is_blocked_during_model_single_batch_and_validation_tasks(
    qapp: QApplication,
    qtbot: QtBot,
    application_paths: ApplicationPaths,
) -> None:
    window = _create_window(application_paths)
    qtbot.addWidget(window)

    window._model_loading = True
    assert "模型正在加载" in window._device_switch_block_reason()
    window._model_loading = False
    window._single_prediction_running = True
    assert "推理任务" in window._device_switch_block_reason()
    window._single_prediction_running = False
    window.batch_task.status = BatchStatus.PAUSED
    assert "批量任务" in window._device_switch_block_reason()
    window.batch_task.status = BatchStatus.CREATED
    window.validation_task = SimpleNamespace(  # type: ignore[assignment]
        status=ValidationStatus.PAUSED
    )
    assert "模型验证" in window._device_switch_block_reason()
    window.validation_task = None


def test_blocked_device_change_is_not_persisted(
    qapp: QApplication,
    qtbot: QtBot,
    application_paths: ApplicationPaths,
) -> None:
    window = _create_window(application_paths)
    qtbot.addWidget(window)
    window.batch_task.status = BatchStatus.RUNNING
    updated = window.settings.model_copy(update={"device_preference": "cpu"})

    window._save_settings(updated)

    assert window.settings.device_preference == "auto"
    assert window._settings_page.settings.device_preference == "auto"
    assert "批量任务" in window._settings_page.feedback_label.text()
    window.batch_task.status = BatchStatus.CREATED


def test_explicit_cuda_fallback_cancel_keeps_preference(
    qapp: QApplication,
    qtbot: QtBot,
    application_paths: ApplicationPaths,
    monkeypatch,
) -> None:
    window = _create_window(application_paths)
    qtbot.addWidget(window)
    cuda_settings = window.settings.model_copy(update={"device_preference": "cuda:0"})
    window.settings = cuda_settings
    window._settings_page.apply_settings(cuda_settings)
    monkeypatch.setattr(window, "_ask_cpu_fallback", lambda *_: "cancel")

    window._offer_explicit_cpu_fallback("model@1", "cuda:0", "probe failed")

    assert window.settings.device_preference == "cuda:0"


def test_explicit_cuda_fallback_confirmation_persists_cpu_and_retries(
    qapp: QApplication,
    qtbot: QtBot,
    application_paths: ApplicationPaths,
    monkeypatch,
) -> None:
    window = _create_window(application_paths)
    qtbot.addWidget(window)
    cuda_settings = window.settings.model_copy(update={"device_preference": "cuda:0"})
    window.settings = cuda_settings
    window._settings_page.apply_settings(cuda_settings)
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(window, "_ask_cpu_fallback", lambda *_: "cpu")
    monkeypatch.setattr(
        window,
        "_begin_model_load",
        lambda identifier, *, startup, device_preference=None: calls.append(
            (identifier, device_preference)
        ),
    )

    window._offer_explicit_cpu_fallback("model@1", "cuda:0", "probe failed")
    qtbot.waitUntil(lambda: bool(calls))

    assert window.settings.device_preference == "cpu"
    assert calls == [("model@1", "cpu")]


def test_cuda_failure_without_session_is_truthfully_displayed(
    qapp: QApplication,
    qtbot: QtBot,
    application_paths: ApplicationPaths,
) -> None:
    window = _create_window(application_paths)
    qtbot.addWidget(window)

    window._show_preserved_or_failed_session("cuda:0", "tensor failed")

    assert window.header_statuses["model"].value_label.text() == "加载失败"
    assert (
        window.header_statuses["device"].value_label.text()
        == "CUDA:0 不可用"
    )


def test_auto_model_load_failure_retries_cpu_without_prompt(
    qapp: QApplication,
    qtbot: QtBot,
    application_paths: ApplicationPaths,
) -> None:
    window = _create_window(application_paths)
    qtbot.addWidget(window)
    report = _device_report(cuda_available=True)
    window.device_service = DeviceService(
        SimpleNamespace(probe_devices=lambda: report)
    )
    calls: list[str] = []

    def activate_model(identifier: str, *, device: str) -> LoadedModel:
        del identifier
        calls.append(device)
        if device.startswith("cuda:"):
            raise ModelActivationError("CUDA model initialization failed")
        return _loaded_model(device)

    window.model_service = SimpleNamespace(activate_model=activate_model)

    loaded, resolution = window._load_model_with_policy(
        "model@1",
        "auto",
        allow_cpu_fallback=True,
    )

    assert calls == ["cuda:0", "cpu"]
    assert loaded.device == "cpu"
    assert resolution.resolved_device == "cpu"
    assert "CUDA 模型加载失败" in resolution.fallback_reason


def test_explicit_cuda_model_load_failure_does_not_retry_cpu(
    qapp: QApplication,
    qtbot: QtBot,
    application_paths: ApplicationPaths,
) -> None:
    window = _create_window(application_paths)
    qtbot.addWidget(window)
    report = _device_report(cuda_available=True)
    window.device_service = DeviceService(
        SimpleNamespace(probe_devices=lambda: report)
    )
    calls: list[str] = []

    def activate_model(identifier: str, *, device: str) -> LoadedModel:
        del identifier
        calls.append(device)
        raise ModelActivationError("CUDA model initialization failed")

    window.model_service = SimpleNamespace(activate_model=activate_model)

    with pytest.raises(ModelActivationError):
        window._load_model_with_policy(
            "model@1",
            "cuda:0",
            allow_cpu_fallback=True,
        )

    assert calls == ["cuda:0"]


def test_startup_restores_persisted_cpu_without_cuda_probe(
    qapp: QApplication,
    qtbot: QtBot,
    application_paths: ApplicationPaths,
) -> None:
    manager = SettingsManager(application_paths)
    settings = manager.load().model_copy(update={"device_preference": "cpu"})
    manager.save(settings)
    record = _loaded_model("cpu").record
    activation_devices: list[str] = []

    class StartupModelService:
        def active_model(self) -> ModelRecord:
            return record

        def list_models(self) -> tuple[ModelRecord, ...]:
            return (record,)

        def activate_model(self, identifier: str, *, device: str) -> LoadedModel:
            assert identifier == record.identifier
            activation_devices.append(device)
            return _loaded_model(device)

    class ForbiddenProbe:
        def probe_devices(self) -> DeviceProbeReport:
            raise AssertionError("CPU startup must not probe CUDA")

    window = MainWindow(
        settings,
        manager,
        configure_logging(settings.log_directory),
        model_service=StartupModelService(),  # type: ignore[arg-type]
        device_service=DeviceService(ForbiddenProbe()),
    )
    qtbot.addWidget(window)
    qtbot.waitUntil(lambda: window.loaded_model is not None)

    assert activation_devices == ["cpu"]
    assert window.header_statuses["device"].value_label.text() == "CPU"
