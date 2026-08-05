"""Application construction tests."""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

import noise_source_studio.application as application_module
from noise_source_studio.application import create_application, install_exception_handler


def test_application_can_be_created(qapp: QApplication) -> None:
    application = create_application([])

    assert application is qapp
    assert application.applicationName() in {"pytest-qt-qapp", "Noise Source Studio"}


def test_exception_handlers_fault_log_and_normal_exit_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callbacks = []
    fake_app = SimpleNamespace(
        aboutToQuit=SimpleNamespace(connect=callbacks.append),
    )
    enabled = []
    disabled = []
    monkeypatch.setattr(
        application_module.faulthandler,
        "enable",
        lambda **kwargs: enabled.append(kwargs),
    )
    monkeypatch.setattr(application_module.faulthandler, "disable", lambda: disabled.append(True))
    original_system_hook = sys.excepthook
    original_thread_hook = threading.excepthook
    try:
        log_file = tmp_path / "noise-source-studio.log"
        install_exception_handler(fake_app, log_file)  # type: ignore[arg-type]
        marker = tmp_path / "application-exit.status"
        assert marker.read_text(encoding="utf-8").startswith("running ")
        assert enabled and enabled[0]["all_threads"] is True
        callbacks[0]()
        assert marker.read_text(encoding="utf-8").startswith("normal ")
        assert disabled == [True]
    finally:
        sys.excepthook = original_system_hook
        threading.excepthook = original_thread_hook
        handle = application_module._FAULT_LOG_HANDLE
        if handle is not None:
            handle.close()
            application_module._FAULT_LOG_HANDLE = None
