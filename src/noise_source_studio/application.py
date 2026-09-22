"""Application lifecycle and top-level exception handling."""

from __future__ import annotations

import faulthandler
import logging
import sys
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QMessageBox

from noise_source_studio.common.exceptions import NoiseSourceStudioError
from noise_source_studio.common.paths import ApplicationPaths
from noise_source_studio.infrastructure.config import SettingsManager
from noise_source_studio.infrastructure.logging import configure_logging
from noise_source_studio.presentation.main_window import MainWindow
from noise_source_studio.version import APPLICATION_TITLE, __version__

LOGGER = logging.getLogger("noise_source_studio.application")
_FAULT_LOG_HANDLE: TextIO | None = None


def create_application(arguments: Sequence[str] | None = None) -> QApplication:
    """Create or return the process-wide Qt application."""
    existing = QApplication.instance()
    if existing is not None:
        return existing

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(list(arguments) if arguments is not None else sys.argv)
    app.setApplicationName("Noise Source Studio")
    app.setApplicationDisplayName(APPLICATION_TITLE)
    app.setApplicationVersion(__version__)
    app.setOrganizationName("Noise Source Studio")
    app.setStyle("Fusion")
    return app


def apply_stylesheet(app: QApplication) -> None:
    """Load the packaged application-wide stylesheet."""
    stylesheet_path = (
        Path(__file__).resolve().parent / "presentation" / "styles" / "application.qss"
    )
    app.setStyleSheet(stylesheet_path.read_text(encoding="utf-8"))


def install_exception_handler(app: QApplication, log_file: Path | None = None) -> None:
    """Install process/thread handlers plus an optional native-fault log."""

    def handle_exception(
        exception_type: type[BaseException],
        exception: BaseException,
        traceback: TracebackType | None,
    ) -> None:
        if issubclass(exception_type, KeyboardInterrupt):
            sys.__excepthook__(exception_type, exception, traceback)
            return
        LOGGER.critical(
            "Unhandled application exception",
            exc_info=(exception_type, exception, traceback),
        )
        message = (
            str(exception)
            if isinstance(exception, NoiseSourceStudioError)
            else "应用遇到未预期错误。详细信息已写入系统日志。"
        )
        QMessageBox.critical(None, "Noise Source Studio", message)

    def handle_thread_exception(arguments: Any) -> None:
        LOGGER.critical(
            "Unhandled thread exception | thread=%s",
            getattr(arguments.thread, "name", "unknown"),
            exc_info=(arguments.exc_type, arguments.exc_value, arguments.exc_traceback),
        )

    sys.excepthook = handle_exception
    threading.excepthook = handle_thread_exception
    marker = _enable_fault_diagnostics(log_file) if log_file is not None else None

    def record_normal_shutdown() -> None:
        global _FAULT_LOG_HANDLE
        LOGGER.info("Application shutdown | normal=true")
        if marker is not None:
            try:
                marker.write_text(
                    f"normal {datetime.now(UTC).isoformat()}\n",
                    encoding="utf-8",
                )
            except OSError:
                LOGGER.exception("Could not write normal-exit marker | path=%s", marker)
        if _FAULT_LOG_HANDLE is not None:
            faulthandler.disable()
            _FAULT_LOG_HANDLE.close()
            _FAULT_LOG_HANDLE = None

    app.aboutToQuit.connect(record_normal_shutdown)


def _enable_fault_diagnostics(log_file: Path) -> Path:
    global _FAULT_LOG_HANDLE
    marker = log_file.with_name("application-exit.status")
    fault_log = log_file.with_name("noise-source-studio-fault.log")
    try:
        if marker.is_file():
            LOGGER.info(
                "Previous application exit marker | %s",
                marker.read_text(encoding="utf-8").strip(),
            )
        marker.write_text(
            f"running {datetime.now(UTC).isoformat()}\n",
            encoding="utf-8",
        )
        if _FAULT_LOG_HANDLE is not None:
            faulthandler.disable()
            _FAULT_LOG_HANDLE.close()
        _FAULT_LOG_HANDLE = fault_log.open("a", encoding="utf-8", buffering=1)
        faulthandler.enable(file=_FAULT_LOG_HANDLE, all_threads=True)
        LOGGER.info("Fault diagnostics enabled | file=%s", fault_log)
    except (OSError, RuntimeError):
        LOGGER.exception("Could not enable fault diagnostics | file=%s", fault_log)
        if _FAULT_LOG_HANDLE is not None:
            _FAULT_LOG_HANDLE.close()
            _FAULT_LOG_HANDLE = None
    return marker


def main(arguments: Sequence[str] | None = None) -> int:
    """Initialize services, create the main window and run the Qt event loop."""
    app = create_application(arguments)
    paths = ApplicationPaths.from_platformdirs()
    settings_manager = SettingsManager(paths)

    try:
        settings = settings_manager.load()
        log_file = configure_logging(settings.log_directory, settings.log_level)
        apply_stylesheet(app)
    except (NoiseSourceStudioError, OSError) as exc:
        logging.exception("Application initialization failed")
        QMessageBox.critical(None, "启动失败", f"应用无法完成初始化：{exc}")
        return 1

    install_exception_handler(app, log_file)
    LOGGER.info("Application startup | version=%s", __version__)
    LOGGER.info("Configuration file | %s", paths.config_file)
    window = MainWindow(settings, settings_manager, log_file)
    window.show()
    return app.exec()
