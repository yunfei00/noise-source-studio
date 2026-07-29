"""Application lifecycle and top-level exception handling."""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

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


def install_exception_handler(app: QApplication) -> None:
    """Log uncaught exceptions and show a concise user-facing error."""

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

    sys.excepthook = handle_exception
    app.aboutToQuit.connect(lambda: LOGGER.info("Application shutdown"))


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

    install_exception_handler(app)
    LOGGER.info("Application startup | version=%s", __version__)
    LOGGER.info("Configuration file | %s", paths.config_file)
    window = MainWindow(settings, settings_manager, log_file)
    window.show()
    return app.exec()
