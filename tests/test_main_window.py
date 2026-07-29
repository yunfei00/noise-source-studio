"""Main window and navigation tests."""

from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from noise_source_studio.common.paths import ApplicationPaths
from noise_source_studio.infrastructure.config import SettingsManager
from noise_source_studio.infrastructure.logging import configure_logging
from noise_source_studio.presentation.main_window import MainWindow


def _create_window(application_paths: ApplicationPaths) -> MainWindow:
    manager = SettingsManager(application_paths)
    settings = manager.load()
    log_file = configure_logging(settings.log_directory)
    return MainWindow(settings, manager, log_file)


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
