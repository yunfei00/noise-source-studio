"""Main window, status and navigation tests."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from noise_source_studio.common.paths import ApplicationPaths
from noise_source_studio.infrastructure.config import SettingsManager
from noise_source_studio.infrastructure.logging import configure_logging
from noise_source_studio.presentation.main_window import MainWindow
from noise_source_studio.presentation.pages.dashboard_page import DashboardPage
from noise_source_studio.version import APPLICATION_TITLE


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
