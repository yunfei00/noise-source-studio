"""Application construction tests."""

from PySide6.QtWidgets import QApplication

from noise_source_studio.application import create_application


def test_application_can_be_created(qapp: QApplication) -> None:
    application = create_application([])

    assert application is qapp
    assert application.applicationName() in {"pytest-qt-qapp", "Noise Source Studio"}
