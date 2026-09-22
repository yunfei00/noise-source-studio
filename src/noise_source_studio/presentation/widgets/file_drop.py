"""Reusable file drop target."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class FileDropZone(QWidget):
    """Accept one local file by drag-and-drop or button selection."""

    browse_requested = Signal()
    file_dropped = Signal(str)
    file_rejected = Signal(str)

    def __init__(
        self,
        title: str = "拖放文件到此处",
        description: str = "或从本机选择一个文件",
        accepted_extensions: Iterable[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.accepted_extensions = {
            extension.lower() if extension.startswith(".") else f".{extension.lower()}"
            for extension in (accepted_extensions or ())
        }
        self.setObjectName("fileDropZone")
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(7)

        title_label = QLabel(title)
        title_label.setObjectName("dropTitle")
        description_label = QLabel(description)
        description_label.setObjectName("dropDescription")
        self.browse_button = QPushButton("选择文件")
        self.browse_button.setObjectName("secondaryButton")
        self.browse_button.clicked.connect(self.browse_requested)

        layout.addStretch()
        layout.addWidget(title_label, alignment=layout.alignment())
        layout.addWidget(description_label, alignment=layout.alignment())
        layout.addWidget(self.browse_button)
        layout.addStretch()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        """Accept drags containing at least one supported local file."""
        if event.mimeData().hasUrls() and any(
            url.isLocalFile() and self._is_supported(url.toLocalFile())
            for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        """Emit the first dropped local file path."""
        for url in event.mimeData().urls():
            if url.isLocalFile():
                local_path = url.toLocalFile()
                if self._is_supported(local_path):
                    self.file_dropped.emit(local_path)
                    event.acceptProposedAction()
                else:
                    self.file_rejected.emit(local_path)
                return

    def _is_supported(self, path: str) -> bool:
        return not self.accepted_extensions or Path(path).suffix.lower() in self.accepted_extensions
