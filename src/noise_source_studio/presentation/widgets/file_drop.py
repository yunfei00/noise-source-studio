"""Reusable file drop target."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class FileDropZone(QWidget):
    """Accept one local file by drag-and-drop or button selection."""

    browse_requested = Signal()
    file_dropped = Signal(str)

    def __init__(
        self,
        title: str = "拖放文件到此处",
        description: str = "或从本机选择一个文件",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("fileDropZone")
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 28, 24, 28)
        layout.setSpacing(8)

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
        """Accept drags containing at least one local file."""
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        """Emit the first dropped local file path."""
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.file_dropped.emit(url.toLocalFile())
                event.acceptProposedAction()
                return
