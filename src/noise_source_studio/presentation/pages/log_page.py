"""Application log viewer page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from noise_source_studio.presentation.widgets import PageHeader, SectionCard


class LogPage(QWidget):
    """Read-only view over the actual application log file."""

    def __init__(self, log_file: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.log_file = log_file
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 26, 28, 28)
        layout.setSpacing(20)
        layout.addWidget(
            PageHeader(
                "系统日志",
                "查看当前应用日志。完整异常堆栈仅保存在日志文件中。",
            )
        )

        controls = SectionCard("日志筛选")
        control_layout = QHBoxLayout()
        self.level_filter = QComboBox()
        self.level_filter.addItems(("全部级别", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
        self.keyword_filter = QLineEdit()
        self.keyword_filter.setPlaceholderText("搜索日志消息")
        refresh_button = QPushButton("刷新")
        refresh_button.setObjectName("primaryButton")
        refresh_button.clicked.connect(self.refresh)
        open_button = QPushButton("打开日志目录")
        open_button.clicked.connect(self.open_log_directory)
        clear_button = QPushButton("清理显示")
        clear_button.clicked.connect(self.clear_display)
        control_layout.addWidget(self.level_filter)
        control_layout.addWidget(self.keyword_filter, 1)
        control_layout.addWidget(refresh_button)
        control_layout.addWidget(open_button)
        control_layout.addWidget(clear_button)
        controls.content_layout.addLayout(control_layout)
        layout.addWidget(controls)

        viewer_card = SectionCard("应用日志")
        self.log_viewer = QPlainTextEdit()
        self.log_viewer.setObjectName("logViewer")
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        viewer_card.content_layout.addWidget(self.log_viewer)
        layout.addWidget(viewer_card, 1)

        self.level_filter.currentTextChanged.connect(self.refresh)
        self.keyword_filter.returnPressed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        """Reload the current log file and apply display filters."""
        try:
            lines = self.log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []

        level = self.level_filter.currentText()
        keyword = self.keyword_filter.text().strip().casefold()
        if level != "全部级别":
            lines = [line for line in lines if f"| {level}" in line]
        if keyword:
            lines = [line for line in lines if keyword in line.casefold()]
        self.log_viewer.setPlainText("\n".join(lines[-2000:]))
        scrollbar = self.log_viewer.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def open_log_directory(self) -> None:
        """Open the platform file browser at the log directory."""
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.log_file.parent)))

    def clear_display(self) -> None:
        """Clear only the viewer without deleting persisted logs."""
        self.log_viewer.clear()
