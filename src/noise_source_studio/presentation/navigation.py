"""Left-side application navigation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QFrame, QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget


@dataclass(frozen=True, slots=True)
class NavigationItem:
    """Navigation metadata shared by the sidebar and page stack."""

    key: str
    label: str


NAVIGATION_ITEMS = (
    NavigationItem("dashboard", "工作台"),
    NavigationItem("single_prediction", "单文件预测"),
    NavigationItem("batch_prediction", "批量预测"),
    NavigationItem("validation", "模型验证"),
    NavigationItem("model_management", "模型管理"),
    NavigationItem("history", "任务历史"),
    NavigationItem("logs", "系统日志"),
    NavigationItem("settings", "系统设置"),
)

ICON_DIRECTORY = Path(__file__).resolve().parent / "icons"


def navigation_icon(key: str) -> QIcon:
    """Return the scalable line icon for a navigation key."""
    return QIcon(str(ICON_DIRECTORY / f"{key}.svg"))


class NavigationSidebar(QFrame):
    """Fixed-width application navigation."""

    page_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("navigationSidebar")
        self.setFixedWidth(240)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 16, 14, 16)
        layout.setSpacing(8)

        label = QLabel("功能导航")
        label.setObjectName("navigationLabel")
        self.list_widget = QListWidget()
        self.list_widget.setObjectName("navigationList")
        self.list_widget.setIconSize(QSize(20, 20))
        self.list_widget.setSpacing(2)
        for item in NAVIGATION_ITEMS:
            list_item = QListWidgetItem(navigation_icon(item.key), item.label)
            list_item.setData(Qt.ItemDataRole.UserRole, item.key)
            list_item.setSizeHint(QSize(0, 44))
            self.list_widget.addItem(list_item)

        layout.addWidget(label)
        layout.addWidget(self.list_widget, 1)
        self.list_widget.currentRowChanged.connect(self.page_selected)
        self.list_widget.setCurrentRow(0)

    def select_page(self, index: int) -> None:
        """Select a page by stack index."""
        if 0 <= index < self.list_widget.count():
            self.list_widget.setCurrentRow(index)
