"""Reusable widgets that keep page layouts and styling consistent."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

PAGE_CONTENT_MARGINS = (24, 20, 24, 24)
PAGE_CONTENT_SPACING = 16


class PageHeader(QWidget):
    """Standard page title and supporting text."""

    def __init__(self, title: str, subtitle: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("pageSubtitle")
        subtitle_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)


class SectionCard(QFrame):
    """A restrained, bordered content section with an optional heading."""

    def __init__(
        self,
        title: str = "",
        description: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("sectionCard")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.content_layout = QVBoxLayout(self)
        self.content_layout.setContentsMargins(20, 16, 20, 18)
        self.content_layout.setSpacing(12)

        if title:
            title_label = QLabel(title)
            title_label.setObjectName("sectionTitle")
            self.content_layout.addWidget(title_label)
        if description:
            description_label = QLabel(description)
            description_label.setObjectName("sectionDescription")
            description_label.setWordWrap(True)
            self.content_layout.addWidget(description_label)


class EmptyState(QWidget):
    """Consistent empty-state message for unavailable business data."""

    def __init__(
        self,
        title: str,
        description: str,
        max_text_width: int = 520,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        marker = QLabel("—")
        marker.setObjectName("emptyMarker")
        marker.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label = QLabel(title)
        title_label.setObjectName("emptyTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description_label = QLabel(description)
        description_label.setObjectName("emptyDescription")
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description_label.setWordWrap(True)
        description_label.setMinimumWidth(min(340, max_text_width))
        description_label.setMaximumWidth(max_text_width)
        description_label.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )

        self.title_label = title_label
        self.description_label = description_label

        layout.addWidget(marker)
        layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(description_label, alignment=Qt.AlignmentFlag.AlignHCenter)


class MetricCard(QFrame):
    """Compact label/value card for status and validation metrics."""

    def __init__(
        self,
        label: str,
        value: str = "—",
        note: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        label_widget = QLabel(label)
        label_widget.setObjectName("metricLabel")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        self.note_label = QLabel(note)
        self.note_label.setObjectName("metricNote")
        self.note_label.setWordWrap(True)
        layout.addWidget(label_widget)
        layout.addWidget(self.value_label)
        if note:
            layout.addWidget(self.note_label)

    def set_status(self, value: str, note: str = "") -> None:
        """Update a dashboard metric without rebuilding the frozen card layout."""
        self.value_label.setText(value)
        self.note_label.setText(note)
        self.note_label.setVisible(bool(note))


def create_table(headers: Iterable[str], minimum_height: int = 220) -> QTableWidget:
    """Create a consistently configured, read-only data table."""
    header_list = list(headers)
    table = QTableWidget(0, len(header_list))
    table.setHorizontalHeaderLabels(header_list)
    table.setMinimumHeight(minimum_height)
    table.setAlternatingRowColors(True)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    return table


def horizontal_row(*widgets: QWidget, spacing: int = 10) -> QWidget:
    """Arrange widgets in a zero-margin horizontal row."""
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for widget in widgets:
        layout.addWidget(widget)
    return container
