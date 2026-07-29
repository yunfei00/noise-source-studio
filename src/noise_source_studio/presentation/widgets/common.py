"""Reusable widgets that keep page layouts and styling consistent."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)


class PageHeader(QWidget):
    """Standard page title and supporting text."""

    def __init__(self, title: str, subtitle: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

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
        self.content_layout.setContentsMargins(20, 18, 20, 20)
        self.content_layout.setSpacing(14)

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
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 28, 24, 28)
        layout.setSpacing(7)
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

        layout.addWidget(marker)
        layout.addWidget(title_label)
        layout.addWidget(description_label)


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
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(5)

        label_widget = QLabel(label)
        label_widget.setObjectName("metricLabel")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        layout.addWidget(label_widget)
        layout.addWidget(self.value_label)
        if note:
            note_widget = QLabel(note)
            note_widget.setObjectName("metricNote")
            layout.addWidget(note_widget)


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
    table.horizontalHeader().setStretchLastSection(True)
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
