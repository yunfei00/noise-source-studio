"""Virtualized model and fast proxy filters for batch results."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt

from noise_source_studio.domain.batch import (
    BatchFileItem,
    BatchItemStatus,
    BatchPredictionTask,
)
from noise_source_studio.domain.confidence import (
    candidate_probability_margin,
    highest_combination_probability,
    is_low_confidence,
    result_confidence,
)
from noise_source_studio.services.result_adapter import primary_probability_summary

TASK_STATUS_TEXT = {
    BatchItemStatus.PENDING: "等待",
    BatchItemStatus.VALIDATING: "校验中",
    BatchItemStatus.RUNNING: "推理中",
    BatchItemStatus.SUCCESS: "成功",
    BatchItemStatus.FAILED: "失败",
    BatchItemStatus.SKIPPED: "跳过",
    BatchItemStatus.STOPPED: "已停止",
}


class BatchTaskTableModel(QAbstractTableModel):
    """Virtualized task table with an item-id index for incremental updates."""

    ItemRole = Qt.ItemDataRole.UserRole + 1
    ItemIdRole = Qt.ItemDataRole.UserRole + 2
    SortRole = Qt.ItemDataRole.UserRole + 3
    HEADERS = (
        "序号",
        "文件名",
        "目录",
        "状态",
        "预测组合",
        "结果",
        "主要概率",
        "耗时",
        "错误信息",
    )

    def __init__(self) -> None:
        super().__init__()
        self.task: BatchPredictionTask | None = None
        self.items: list[BatchFileItem] = []
        self.row_by_item_id: dict[str, int] = {}
        self.search_texts: list[str] = []

    def set_task(self, task: BatchPredictionTask | None) -> None:
        self.beginResetModel()
        self.task = task
        self.items = list(task.items) if task is not None else []
        self.search_texts = [f"{item.file_name} {item.file_path}".casefold() for item in self.items]
        self._reindex_items()
        self.endResetModel()

    def rebuild(self) -> None:
        self.set_task(self.task)

    def insert_item(self, item: BatchFileItem) -> None:
        if item.item_id in self.row_by_item_id:
            self.notify_items_changed({item.item_id})
            return
        row = len(self.items)
        self.beginInsertRows(QModelIndex(), row, row)
        self.items.append(item)
        self.search_texts.append(f"{item.file_name} {item.file_path}".casefold())
        self.row_by_item_id[item.item_id] = row
        self.endInsertRows()

    def remove_item(self, item_id: str) -> None:
        row = self.row_by_item_id.get(item_id)
        if row is None:
            return
        self.beginRemoveRows(QModelIndex(), row, row)
        self.items.pop(row)
        self.search_texts.pop(row)
        self._reindex_items()
        self.endRemoveRows()

    def notify_items_changed(self, item_ids: set[str]) -> None:
        rows = sorted(
            row
            for item_id in item_ids
            if (row := self.row_by_item_id.get(item_id)) is not None
        )
        if not rows:
            return
        first = previous = rows[0]
        for row in (*rows[1:], -1):
            if row == previous + 1:
                previous = row
                continue
            self.dataChanged.emit(
                self.index(first, 0),
                self.index(previous, self.columnCount() - 1),
            )
            first = previous = row

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(self.items)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not 0 <= index.row() < len(self.items):
            return None
        item = self.items[index.row()]
        if role == self.ItemRole:
            return item
        if role == self.ItemIdRole:
            return item.item_id
        if role == Qt.ItemDataRole.ToolTipRole:
            return (
                str(item.file_path)
                if index.column() in {1, 2}
                else self._display(item, index.column())
            )
        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() in {0, 3, 7}:
            return int(Qt.AlignmentFlag.AlignCenter)
        if role == self.SortRole:
            return self._sort_value(item, index.column())
        if role == Qt.ItemDataRole.DisplayRole:
            return self._display(item, index.column())
        if role == Qt.ItemDataRole.ForegroundRole and item.status == BatchItemStatus.FAILED:
            return Qt.GlobalColor.darkRed
        return None

    def item_at(self, row: int) -> BatchFileItem | None:
        return self.items[row] if 0 <= row < len(self.items) else None

    def search_text_at(self, row: int) -> str:
        return self.search_texts[row] if 0 <= row < len(self.search_texts) else ""

    def _reindex_items(self) -> None:
        self.row_by_item_id = {item.item_id: row for row, item in enumerate(self.items)}

    @staticmethod
    def _display(item: BatchFileItem, column: int) -> Any:
        values = (
            item.sequence,
            item.file_name,
            str(item.file_path.parent),
            TASK_STATUS_TEXT[item.status],
            item.predicted_combination or "—",
            ", ".join(item.predicted_sources) or "—",
            primary_probability_summary(item.result or {}),
            f"{item.elapsed_ms:.1f} ms" if item.elapsed_ms is not None else "—",
            item.error_message or "—",
        )
        return values[column]

    @staticmethod
    def _sort_value(item: BatchFileItem, column: int) -> Any:
        values = (
            item.sequence,
            item.file_name.casefold(),
            str(item.file_path.parent).casefold(),
            item.status.value,
            item.predicted_combination,
            " ".join(item.predicted_sources).casefold(),
            result_confidence(item.result or {}) or -1.0,
            item.elapsed_ms or -1.0,
            item.error_message.casefold(),
        )
        return values[column]


class BatchTaskFilterProxyModel(QSortFilterProxyModel):
    """Fast task keyword/status filtering over the virtualized table."""

    def __init__(self) -> None:
        super().__init__()
        self.keyword = ""
        self.status: BatchItemStatus | None = None
        self.setSortRole(BatchTaskTableModel.SortRole)
        self.setDynamicSortFilter(True)

    def set_filters(self, keyword: str, status: BatchItemStatus | None) -> None:
        self.beginFilterChange()
        self.keyword = keyword.strip().casefold()
        self.status = status
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    def filterAcceptsRow(  # noqa: N802
        self,
        source_row: int,
        source_parent: QModelIndex,
    ) -> bool:
        model = self.sourceModel()
        if not isinstance(model, BatchTaskTableModel):
            return True
        item = model.item_at(source_row)
        if item is None:
            return False
        if self.keyword and self.keyword not in model.search_text_at(source_row):
            return False
        return self.status is None or item.status == self.status


class BatchResultTableModel(QAbstractTableModel):
    """Read-only lightweight view over authoritative task items."""

    ItemRole = Qt.ItemDataRole.UserRole + 1
    SortRole = Qt.ItemDataRole.UserRole + 2
    HEADERS = (
        "序号",
        "文件名",
        "所在目录",
        "状态",
        "预测组合",
        "识别噪声源",
        "最高组合概率",
        "候选概率差",
        "推理耗时",
        "错误摘要",
    )

    def __init__(self) -> None:
        super().__init__()
        self.task: BatchPredictionTask | None = None
        self.items: list[BatchFileItem] = []
        self._row_by_item_id: dict[str, int] = {}
        self.search_texts: list[str] = []

    def set_task(self, task: BatchPredictionTask | None) -> None:
        self.beginResetModel()
        self.task = task
        self.items = list(task.items) if task is not None else []
        self.search_texts = [f"{item.file_name} {item.file_path}".casefold() for item in self.items]
        self._reindex_items()
        self.endResetModel()

    def refresh(self) -> None:
        if self.task is None:
            return
        self.beginResetModel()
        self.items = list(self.task.items)
        self.search_texts = [f"{item.file_name} {item.file_path}".casefold() for item in self.items]
        self._reindex_items()
        self.endResetModel()

    def notify_items_changed(self, item_ids: set[str]) -> None:
        """Notify views about changed task objects without resetting all rows."""
        rows = sorted(
            row
            for item_id in item_ids
            if (row := self._row_by_item_id.get(item_id)) is not None
        )
        if not rows:
            return
        first = previous = rows[0]
        for row in (*rows[1:], -1):
            if row == previous + 1:
                previous = row
                continue
            self.dataChanged.emit(
                self.index(first, 0),
                self.index(previous, self.columnCount() - 1),
            )
            first = previous = row

    def _reindex_items(self) -> None:
        self._row_by_item_id = {item.item_id: row for row, item in enumerate(self.items)}

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(self.items)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not 0 <= index.row() < len(self.items):
            return None
        item = self.items[index.row()]
        payload = item.result or {}
        if role == self.ItemRole:
            return item
        if role == Qt.ItemDataRole.ToolTipRole:
            return (
                str(item.file_path)
                if index.column() in {1, 2}
                else self._display(item, index.column())
            )
        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() in {0, 3, 6, 7, 8}:
            return int(Qt.AlignmentFlag.AlignCenter)
        if role == self.SortRole:
            return self._sort_value(item, index.column())
        if role == Qt.ItemDataRole.DisplayRole:
            return self._display(item, index.column())
        if role == Qt.ItemDataRole.ForegroundRole and item.status == BatchItemStatus.FAILED:
            return Qt.GlobalColor.darkRed
        if role == Qt.ItemDataRole.BackgroundRole and is_low_confidence(payload):
            return Qt.GlobalColor.lightGray
        return None

    @staticmethod
    def _display(item: BatchFileItem, column: int) -> Any:
        payload = item.result or {}
        top = highest_combination_probability(payload)
        margin = candidate_probability_margin(payload)
        values = (
            item.sequence,
            item.file_name,
            str(item.file_path.parent),
            item.status.value,
            item.predicted_combination or "—",
            ", ".join(item.predicted_sources) or "—",
            f"{top * 100:.2f}%" if top is not None else "—",
            f"{margin * 100:.2f}%" if margin is not None else "—",
            f"{item.elapsed_ms:.1f} ms" if item.elapsed_ms is not None else "—",
            item.error_message or "—",
        )
        return values[column]

    @staticmethod
    def _sort_value(item: BatchFileItem, column: int) -> Any:
        payload = item.result or {}
        values = (
            item.sequence,
            item.file_name.casefold(),
            str(item.file_path.parent).casefold(),
            item.status.value,
            item.predicted_combination,
            " ".join(item.predicted_sources).casefold(),
            highest_combination_probability(payload) or -1.0,
            candidate_probability_margin(payload) or -1.0,
            item.elapsed_ms or -1.0,
            item.error_message.casefold(),
        )
        return values[column]

    def item_at(self, row: int) -> BatchFileItem | None:
        return self.items[row] if 0 <= row < len(self.items) else None

    def search_text_at(self, row: int) -> str:
        return self.search_texts[row] if 0 <= row < len(self.search_texts) else ""


class BatchResultFilterProxyModel(QSortFilterProxyModel):
    """Composable O(n) filters suitable for ten thousand lightweight rows."""

    def __init__(self) -> None:
        super().__init__()
        self.keyword = ""
        self.status: BatchItemStatus | None = None
        self.combination = ""
        self.sources: set[str] = set()
        self.confidence_min = 0.0
        self.confidence_max = 1.0
        self.low_confidence_only = False
        self.errors_only = False
        self.error_type = ""
        self.setSortRole(BatchResultTableModel.SortRole)
        self.setDynamicSortFilter(True)

    def set_filters(
        self,
        *,
        keyword: str | None = None,
        status: BatchItemStatus | None | object = ...,
        combination: str | None = None,
        sources: set[str] | None = None,
        confidence_min: float | None = None,
        confidence_max: float | None = None,
        low_confidence_only: bool | None = None,
        errors_only: bool | None = None,
        error_type: str | None = None,
    ) -> None:
        self.beginFilterChange()
        if keyword is not None:
            self.keyword = keyword.strip().casefold()
        if status is not ...:
            self.status = status  # type: ignore[assignment]
        if combination is not None:
            self.combination = combination
        if sources is not None:
            self.sources = set(sources)
        if confidence_min is not None:
            self.confidence_min = confidence_min
        if confidence_max is not None:
            self.confidence_max = confidence_max
        if low_confidence_only is not None:
            self.low_confidence_only = low_confidence_only
        if errors_only is not None:
            self.errors_only = errors_only
        if error_type is not None:
            self.error_type = error_type
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    def filterAcceptsRow(  # noqa: N802
        self, source_row: int, source_parent: QModelIndex
    ) -> bool:
        model = self.sourceModel()
        if not isinstance(model, BatchResultTableModel):
            return True
        item = model.item_at(source_row)
        if item is None:
            return False
        payload = item.result or {}
        if self.keyword and self.keyword not in model.search_text_at(source_row):
            return False
        if self.status is not None and item.status != self.status:
            return False
        if self.combination and item.predicted_combination != self.combination:
            return False
        if self.sources and not self.sources.intersection(item.predicted_sources):
            return False
        score = result_confidence(payload)
        if score is not None and not self.confidence_min <= score <= self.confidence_max:
            return False
        if self.low_confidence_only and not is_low_confidence(payload):
            return False
        if self.errors_only and item.status != BatchItemStatus.FAILED:
            return False
        return not self.error_type or item.error_type == self.error_type

    def filtered_items(self) -> list[BatchFileItem]:
        model = self.sourceModel()
        if not isinstance(model, BatchResultTableModel):
            return []
        return [
            item
            for row in range(self.rowCount())
            if (item := model.item_at(self.mapToSource(self.index(row, 0)).row())) is not None
        ]


__all__ = [
    "BatchResultFilterProxyModel",
    "BatchResultTableModel",
    "BatchTaskFilterProxyModel",
    "BatchTaskTableModel",
]
