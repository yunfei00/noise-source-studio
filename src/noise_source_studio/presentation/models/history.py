"""Qt table model for paginated history records."""

from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QStyle

from noise_source_studio.domain.history import HistoryStatus, TaskHistoryRecord, TaskType

HISTORY_COLUMNS = (
    ("created_at", "创建时间"),
    ("task_type", "任务类型"),
    ("task_name", "任务名称"),
    ("status", "状态"),
    ("model_name", "模型"),
    ("device", "设备"),
    ("total_count", "总数"),
    ("success_count", "成功"),
    ("failed_count", "失败"),
    ("duration_ms", "耗时"),
    ("primary_summary", "摘要"),
    ("integrity_status", "完整性"),
)
TYPE_TEXT = {
    TaskType.SINGLE: "单文件预测",
    TaskType.BATCH: "批量预测",
    TaskType.VALIDATION: "模型验证",
}
STATUS_TEXT = {
    HistoryStatus.CREATED: "已创建",
    HistoryStatus.RUNNING: "运行中",
    HistoryStatus.COMPLETED: "已完成",
    HistoryStatus.COMPLETED_WITH_ERRORS: "完成（有错误）",
    HistoryStatus.FAILED: "失败",
    HistoryStatus.STOPPED: "已停止",
    HistoryStatus.INTERRUPTED: "已中断",
}


class HistoryTableModel(QAbstractTableModel):
    """Read-only display model; filtering and pagination stay in SQLite."""

    sort_requested = Signal(str, bool)

    def __init__(self) -> None:
        super().__init__()
        self.records: tuple[TaskHistoryRecord, ...] = ()

    def set_records(self, records: tuple[TaskHistoryRecord, ...]) -> None:
        self.beginResetModel()
        self.records = records
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(self.records)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(HISTORY_COLUMNS)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> object:
        if not index.isValid() or not 0 <= index.row() < len(self.records):
            return None
        record = self.records[index.row()]
        key = HISTORY_COLUMNS[index.column()][0]
        if role == Qt.ItemDataRole.DisplayRole:
            value = getattr(record, key)
            if key == "task_type":
                return TYPE_TEXT[record.task_type]
            if key == "status":
                return STATUS_TEXT[record.status]
            if key in {"created_at", "finished_at"}:
                return str(value).replace("T", " ")[:19] if value else "—"
            if key == "model_name":
                return record.model_identifier or "—"
            if key == "device":
                return value or "—"
            if key == "duration_ms":
                return f"{record.duration_ms / 1000.0:.3f} s"
            if key == "integrity_status":
                return {
                    "unknown": "未校验",
                    "ok": "完整",
                    "missing": "缺失",
                    "partial": "部分缺失",
                    "corrupt": "损坏",
                    "external": "外部目录",
                }.get(record.integrity_status.value, record.integrity_status.value)
            return value
        if role == Qt.ItemDataRole.DecorationRole and key == "status":
            return self._status_icon(record.status)
        if role == Qt.ItemDataRole.ToolTipRole:
            if key in {"task_name", "primary_summary"}:
                return str(getattr(record, key))
        if role == Qt.ItemDataRole.TextAlignmentRole and key in {
            "total_count",
            "success_count",
            "failed_count",
        }:
            return Qt.AlignmentFlag.AlignCenter
        return None

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> object:  # noqa: N802
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return HISTORY_COLUMNS[section][1]
        return super().headerData(section, orientation, role)

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        if 0 <= column < len(HISTORY_COLUMNS):
            self.sort_requested.emit(
                HISTORY_COLUMNS[column][0],
                order == Qt.SortOrder.DescendingOrder,
            )

    def record_at(self, row: int) -> TaskHistoryRecord | None:
        return self.records[row] if 0 <= row < len(self.records) else None

    @staticmethod
    def _status_icon(status: HistoryStatus) -> QIcon:
        style = QApplication.style()
        icons = {
            HistoryStatus.CREATED: QStyle.StandardPixmap.SP_FileIcon,
            HistoryStatus.RUNNING: QStyle.StandardPixmap.SP_BrowserReload,
            HistoryStatus.COMPLETED: QStyle.StandardPixmap.SP_DialogApplyButton,
            HistoryStatus.COMPLETED_WITH_ERRORS: QStyle.StandardPixmap.SP_MessageBoxWarning,
            HistoryStatus.FAILED: QStyle.StandardPixmap.SP_MessageBoxCritical,
            HistoryStatus.STOPPED: QStyle.StandardPixmap.SP_BrowserStop,
            HistoryStatus.INTERRUPTED: QStyle.StandardPixmap.SP_MessageBoxWarning,
        }
        return style.standardIcon(icons[status]) if style is not None else QIcon()


__all__ = ["HISTORY_COLUMNS", "HistoryTableModel", "STATUS_TEXT", "TYPE_TEXT"]
