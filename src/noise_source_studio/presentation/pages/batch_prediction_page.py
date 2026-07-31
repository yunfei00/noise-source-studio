"""Batch prediction queue, controls, progress and result inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStyle,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from noise_source_studio.domain.batch import (
    RUNNING_BATCH_STATUSES,
    BatchFileItem,
    BatchItemStatus,
    BatchPredictionTask,
    BatchScanSummary,
    BatchStatus,
)
from noise_source_studio.presentation.widgets import (
    PAGE_CONTENT_MARGINS,
    PAGE_CONTENT_SPACING,
    PageHeader,
    SectionCard,
    create_table,
)
from noise_source_studio.services.result_adapter import primary_probability_summary

STATUS_TEXT = {
    BatchItemStatus.PENDING: "等待",
    BatchItemStatus.VALIDATING: "校验中",
    BatchItemStatus.RUNNING: "推理中",
    BatchItemStatus.SUCCESS: "成功",
    BatchItemStatus.FAILED: "失败",
    BatchItemStatus.SKIPPED: "跳过",
    BatchItemStatus.STOPPED: "已停止",
}


class BatchPredictionPage(QWidget):
    """Interactive batch workflow without owning inference execution."""

    paths_added = Signal(object, bool)
    remove_requested = Signal(object)
    clear_requested = Signal()
    deduplicate_requested = Signal()
    start_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    stop_requested = Signal()
    retry_requested = Signal(object)
    export_requested = Signal(object)
    logs_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.task: BatchPredictionTask | None = None
        self.model_available = False
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*PAGE_CONTENT_MARGINS)
        layout.setSpacing(PAGE_CONTENT_SPACING)
        layout.addWidget(
            PageHeader(
                "批量预测",
                "复用当前已加载模型，按队列顺序逐文件推理；单个文件失败不会中断其余任务。",
            )
        )

        controls = SectionCard("文件队列")
        first_row = QHBoxLayout()
        self.add_files_button = self._button(
            "添加文件", QStyle.StandardPixmap.SP_FileIcon, self._choose_files
        )
        self.add_folder_button = self._button(
            "添加文件夹", QStyle.StandardPixmap.SP_DirIcon, self._choose_folder
        )
        self.recursive_checkbox = QCheckBox("递归扫描子文件夹")
        self.remove_button = self._button("移除所选", callback=self._remove_selected)
        self.clear_button = self._button("清空", callback=self.clear_requested.emit)
        self.deduplicate_button = self._button("去重", callback=self.deduplicate_requested.emit)
        for widget in (
            self.add_files_button,
            self.add_folder_button,
            self.recursive_checkbox,
            self.remove_button,
            self.clear_button,
            self.deduplicate_button,
        ):
            first_row.addWidget(widget)
        first_row.addStretch()
        controls.content_layout.addLayout(first_row)

        second_row = QHBoxLayout()
        self.start_button = self._button(
            "开始批量预测",
            QStyle.StandardPixmap.SP_MediaPlay,
            self.start_requested.emit,
            primary=True,
        )
        self.pause_button = self._button(
            "暂停", QStyle.StandardPixmap.SP_MediaPause, self.pause_requested.emit
        )
        self.resume_button = self._button(
            "继续", QStyle.StandardPixmap.SP_MediaPlay, self.resume_requested.emit
        )
        self.stop_button = self._button(
            "停止", QStyle.StandardPixmap.SP_MediaStop, self.stop_requested.emit
        )
        self.retry_selected_button = self._button("重试所选失败项", callback=self._retry_selected)
        self.retry_all_button = self._button(
            "重试全部失败项", callback=lambda: self.retry_requested.emit(None)
        )
        self.export_button = self._button(
            "导出副本", QStyle.StandardPixmap.SP_DialogSaveButton, self._choose_export
        )
        for widget in (
            self.start_button,
            self.pause_button,
            self.resume_button,
            self.stop_button,
            self.retry_selected_button,
            self.retry_all_button,
            self.export_button,
        ):
            second_row.addWidget(widget)
        second_row.addStretch()
        controls.content_layout.addLayout(second_row)

        search_row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索文件名或路径")
        self.search_edit.setClearButtonEnabled(True)
        self.status_filter = QComboBox()
        self.status_filter.addItem("全部状态", None)
        for status, text in STATUS_TEXT.items():
            self.status_filter.addItem(text, status)
        self.search_edit.textChanged.connect(self._apply_filter)
        self.status_filter.currentIndexChanged.connect(self._apply_filter)
        search_row.addWidget(QLabel("筛选"))
        search_row.addWidget(self.search_edit, 1)
        search_row.addWidget(self.status_filter)
        controls.content_layout.addLayout(search_row)

        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0 / 0")
        self.current_file_label = QLabel("尚未开始")
        self.current_file_label.setMinimumWidth(240)
        self.state_label = QLabel("等待文件")
        self.state_label.setObjectName("statusNeutral")
        progress_row.addWidget(self.progress_bar, 1)
        progress_row.addWidget(self.current_file_label)
        progress_row.addWidget(self.state_label)
        controls.content_layout.addLayout(progress_row)

        stats_row = QHBoxLayout()
        self.stats_labels: dict[str, QLabel] = {}
        for key, title in (
            ("total", "总数"),
            ("success", "成功"),
            ("failed", "失败"),
            ("skipped", "跳过"),
            ("stopped", "停止"),
            ("pending", "待处理"),
            ("elapsed", "已用时"),
            ("average", "平均"),
            ("eta", "预计剩余"),
        ):
            label = QLabel(f"{title}：—")
            self.stats_labels[key] = label
            stats_row.addWidget(label)
        stats_row.addStretch()
        controls.content_layout.addLayout(stats_row)
        self.feedback_label = QLabel()
        self.feedback_label.setObjectName("operationFeedback")
        self.feedback_label.setWordWrap(True)
        self.feedback_label.hide()
        controls.content_layout.addWidget(self.feedback_label)
        layout.addWidget(controls)

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)
        table_card = SectionCard("任务明细")
        self.task_table = create_table(
            (
                "序号",
                "文件名",
                "目录",
                "状态",
                "预测组合",
                "结果",
                "主要概率",
                "耗时",
                "错误信息",
            ),
            minimum_height=310,
        )
        self.task_table.setObjectName("batchTaskTable")
        self.task_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.task_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.task_table.setSortingEnabled(True)
        self.task_table.horizontalHeader().setSortIndicator(
            0, Qt.SortOrder.AscendingOrder
        )
        self.task_table.itemSelectionChanged.connect(self._show_selected_detail)
        self.task_table.itemDoubleClicked.connect(lambda _: self._show_selected_detail())
        table_card.content_layout.addWidget(self.task_table)
        splitter.addWidget(table_card)

        detail_card = SectionCard("文件详情")
        self.detail_view = QPlainTextEdit()
        self.detail_view.setObjectName("predictionDetails")
        self.detail_view.setReadOnly(True)
        self.detail_view.setPlaceholderText("选择一条任务查看输入、模型输出或完整错误。")
        detail_card.content_layout.addWidget(self.detail_view)
        self.open_folder_button = self._button(
            "打开所在目录", QStyle.StandardPixmap.SP_DirOpenIcon, self._open_selected_folder
        )
        detail_actions = QHBoxLayout()
        detail_actions.addWidget(self.open_folder_button)
        self.open_logs_button = self._button(
            "查看系统日志",
            QStyle.StandardPixmap.SP_FileDialogDetailedView,
            self.logs_requested.emit,
        )
        detail_actions.addWidget(self.open_logs_button)
        detail_actions.addStretch()
        detail_card.content_layout.addLayout(detail_actions)
        splitter.addWidget(detail_card)
        splitter.setSizes([860, 330])
        layout.addWidget(splitter, 1)

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self.refresh_summary)
        self._elapsed_timer.start(500)
        self._update_actions()

    def set_task(self, task: BatchPredictionTask) -> None:
        self.task = task
        self.refresh_table()

    def set_model_available(self, available: bool) -> None:
        self.model_available = available
        self._update_actions()

    def refresh_table(self, *_: Any) -> None:
        task = self.task
        selected = set(self._selected_ids())
        self.task_table.setSortingEnabled(False)
        self.task_table.setRowCount(0 if task is None else len(task.items))
        if task is not None:
            for row, item in enumerate(task.items):
                values = (
                    str(item.sequence),
                    item.file_name,
                    str(item.file_path.parent),
                    STATUS_TEXT[item.status],
                    item.predicted_combination or "—",
                    ", ".join(item.predicted_sources) or "—",
                    primary_probability_summary(item.result or {}),
                    f"{item.elapsed_ms:.1f} ms" if item.elapsed_ms is not None else "—",
                    item.error_message or "—",
                )
                for column, value in enumerate(values):
                    cell = QTableWidgetItem(value)
                    if column == 0:
                        cell.setData(Qt.ItemDataRole.DisplayRole, item.sequence)
                    cell.setData(Qt.ItemDataRole.UserRole, item.item_id)
                    cell.setToolTip(value)
                    if column == 3:
                        cell.setIcon(self._status_icon(item.status))
                    if column in {0, 3, 7}:
                        cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.task_table.setItem(row, column, cell)
                if item.item_id in selected:
                    self.task_table.selectRow(row)
        self.task_table.setSortingEnabled(True)
        self.task_table.resizeColumnsToContents()
        self.task_table.horizontalHeader().setStretchLastSection(True)
        self._apply_filter()
        self.refresh_summary()
        self._update_actions()
        self._show_selected_detail()

    def refresh_item(self, item: BatchFileItem) -> None:
        if item.status in {BatchItemStatus.VALIDATING, BatchItemStatus.RUNNING}:
            self.current_file_label.setText(f"当前：{item.file_name}")
        self.refresh_table()

    def refresh_summary(self, *_: Any) -> None:
        task = self.task
        if task is None:
            return
        task.refresh_counts()
        completed = task.completed_count
        value = round(task.progress * 10)
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{completed} / {task.total_count}  ({task.progress:.1f}%)")
        self.stats_labels["total"].setText(f"总数：{task.total_count}")
        self.stats_labels["success"].setText(f"成功：{task.success_count}")
        self.stats_labels["failed"].setText(f"失败：{task.failed_count}")
        self.stats_labels["skipped"].setText(f"跳过：{task.skipped_count}")
        self.stats_labels["stopped"].setText(f"停止：{task.stopped_count}")
        self.stats_labels["pending"].setText(f"待处理：{task.pending_count}")
        self.stats_labels["elapsed"].setText(f"已用时：{self._duration(task.elapsed_seconds)}")
        average = task.average_item_seconds
        eta = task.estimated_remaining_seconds
        self.stats_labels["average"].setText(
            f"平均：{self._duration(average)}" if average is not None else "平均：计算中"
        )
        self.stats_labels["eta"].setText(
            f"预计剩余：{self._duration(eta)}" if eta is not None else "预计剩余：计算中"
        )
        state = {
            BatchStatus.CREATED: "等待开始",
            BatchStatus.RUNNING: "批量推理中",
            BatchStatus.PAUSED: "已暂停",
            BatchStatus.STOPPING: "正在停止",
            BatchStatus.STOPPED: "已停止",
            BatchStatus.COMPLETED: "已完成",
            BatchStatus.COMPLETED_WITH_ERRORS: "完成（有失败）",
            BatchStatus.FAILED: "任务失败",
        }[task.status]
        self.state_label.setText(state)

    def show_scan_summary(self, summary: BatchScanSummary) -> None:
        self.show_feedback(
            f"扫描 {summary.scanned_count} 项，新增 {summary.added_count} 个 CSV，"
            f"忽略重复 {summary.duplicate_count} 个，无效 {summary.invalid_count} 个。"
        )
        self.refresh_table()

    def show_export_result(self, result: Any) -> None:
        self.show_feedback(f"结果已导出：{result.output_directory}")
        self.refresh_summary()

    def show_pause_requested(self) -> None:
        """Expose the cooperative file-boundary pause transition."""
        self.state_label.setText("正在暂停")
        self.show_feedback("当前文件完成后暂停，不会启动下一个文件。")

    def show_feedback(self, message: str, *, error: bool = False) -> None:
        self.feedback_label.setText(message)
        self.feedback_label.setProperty("error", error)
        self.feedback_label.style().unpolish(self.feedback_label)
        self.feedback_label.style().polish(self.feedback_label)
        self.feedback_label.show()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.paths_added.emit(paths, self.recursive_checkbox.isChecked())
            event.acceptProposedAction()

    def _button(
        self,
        text: str,
        icon: QStyle.StandardPixmap | None = None,
        callback: Any | None = None,
        *,
        primary: bool = False,
    ) -> QPushButton:
        button = QPushButton(text)
        if icon is not None:
            button.setIcon(self.style().standardIcon(icon))
        if callback is not None:
            button.clicked.connect(callback)
        if primary:
            button.setObjectName("primaryButton")
        return button

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "添加 CSV 文件", "", "CSV 文件 (*.csv)")
        if paths:
            self.paths_added.emit([Path(path) for path in paths], False)

    def _choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "添加文件夹")
        if path:
            self.paths_added.emit([Path(path)], self.recursive_checkbox.isChecked())

    def _choose_export(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if path:
            self.export_requested.emit(Path(path))

    def _selected_ids(self) -> list[str]:
        ids: list[str] = []
        for index in self.task_table.selectionModel().selectedRows():
            item = self.task_table.item(index.row(), 0)
            if item is not None:
                ids.append(str(item.data(Qt.ItemDataRole.UserRole)))
        return ids

    def _remove_selected(self) -> None:
        ids = self._selected_ids()
        if ids:
            self.remove_requested.emit(ids)

    def _retry_selected(self) -> None:
        ids = self._selected_ids()
        if ids:
            self.retry_requested.emit(ids)

    def _selected_item(self) -> BatchFileItem | None:
        if self.task is None:
            return None
        selected = self._selected_ids()
        if not selected:
            return None
        return next((item for item in self.task.items if item.item_id == selected[0]), None)

    def _show_selected_detail(self) -> None:
        item = self._selected_item()
        if item is None:
            self.detail_view.clear()
            self._update_actions()
            return
        payload = {
            "task_id": self.task.task_id if self.task is not None else "",
            "model_name": self.task.model_name if self.task is not None else "",
            "model_version": self.task.model_version if self.task is not None else "",
            "runtime_version": self.task.runtime_version if self.task is not None else "",
            "device": self.task.device if self.task is not None else "",
            "sequence": item.sequence,
            "file_path": str(item.file_path),
            "file_size": item.file_size,
            "modified_at": item.modified_at,
            "status": item.status.value,
            "status_message": item.status_message,
            "elapsed_ms": item.elapsed_ms,
            "labels": item.labels,
            "decision_mode": item.decision_mode,
            "display_probabilities": item.display_probabilities,
            "predicted_combination": item.predicted_combination,
            "predicted_sources": item.predicted_sources,
            "decoded_label_vector": item.decoded_label_vector,
            "error_type": item.error_type,
            "error_message": item.error_message,
            "failure_stage": item.status_message if item.status == BatchItemStatus.FAILED else "",
            "retry_available": item.status == BatchItemStatus.FAILED,
            "retry_count": item.retry_count,
            "runtime_result": item.result,
        }
        self.detail_view.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))
        self._update_actions()

    def _open_selected_folder(self) -> None:
        item = self._selected_item()
        if item is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(item.file_path.parent)))

    def _apply_filter(self, *_: Any) -> None:
        query = self.search_edit.text().strip().casefold()
        status = self.status_filter.currentData()
        if self.task is None:
            return
        by_id = {item.item_id: item for item in self.task.items}
        for row in range(self.task_table.rowCount()):
            cell = self.task_table.item(row, 0)
            item = by_id.get(str(cell.data(Qt.ItemDataRole.UserRole))) if cell else None
            matches = item is not None
            if item is not None and query:
                matches = query in f"{item.file_name} {item.file_path}".casefold()
            if item is not None and status is not None:
                matches = matches and item.status == status
            self.task_table.setRowHidden(row, not matches)

    def _update_actions(self) -> None:
        task = self.task
        running = task is not None and task.status in RUNNING_BATCH_STATUSES
        paused = task is not None and task.status == BatchStatus.PAUSED
        editable = not running
        has_items = bool(task and task.items)
        has_failed = bool(task and task.failed_count)
        selected_ids = set(self._selected_ids()) if task is not None else set()
        has_selected = bool(selected_ids)
        selected_failed = bool(
            task
            and any(
                item.item_id in selected_ids and item.status == BatchItemStatus.FAILED
                for item in task.items
            )
        )
        startable = bool(
            task
            and any(
                item.status in {BatchItemStatus.PENDING, BatchItemStatus.STOPPED}
                for item in task.items
            )
        )
        self.add_files_button.setEnabled(editable)
        self.add_folder_button.setEnabled(editable)
        self.recursive_checkbox.setEnabled(editable)
        self.remove_button.setEnabled(editable and has_selected)
        self.clear_button.setEnabled(editable and has_items)
        self.deduplicate_button.setEnabled(editable and has_items)
        self.start_button.setEnabled(editable and startable and self.model_available)
        self.pause_button.setEnabled(running and not paused)
        self.resume_button.setEnabled(paused)
        self.stop_button.setEnabled(running)
        self.retry_selected_button.setEnabled(editable and selected_failed)
        self.retry_all_button.setEnabled(editable and has_failed)
        self.export_button.setEnabled(bool(task and task.completed_count and not running))
        self.open_folder_button.setEnabled(has_selected)
        self.open_logs_button.setEnabled(has_selected)

    @staticmethod
    def _duration(seconds: float | None) -> str:
        if seconds is None:
            return "—"
        if seconds < 60:
            return f"{seconds:.1f} 秒"
        minutes, remaining = divmod(int(seconds), 60)
        return f"{minutes} 分 {remaining} 秒"

    def _status_icon(self, status: BatchItemStatus) -> Any:
        pixmap = {
            BatchItemStatus.PENDING: QStyle.StandardPixmap.SP_BrowserReload,
            BatchItemStatus.VALIDATING: QStyle.StandardPixmap.SP_BrowserReload,
            BatchItemStatus.RUNNING: QStyle.StandardPixmap.SP_MediaPlay,
            BatchItemStatus.SUCCESS: QStyle.StandardPixmap.SP_DialogApplyButton,
            BatchItemStatus.FAILED: QStyle.StandardPixmap.SP_MessageBoxCritical,
            BatchItemStatus.SKIPPED: QStyle.StandardPixmap.SP_MessageBoxWarning,
            BatchItemStatus.STOPPED: QStyle.StandardPixmap.SP_MediaStop,
        }[status]
        return self.style().standardIcon(pixmap)


__all__ = ["BatchPredictionPage"]
