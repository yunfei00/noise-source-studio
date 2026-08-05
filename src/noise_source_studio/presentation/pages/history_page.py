"""Unified, database-backed task history page."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from PySide6.QtCore import QDate, QItemSelection, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from noise_source_studio.domain.history import (
    HistoryOverview,
    HistoryPageResult,
    HistoryQuery,
    HistoryStatus,
    TaskHistoryRecord,
    TaskType,
)
from noise_source_studio.presentation.models.history import HistoryTableModel
from noise_source_studio.presentation.widgets import (
    PAGE_CONTENT_MARGINS,
    PAGE_CONTENT_SPACING,
    PageHeader,
    SectionCard,
)


class HistoryPage(QWidget):
    """Search, inspect, and operate on all indexed task kinds."""

    query_requested = Signal(object)
    open_task_requested = Signal(str)
    open_directory_requested = Signal(str)
    verify_requested = Signal(str)
    notes_requested = Signal(str, str, object)
    delete_requested = Signal(str, bool)
    export_requested = Signal(object, object)
    scan_requested = Signal()
    scan_cancel_requested = Signal()
    rebuild_requested = Signal()
    backup_requested = Signal()
    database_directory_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._page = 1
        self._page_count = 1
        self._sort_by = "created_at"
        self._descending = True
        self._current_record: TaskHistoryRecord | None = None

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(*PAGE_CONTENT_MARGINS)
        layout.setSpacing(PAGE_CONTENT_SPACING)
        layout.addWidget(
            PageHeader(
                "任务历史",
                "统一查询单文件预测、批量预测和模型验证；完整结果仍保存在文件系统。",
            )
        )
        layout.addWidget(self._build_overview())
        layout.addWidget(self._build_filters())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_table_card())
        splitter.addWidget(self._build_detail_card())
        splitter.setSizes([840, 390])
        layout.addWidget(splitter, 1)
        layout.addWidget(self._build_database_bar())
        scroll.setWidget(content)
        root_layout.addWidget(scroll)

        self.search_timer = QTimer(self)
        self.search_timer.setSingleShot(True)
        self.search_timer.setInterval(250)
        self.search_timer.timeout.connect(self.request_query)
        for edit in (
            self.keyword_edit,
            self.model_filter,
            self.version_filter,
            self.device_filter,
        ):
            edit.textChanged.connect(lambda _text: self.search_timer.start())
        for combo in (self.type_filter, self.status_filter):
            combo.currentIndexChanged.connect(self._filters_changed)
        for check in (
            self.date_filter_enabled,
            self.failed_only,
            self.missing_only,
        ):
            check.toggled.connect(self._filters_changed)
        self.start_date.dateChanged.connect(self._filters_changed)
        self.end_date.dateChanged.connect(self._filters_changed)
        self.page_size.currentIndexChanged.connect(self._page_size_changed)
        self.table_model.sort_requested.connect(self._sort_changed)

    def current_query(self) -> HistoryQuery:
        """Build the authoritative database query from current controls."""
        task_type = self.type_filter.currentData()
        status = self.status_filter.currentData()
        date_from = ""
        date_to = ""
        if self.date_filter_enabled.isChecked():
            date_from = self.start_date.date().toString("yyyy-MM-dd") + "T00:00:00"
            date_to = self.end_date.date().toString("yyyy-MM-dd") + "T23:59:59.999999"
        return HistoryQuery(
            task_type=TaskType(task_type) if task_type else None,
            status=HistoryStatus(status) if status else None,
            date_from=date_from,
            date_to=date_to,
            model_name=self.model_filter.text().strip(),
            model_version=self.version_filter.text().strip(),
            device=self.device_filter.text().strip(),
            keyword=self.keyword_edit.text().strip(),
            failed_only=self.failed_only.isChecked(),
            missing_only=self.missing_only.isChecked(),
            page=self._page,
            page_size=int(self.page_size.currentData()),
            sort_by=self._sort_by,
            descending=self._descending,
        )

    def request_query(self) -> None:
        self.loading_label.setText("正在查询历史索引…")
        self.query_requested.emit(self.current_query())

    def show_results(
        self,
        result: HistoryPageResult,
        overview: HistoryOverview,
    ) -> None:
        """Render one database page and its aggregate overview."""
        self._page = result.page
        self._page_count = result.page_count
        self.table_model.set_records(result.records)
        self.page_label.setText(
            f"第 {result.page} / {result.page_count} 页 · 共 {result.total:,} 条"
        )
        self.first_button.setEnabled(result.page > 1)
        self.previous_button.setEnabled(result.page > 1)
        self.next_button.setEnabled(result.page < result.page_count)
        self.last_button.setEnabled(result.page < result.page_count)
        self.loading_label.setText("查询完成")
        values = (
            overview.total,
            overview.completed,
            overview.failed,
            overview.interrupted,
            overview.last_seven_days,
            overview.missing_artifacts,
        )
        for label, value in zip(self.overview_values, values, strict=True):
            label.setText(f"{value:,}")
        self._clear_selection()

    def show_error(self, message: str) -> None:
        self.loading_label.setText(message)
        self.loading_label.setProperty("state", "error")

    def set_unavailable(self, message: str) -> None:
        """Degrade only this page when the index cannot initialize."""
        self.show_error(f"历史索引不可用：{message}")
        for widget in (
            self.table,
            self.scan_button,
            self.rebuild_button,
            self.backup_button,
            self.export_list_button,
        ):
            widget.setEnabled(False)

    def show_scan_progress(self, current: int, total: int, path: str) -> None:
        self.scan_progress.show()
        self.cancel_scan_button.show()
        self.scan_progress.setMaximum(max(1, total))
        self.scan_progress.setValue(current)
        self.loading_label.setText(f"扫描 {current}/{total}：{Path(path).name}")

    def show_operation_result(self, message: str, *, refresh: bool = True) -> None:
        self.scan_progress.hide()
        self.cancel_scan_button.hide()
        self.loading_label.setText(message)
        if refresh:
            self.request_query()

    def selected_record(self) -> TaskHistoryRecord | None:
        return self._current_record

    def _build_overview(self) -> QWidget:
        card = SectionCard("历史概览")
        row = QHBoxLayout()
        titles = ("全部任务", "已完成", "失败", "中断", "近 7 天", "文件异常")
        self.overview_values: list[QLabel] = []
        for title in titles:
            unit = QWidget()
            unit_layout = QVBoxLayout(unit)
            unit_layout.setContentsMargins(8, 2, 8, 2)
            title_label = QLabel(title)
            title_label.setObjectName("mutedText")
            value_label = QLabel("0")
            value_label.setObjectName("metricValue")
            unit_layout.addWidget(title_label)
            unit_layout.addWidget(value_label)
            self.overview_values.append(value_label)
            row.addWidget(unit, 1)
        card.content_layout.addLayout(row)
        return card

    def _build_filters(self) -> QWidget:
        card = SectionCard("筛选条件")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        self.type_filter = QComboBox()
        self.type_filter.addItem("全部类型", "")
        self.type_filter.addItem("单文件预测", TaskType.SINGLE.value)
        self.type_filter.addItem("批量预测", TaskType.BATCH.value)
        self.type_filter.addItem("模型验证", TaskType.VALIDATION.value)
        self.status_filter = QComboBox()
        self.status_filter.addItem("全部状态", "")
        for status, label in (
            (HistoryStatus.CREATED, "已创建"),
            (HistoryStatus.RUNNING, "运行中"),
            (HistoryStatus.COMPLETED, "已完成"),
            (HistoryStatus.COMPLETED_WITH_ERRORS, "完成（有错误）"),
            (HistoryStatus.FAILED, "失败"),
            (HistoryStatus.STOPPED, "已停止"),
            (HistoryStatus.INTERRUPTED, "已中断"),
        ):
            self.status_filter.addItem(label, status.value)
        self.date_filter_enabled = QCheckBox("限定日期")
        self.start_date = QDateEdit(QDate.currentDate().addMonths(-1))
        self.start_date.setCalendarPopup(True)
        self.end_date = QDateEdit(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        self.keyword_edit = QLineEdit()
        self.keyword_edit.setObjectName("historyKeyword")
        self.keyword_edit.setPlaceholderText("搜索任务 ID、名称、文件、模型、标签或备注")
        self.failed_only = QCheckBox("仅看失败")
        self.missing_only = QCheckBox("仅看文件异常")
        self.model_filter = QLineEdit()
        self.model_filter.setPlaceholderText("模型名称（精确）")
        self.version_filter = QLineEdit()
        self.version_filter.setPlaceholderText("模型版本（精确）")
        self.device_filter = QLineEdit()
        self.device_filter.setPlaceholderText("设备，如 cpu / cuda:0")
        search_button = QPushButton("立即查询")
        search_button.setObjectName("primaryButton")
        search_button.clicked.connect(self.request_query)
        reset_button = QPushButton("重置")
        reset_button.clicked.connect(self._reset_filters)
        grid.addWidget(self.type_filter, 0, 0)
        grid.addWidget(self.status_filter, 0, 1)
        grid.addWidget(self.date_filter_enabled, 0, 2)
        grid.addWidget(self.start_date, 0, 3)
        grid.addWidget(QLabel("至"), 0, 4)
        grid.addWidget(self.end_date, 0, 5)
        grid.addWidget(self.keyword_edit, 1, 0, 1, 5)
        grid.addWidget(self.failed_only, 1, 5)
        grid.addWidget(self.missing_only, 1, 6)
        grid.addWidget(reset_button, 1, 7)
        grid.addWidget(self.model_filter, 2, 0, 1, 2)
        grid.addWidget(self.version_filter, 2, 2, 1, 2)
        grid.addWidget(self.device_filter, 2, 4, 1, 2)
        grid.addWidget(search_button, 2, 6, 1, 2)
        grid.setColumnStretch(5, 1)
        card.content_layout.addLayout(grid)
        return card

    def _build_table_card(self) -> QWidget:
        card = SectionCard("历史任务")
        self.table_model = HistoryTableModel()
        self.table = QTableView()
        self.table.setObjectName("historyTable")
        self.table.setModel(self.table_model)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(300)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        widths = (145, 90, 170, 115, 130, 80, 55, 55, 55, 80, 190)
        for column, width in enumerate(widths):
            self.table.setColumnWidth(column, width)
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)
        self.table.doubleClicked.connect(lambda _index: self._emit_open())
        card.content_layout.addWidget(self.table)

        pagination = QHBoxLayout()
        self.loading_label = QLabel("等待查询")
        self.loading_label.setObjectName("operationFeedback")
        self.page_size = QComboBox()
        for size in (20, 50, 100, 200):
            self.page_size.addItem(f"每页 {size} 条", size)
        self.page_size.setCurrentIndex(1)
        self.previous_button = QPushButton("上一页")
        self.previous_button.clicked.connect(self._previous_page)
        self.next_button = QPushButton("下一页")
        self.next_button.clicked.connect(self._next_page)
        self.first_button = QPushButton("首页")
        self.first_button.clicked.connect(self._first_page)
        self.last_button = QPushButton("末页")
        self.last_button.clicked.connect(self._last_page)
        self.page_label = QLabel("第 1 / 1 页 · 共 0 条")
        pagination.addWidget(self.loading_label, 1)
        pagination.addWidget(self.page_size)
        pagination.addWidget(self.first_button)
        pagination.addWidget(self.previous_button)
        pagination.addWidget(self.page_label)
        pagination.addWidget(self.next_button)
        pagination.addWidget(self.last_button)
        card.content_layout.addLayout(pagination)
        return card

    def _build_detail_card(self) -> QWidget:
        card = SectionCard("任务详情")
        self.detail_view = QPlainTextEdit()
        self.detail_view.setObjectName("historyDetails")
        self.detail_view.setReadOnly(True)
        self.detail_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.detail_view.setPlaceholderText("选择一条任务查看完整索引信息。")
        card.content_layout.addWidget(self.detail_view, 1)
        actions = QGridLayout()
        self.detail_button = QPushButton("查看详情")
        self.detail_button.clicked.connect(self.detail_view.setFocus)
        self.open_button = QPushButton("在原页面打开")
        self.open_button.setObjectName("primaryButton")
        self.open_button.clicked.connect(self._emit_open)
        self.directory_button = QPushButton("打开结果目录")
        self.directory_button.clicked.connect(self._emit_directory)
        self.copy_id_button = QPushButton("复制任务 ID")
        self.copy_id_button.clicked.connect(self._copy_id)
        self.export_task_button = QPushButton("导出任务摘要")
        self.export_task_button.clicked.connect(self._choose_task_export)
        self.verify_button = QPushButton("校验文件")
        self.verify_button.clicked.connect(self._emit_verify)
        self.notes_button = QPushButton("备注与标签")
        self.notes_button.clicked.connect(self._edit_notes)
        self.delete_index_button = QPushButton("仅删除索引")
        self.delete_index_button.clicked.connect(lambda: self._emit_delete(False))
        self.delete_files_button = QPushButton("删除索引和文件")
        self.delete_files_button.clicked.connect(lambda: self._emit_delete(True))
        for index, button in enumerate(
            (
                self.detail_button,
                self.open_button,
                self.directory_button,
                self.copy_id_button,
                self.export_task_button,
                self.verify_button,
                self.notes_button,
                self.delete_index_button,
                self.delete_files_button,
            )
        ):
            actions.addWidget(button, index // 2, index % 2)
            button.setEnabled(False)
        card.content_layout.addLayout(actions)
        return card

    def _build_database_bar(self) -> QWidget:
        card = SectionCard("历史索引维护")
        row = QHBoxLayout()
        self.scan_button = QPushButton("扫描已有结果")
        self.scan_button.clicked.connect(self.scan_requested)
        self.rebuild_button = QPushButton("重建索引")
        self.rebuild_button.clicked.connect(self.rebuild_requested)
        self.backup_button = QPushButton("备份数据库")
        self.backup_button.clicked.connect(self.backup_requested)
        database_button = QPushButton("打开数据库目录")
        database_button.clicked.connect(self.database_directory_requested)
        self.export_list_button = QPushButton("导出当前筛选摘要")
        self.export_list_button.clicked.connect(self._choose_export)
        self.scan_progress = QProgressBar()
        self.scan_progress.setMaximumWidth(180)
        self.scan_progress.hide()
        self.cancel_scan_button = QPushButton("取消扫描")
        self.cancel_scan_button.clicked.connect(self.scan_cancel_requested)
        self.cancel_scan_button.hide()
        for button in (
            self.scan_button,
            self.rebuild_button,
            self.backup_button,
            database_button,
            self.export_list_button,
        ):
            row.addWidget(button)
        row.addStretch()
        row.addWidget(self.scan_progress)
        row.addWidget(self.cancel_scan_button)
        card.content_layout.addLayout(row)
        return card

    def _filters_changed(self, _value: object = None) -> None:
        self._page = 1
        self.request_query()

    def _reset_filters(self) -> None:
        self.type_filter.setCurrentIndex(0)
        self.status_filter.setCurrentIndex(0)
        self.date_filter_enabled.setChecked(False)
        self.failed_only.setChecked(False)
        self.missing_only.setChecked(False)
        self.keyword_edit.clear()
        self.model_filter.clear()
        self.version_filter.clear()
        self.device_filter.clear()
        self._page = 1
        self.request_query()

    def _page_size_changed(self, _index: int) -> None:
        self._page = 1
        self.request_query()

    def _sort_changed(self, key: str, descending: bool) -> None:
        self._sort_by = key
        self._descending = descending
        self._page = 1
        self.request_query()

    def _previous_page(self) -> None:
        if self._page > 1:
            self._page -= 1
            self.request_query()

    def _first_page(self) -> None:
        if self._page != 1:
            self._page = 1
            self.request_query()

    def _next_page(self) -> None:
        if self._page < self._page_count:
            self._page += 1
            self.request_query()

    def _last_page(self) -> None:
        if self._page != self._page_count:
            self._page = self._page_count
            self.request_query()

    def _selection_changed(
        self,
        selected: QItemSelection,
        _deselected: QItemSelection,
    ) -> None:
        indexes = selected.indexes()
        record = self.table_model.record_at(indexes[0].row()) if indexes else None
        self._current_record = record
        enabled = record is not None
        for button in (
            self.detail_button,
            self.open_button,
            self.directory_button,
            self.copy_id_button,
            self.export_task_button,
            self.verify_button,
            self.notes_button,
            self.delete_index_button,
            self.delete_files_button,
        ):
            button.setEnabled(enabled)
        if record is None:
            self.detail_view.clear()
            return
        payload = asdict(record)
        payload["task_type"] = record.task_type.value
        payload["status"] = record.status.value
        payload["integrity_status"] = record.integrity_status.value
        payload["file_exists"] = {
            key: Path(value).is_file()
            for key, value in (
                ("task_file", record.task_file),
                ("summary_file", record.summary_file),
                ("detail_file", record.detail_file),
                ("error_file", record.error_file),
                ("report_file", record.report_file),
                ("manifest_file", record.manifest_file),
            )
            if value
        }
        self.detail_view.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))

    def _clear_selection(self) -> None:
        self.table.clearSelection()
        self._selection_changed(QItemSelection(), QItemSelection())

    def _emit_open(self) -> None:
        if self._current_record is not None:
            self.open_task_requested.emit(self._current_record.task_id)

    def _emit_directory(self) -> None:
        if self._current_record is not None:
            self.open_directory_requested.emit(self._current_record.task_id)

    def _emit_verify(self) -> None:
        if self._current_record is not None:
            self.verify_requested.emit(self._current_record.task_id)

    def _emit_delete(self, with_files: bool) -> None:
        if self._current_record is not None:
            self.delete_requested.emit(self._current_record.task_id, with_files)

    def _copy_id(self) -> None:
        if self._current_record is not None:
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(self._current_record.task_id)
            self.loading_label.setText("任务 ID 已复制")

    def _edit_notes(self) -> None:
        if self._current_record is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("备注与标签")
        dialog.resize(480, 320)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        tags_edit = QLineEdit(", ".join(self._current_record.tags))
        tags_edit.setPlaceholderText("多个标签使用逗号分隔")
        notes_edit = QTextEdit(self._current_record.notes)
        form.addRow("标签", tags_edit)
        form.addRow("备注", notes_edit)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            tags = tuple(
                value.strip()
                for value in tags_edit.text().replace("，", ",").split(",")
                if value.strip()
            )
            self.notes_requested.emit(
                self._current_record.task_id,
                notes_edit.toPlainText(),
                tags,
            )

    def _choose_export(self) -> None:
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "导出历史任务摘要",
            "task_history.csv",
            "CSV 文件 (*.csv)",
        )
        if path:
            self.export_requested.emit(replace(self.current_query(), page=1), Path(path))

    def _choose_task_export(self) -> None:
        if self._current_record is None:
            return
        default_name = f"history_{self._current_record.task_id}.csv"
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "导出任务摘要",
            default_name,
            "CSV 文件 (*.csv)",
        )
        if path:
            self.export_requested.emit(
                HistoryQuery(keyword=self._current_record.task_id, page_size=20),
                Path(path),
            )


__all__ = ["HistoryPage"]
