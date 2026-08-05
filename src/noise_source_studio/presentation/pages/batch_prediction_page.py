"""Batch prediction workflow and in-application result center."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import QItemSelectionModel, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStyle,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
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
from noise_source_studio.domain.confidence import (
    candidate_probability_margin,
    combination_candidates,
    highest_combination_probability,
    is_low_confidence,
)
from noise_source_studio.domain.models import SignalPreview
from noise_source_studio.presentation.models import (
    BatchResultFilterProxyModel,
    BatchResultTableModel,
    BatchTaskFilterProxyModel,
    BatchTaskTableModel,
)
from noise_source_studio.presentation.widgets import (
    PAGE_CONTENT_MARGINS,
    PAGE_CONTENT_SPACING,
    PageHeader,
    SectionCard,
)
from noise_source_studio.services.batch_prediction_service import BatchPredictionService

STATUS_TEXT = {
    BatchItemStatus.PENDING: "等待",
    BatchItemStatus.VALIDATING: "校验中",
    BatchItemStatus.RUNNING: "推理中",
    BatchItemStatus.SUCCESS: "成功",
    BatchItemStatus.FAILED: "失败",
    BatchItemStatus.SKIPPED: "跳过",
    BatchItemStatus.STOPPED: "已停止",
}
CONFIDENCE_RANGES = {
    "0–40%": (0.0, 0.4),
    "40–60%": (0.4, 0.6),
    "60–80%": (0.6, 0.8),
    "80–100%": (0.8, 1.0),
    "无置信度": (0.0, 1.0),
}


class BatchPredictionPage(QWidget):
    """Interactive batch execution plus searchable result and statistics tabs."""

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
    filtered_export_requested = Signal(object, object)
    history_requested = Signal(object)
    waveform_requested = Signal(object)
    logs_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.task: BatchPredictionTask | None = None
        self.model_available = False
        self.last_comparison_dialog: QDialog | None = None
        self._selected_result_item: BatchFileItem | None = None
        self.dirty_item_ids: set[str] = set()
        self._task_row_by_item_id: dict[str, int] = {}
        self._task_item_by_id: dict[str, BatchFileItem] = {}
        self._known_combinations: set[str] = set()
        self._known_sources: set[str] = set()
        self._timed_item_ids: set[str] = set()
        self._elapsed_item_total_ms = 0.0
        self._summary_dirty = False
        self._statistics_dirty = False
        self._result_view_dirty = False
        self._last_chart_refresh = 0.0
        self._last_result_view_refresh = 0.0
        self._task_sort_column: int | None = None
        self._result_sort_column: int | None = None
        self.full_table_rebuild_count = 0
        self.item_row_update_count = 0
        self.chart_refresh_count = 0
        self.summary_refresh_count = 0
        self.main_thread_longest_refresh_ms = 0.0
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*PAGE_CONTENT_MARGINS)
        layout.setSpacing(PAGE_CONTENT_SPACING)
        layout.addWidget(
            PageHeader(
                "批量预测",
                "顺序执行文件预测，并在应用内完成结果检索、详情查看、统计分析和历史恢复。",
            )
        )
        layout.addWidget(self._build_controls())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_file_tasks_tab(), "文件任务")
        self.tabs.addTab(self._build_results_tab(), "预测结果")
        self.tabs.addTab(self._build_statistics_tab(), "统计分析")
        self.tabs.currentChanged.connect(self._tab_changed)
        layout.addWidget(self.tabs, 1)

        self._item_update_timer = QTimer(self)
        self._item_update_timer.setSingleShot(True)
        self._item_update_timer.setInterval(150)
        self._item_update_timer.timeout.connect(self.flush_pending_ui_updates)
        self._ui_refresh_timer = QTimer(self)
        self._ui_refresh_timer.setInterval(350)
        self._ui_refresh_timer.timeout.connect(self._refresh_dirty_ui)
        self._ui_refresh_timer.start()
        self._update_actions()

    def _build_controls(self) -> SectionCard:
        controls = SectionCard("批量任务")
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
        self.history_button = self._button(
            "打开历史批量结果",
            QStyle.StandardPixmap.SP_DirOpenIcon,
            self._choose_history,
        )
        for widget in (
            self.add_files_button,
            self.add_folder_button,
            self.recursive_checkbox,
            self.remove_button,
            self.clear_button,
            self.deduplicate_button,
            self.history_button,
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
            "导出批次副本",
            QStyle.StandardPixmap.SP_DialogSaveButton,
            self._choose_export,
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

        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0 / 0")
        self.current_file_label = QLabel("尚未开始")
        self.current_file_label.setMinimumWidth(220)
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
        return controls

    def _build_file_tasks_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        filter_row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索文件名或路径")
        self.search_edit.setClearButtonEnabled(True)
        self.status_filter = QComboBox()
        self.status_filter.addItem("全部状态", None)
        for status, text in STATUS_TEXT.items():
            self.status_filter.addItem(text, status)
        self.search_edit.textChanged.connect(self._apply_task_filter)
        self.status_filter.currentIndexChanged.connect(self._apply_task_filter)
        filter_row.addWidget(QLabel("筛选"))
        filter_row.addWidget(self.search_edit, 1)
        filter_row.addWidget(self.status_filter)
        layout.addLayout(filter_row)

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)
        table_card = SectionCard("任务明细")
        self.task_model = BatchTaskTableModel()
        self.task_proxy = BatchTaskFilterProxyModel()
        self.task_proxy.setSourceModel(self.task_model)
        self.task_table = QTableView()
        self.task_table.setObjectName("batchTaskTable")
        self.task_table.setModel(self.task_proxy)
        self.task_table.setMinimumHeight(310)
        self.task_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.task_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.task_table.setAlternatingRowColors(True)
        self.task_table.verticalHeader().setVisible(False)
        self.task_table.setSortingEnabled(False)
        self.task_table.horizontalHeader().setSortIndicatorShown(True)
        self.task_table.horizontalHeader().setSortIndicator(0, Qt.SortOrder.AscendingOrder)
        self.task_table.horizontalHeader().sectionClicked.connect(self._sort_task_table)
        self.task_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        task_widths = (55, 150, 250, 90, 110, 160, 110, 95, 260)
        for column, width in enumerate(task_widths):
            self.task_table.setColumnWidth(column, width)
        self.task_table.horizontalHeader().setStretchLastSection(True)
        self.task_table.selectionModel().selectionChanged.connect(self._show_task_detail)
        table_card.content_layout.addWidget(self.task_table)
        splitter.addWidget(table_card)

        detail_card = SectionCard("文件任务详情")
        self.detail_view = QPlainTextEdit()
        self.detail_view.setObjectName("predictionDetails")
        self.detail_view.setReadOnly(True)
        self.detail_view.setPlaceholderText("选择一条任务查看输入、输出或完整错误。")
        detail_card.content_layout.addWidget(self.detail_view)
        detail_actions = QHBoxLayout()
        self.open_folder_button = self._button(
            "打开所在目录",
            QStyle.StandardPixmap.SP_DirOpenIcon,
            self._open_selected_folder,
        )
        self.open_logs_button = self._button(
            "查看系统日志",
            QStyle.StandardPixmap.SP_FileDialogDetailedView,
            self.logs_requested.emit,
        )
        detail_actions.addWidget(self.open_folder_button)
        detail_actions.addWidget(self.open_logs_button)
        detail_actions.addStretch()
        detail_card.content_layout.addLayout(detail_actions)
        splitter.addWidget(detail_card)
        splitter.setSizes([860, 330])
        layout.addWidget(splitter, 1)
        return tab

    def _build_results_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        filter_card = SectionCard("结果筛选")

        row_one = QHBoxLayout()
        self.result_search = QLineEdit()
        self.result_search.setPlaceholderText("搜索文件名或路径")
        self.result_search.setClearButtonEnabled(True)
        self.result_status_filter = QComboBox()
        self.result_status_filter.addItem("全部状态", None)
        for status, text in STATUS_TEXT.items():
            self.result_status_filter.addItem(text, status)
        self.combination_filter = QComboBox()
        self.combination_filter.addItem("全部预测组合", "")
        self.confidence_min = QDoubleSpinBox()
        self.confidence_min.setRange(0.0, 1.0)
        self.confidence_min.setSingleStep(0.05)
        self.confidence_min.setDecimals(2)
        self.confidence_max = QDoubleSpinBox()
        self.confidence_max.setRange(0.0, 1.0)
        self.confidence_max.setValue(1.0)
        self.confidence_max.setSingleStep(0.05)
        self.confidence_max.setDecimals(2)
        row_one.addWidget(self.result_search, 2)
        row_one.addWidget(self.result_status_filter)
        row_one.addWidget(self.combination_filter)
        row_one.addWidget(QLabel("可信度"))
        row_one.addWidget(self.confidence_min)
        row_one.addWidget(QLabel("至"))
        row_one.addWidget(self.confidence_max)
        filter_card.content_layout.addLayout(row_one)

        row_two = QHBoxLayout()
        self.source_filter = QListWidget()
        self.source_filter.setMaximumHeight(72)
        self.source_filter.setMinimumWidth(230)
        self.source_filter.setToolTip("可多选噪声源；匹配任一选中来源")
        self.low_confidence_checkbox = QCheckBox("仅看低置信度")
        self.error_only_checkbox = QCheckBox("仅看错误")
        self.filtered_count_label = QLabel("0 条结果")
        self.export_filtered_button = self._button(
            "导出当前筛选结果",
            QStyle.StandardPixmap.SP_DialogSaveButton,
            self._choose_filtered_export,
        )
        self.compare_button = self._button("对比结果", callback=self._compare_selected_results)
        row_two.addWidget(QLabel("噪声源多选"))
        row_two.addWidget(self.source_filter)
        row_two.addWidget(self.low_confidence_checkbox)
        row_two.addWidget(self.error_only_checkbox)
        row_two.addStretch()
        row_two.addWidget(self.filtered_count_label)
        row_two.addWidget(self.compare_button)
        row_two.addWidget(self.export_filtered_button)
        filter_card.content_layout.addLayout(row_two)
        layout.addWidget(filter_card)

        self.result_model = BatchResultTableModel()
        self.result_proxy = BatchResultFilterProxyModel()
        self.result_proxy.setSourceModel(self.result_model)
        self.result_table = QTableView()
        self.result_table.setObjectName("batchResultTable")
        self.result_table.setModel(self.result_proxy)
        self.result_table.setSortingEnabled(False)
        self.result_table.horizontalHeader().setSortIndicatorShown(True)
        self.result_table.horizontalHeader().setSortIndicator(0, Qt.SortOrder.AscendingOrder)
        self.result_table.horizontalHeader().sectionClicked.connect(self._sort_result_table)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.result_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        widths = (55, 135, 220, 75, 100, 150, 120, 110, 105, 220)
        for column, width in enumerate(widths):
            self.result_table.setColumnWidth(column, width)

        result_splitter = QSplitter()
        result_splitter.addWidget(self.result_table)
        result_splitter.addWidget(self._build_result_detail_panel())
        result_splitter.setSizes([820, 430])
        layout.addWidget(result_splitter, 1)

        for signal in (
            self.result_search.textChanged,
            self.result_status_filter.currentIndexChanged,
            self.combination_filter.currentIndexChanged,
            self.confidence_min.valueChanged,
            self.confidence_max.valueChanged,
            self.low_confidence_checkbox.toggled,
            self.error_only_checkbox.toggled,
            self.source_filter.itemChanged,
        ):
            signal.connect(self._apply_result_filters)
        self.result_table.selectionModel().selectionChanged.connect(
            self._show_selected_result_detail
        )
        return tab

    def _build_result_detail_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 0, 0)
        title = QLabel("结果详情")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.result_metadata = QPlainTextEdit()
        self.result_metadata.setReadOnly(True)
        self.result_metadata.setObjectName("predictionDetails")
        self.result_metadata.setMaximumHeight(235)
        layout.addWidget(self.result_metadata)
        probability_title = QLabel("动态标签概率")
        probability_title.setObjectName("sectionTitle")
        layout.addWidget(probability_title)
        probability_scroll = QScrollArea()
        probability_scroll.setWidgetResizable(True)
        probability_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.probability_widget = QWidget()
        self.probability_layout = QVBoxLayout(self.probability_widget)
        self.probability_layout.setContentsMargins(0, 0, 0, 0)
        probability_scroll.setWidget(self.probability_widget)
        probability_scroll.setMaximumHeight(170)
        layout.addWidget(probability_scroll)
        waveform_title = QLabel("按需原始信号波形")
        waveform_title.setObjectName("sectionTitle")
        layout.addWidget(waveform_title)
        self.result_waveform = pg.PlotWidget()
        self.result_waveform.setBackground("#ffffff")
        self.result_waveform.showGrid(x=True, y=True, alpha=0.18)
        self.result_waveform.setMinimumHeight(170)
        layout.addWidget(self.result_waveform, 1)
        self.waveform_status = QLabel("选择结果后按需读取 CSV，不在批量任务中保存原始波形。")
        self.waveform_status.setObjectName("sectionDescription")
        self.waveform_status.setWordWrap(True)
        layout.addWidget(self.waveform_status)
        return panel

    def _build_statistics_tab(self) -> QWidget:
        tab = QWidget()
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 10, 0, 0)
        metric_card = SectionCard("批次概要")
        metric_row = QHBoxLayout()
        self.analysis_metrics: dict[str, QLabel] = {}
        for key, title in (
            ("total", "总文件数"),
            ("success", "成功数"),
            ("failed", "失败数"),
            ("low", "低置信度"),
            ("elapsed", "总耗时"),
            ("average", "平均耗时"),
        ):
            label = QLabel(f"{title}\n—")
            label.setObjectName("metricValue")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.analysis_metrics[key] = label
            metric_row.addWidget(label, 1)
        metric_card.content_layout.addLayout(metric_row)
        layout.addWidget(metric_card)

        charts = QWidget()
        grid = QGridLayout(charts)
        grid.setContentsMargins(0, 0, 0, 0)
        self.chart_plots: dict[str, pg.PlotWidget] = {}
        for index, (key, title) in enumerate(
            (
                ("combination", "预测组合分布"),
                ("source", "噪声源出现次数"),
                ("confidence", "置信度区间分布"),
                ("error", "错误类型分布"),
            )
        ):
            card = SectionCard(title, "点击柱形可跳转到预测结果并应用筛选")
            plot = pg.PlotWidget()
            plot.setBackground("#ffffff")
            plot.setMinimumHeight(225)
            plot.showGrid(y=True, alpha=0.18)
            card.content_layout.addWidget(plot)
            grid.addWidget(card, index // 2, index % 2)
            self.chart_plots[key] = plot
        layout.addWidget(charts, 1)
        scroll.setWidget(content)
        outer_layout.addWidget(scroll)
        return tab

    def set_task(self, task: BatchPredictionTask) -> None:
        self.dirty_item_ids.clear()
        self._item_update_timer.stop()
        self.task = task
        task.refresh_counts()
        self.result_model.set_task(task)
        self._reset_timing_cache()
        self.rebuild_task_table()
        self._rebuild_result_filter_options()
        self._apply_result_filters()
        self.refresh_summary()
        self.refresh_statistics()
        self._update_actions()
        self._show_task_detail()

    def set_model_available(self, available: bool) -> None:
        self.model_available = available
        self._update_actions()

    def refresh_table(self, *_: Any) -> None:
        """Fully reconcile all views after an intentional bulk task change."""
        task = self.task
        self.dirty_item_ids.clear()
        self._item_update_timer.stop()
        if task is not None:
            task.refresh_counts()
        self._reset_timing_cache()
        self.rebuild_task_table()
        self.result_model.refresh()
        self._rebuild_result_filter_options()
        self._apply_result_filters()
        self.refresh_summary()
        self.refresh_statistics()
        self._update_actions()
        self._show_task_detail()

    def rebuild_task_table(self) -> None:
        """Rebuild the task table only for bulk queue changes or final calibration."""
        started = perf_counter()
        task = self.task
        selected = set(self._selected_ids())
        self.task_model.set_task(task)
        self._reindex_task_rows()
        self._apply_task_filter()
        selection = self.task_table.selectionModel()
        for item_id in selected:
            source_row = self._locate_task_row(item_id)
            if source_row is None:
                continue
            proxy_index = self.task_proxy.mapFromSource(self.task_model.index(source_row, 0))
            if proxy_index.isValid():
                selection.select(
                    proxy_index,
                    QItemSelectionModel.SelectionFlag.Select
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
        self.full_table_rebuild_count += 1
        self._record_ui_refresh(started)

    def insert_item_row(self, item: BatchFileItem) -> None:
        """Insert one task row while keeping the item-to-row index valid."""
        self.task_model.insert_item(item)
        self._reindex_task_rows()

    def update_item_row(self, item: BatchFileItem) -> None:
        """Update only the mutable cells for one item."""
        self._update_item_rows([item])

    def remove_item_row(self, item_id: str) -> None:
        """Remove one indexed row without rebuilding unaffected rows."""
        self.task_model.remove_item(item_id)
        self._reindex_task_rows()

    def refresh_item(self, item: BatchFileItem) -> None:
        if item.status in {BatchItemStatus.VALIDATING, BatchItemStatus.RUNNING}:
            self.current_file_label.setText(f"当前：{item.file_name}")
        self.dirty_item_ids.add(item.item_id)
        self._summary_dirty = True
        self._statistics_dirty = True
        self._result_view_dirty = True
        if not self._item_update_timer.isActive():
            self._item_update_timer.start()

    def flush_pending_ui_updates(self) -> None:
        """Apply coalesced worker notifications on the Qt main thread."""
        self._item_update_timer.stop()
        if not self.dirty_item_ids:
            return
        item_ids = set(self.dirty_item_ids)
        self.dirty_item_ids.clear()
        items = [
            item
            for item_id in item_ids
            if (item := self._task_item_by_id.get(item_id)) is not None
        ]
        self._update_item_rows(items)

    def mark_summary_dirty(self, *_: Any) -> None:
        """Defer worker progress signals to the periodic UI refresh."""
        self._summary_dirty = True

    def batch_started(self, *_: Any) -> None:
        """Refresh lifecycle controls once after the worker enters running state."""
        self._summary_dirty = True
        self._update_actions()

    def begin_batch_updates(self) -> None:
        """Reset run-scoped counters without rebuilding the task table."""
        self.reset_performance_counters()
        self._reset_timing_cache()
        self._summary_dirty = True
        self._statistics_dirty = True
        self._result_view_dirty = True
        self.refresh_summary()
        self._update_actions()

    def finalize_batch_updates(self) -> None:
        """Flush and fully calibrate views once when a batch terminates."""
        self.flush_pending_ui_updates()
        if self.task is not None:
            self.task.refresh_counts()
        self._reset_timing_cache()
        self.rebuild_task_table()
        self.result_model.refresh()
        self._rebuild_result_filter_options()
        self._apply_result_filters()
        self.refresh_summary()
        self.refresh_statistics()
        self._summary_dirty = False
        self._statistics_dirty = False
        self._result_view_dirty = False
        self._update_actions()

    def refresh_actions(self) -> None:
        """Update controls without touching task or result data."""
        self._update_actions()

    def reset_performance_counters(self) -> None:
        self.full_table_rebuild_count = 0
        self.item_row_update_count = 0
        self.chart_refresh_count = 0
        self.summary_refresh_count = 0
        self.main_thread_longest_refresh_ms = 0.0

    def performance_counters(self) -> dict[str, int | float]:
        return {
            "full_table_rebuild_count": self.full_table_rebuild_count,
            "item_row_update_count": self.item_row_update_count,
            "chart_refresh_count": self.chart_refresh_count,
            "summary_refresh_count": self.summary_refresh_count,
            "main_thread_longest_refresh_ms": self.main_thread_longest_refresh_ms,
        }

    def _update_item_rows(self, items: list[BatchFileItem]) -> None:
        if not items:
            return
        started = perf_counter()
        updated_ids: set[str] = set()
        for item in items:
            if self._locate_task_row(item.item_id) is None:
                self.task_model.insert_item(item)
            self._task_item_by_id[item.item_id] = item
            self._collect_result_filter_options(item)
            self._record_item_timing(item)
            self.item_row_update_count += 1
            updated_ids.add(item.item_id)

        self.task_model.notify_items_changed(updated_ids)
        self.result_model.notify_items_changed(updated_ids)
        self._record_ui_refresh(started)

    def _reindex_task_rows(self) -> None:
        self._task_row_by_item_id = dict(self.task_model.row_by_item_id)
        self._task_item_by_id = {item.item_id: item for item in self.task_model.items}

    def _locate_task_row(self, item_id: str) -> int | None:
        return self.task_model.row_by_item_id.get(item_id)

    def _reset_timing_cache(self) -> None:
        task = self.task
        timed = [
            item
            for item in (task.items if task is not None else [])
            if item.elapsed_ms is not None
            and item.status in {BatchItemStatus.SUCCESS, BatchItemStatus.FAILED}
        ]
        self._timed_item_ids = {item.item_id for item in timed}
        self._elapsed_item_total_ms = sum(float(item.elapsed_ms or 0.0) for item in timed)

    def _record_item_timing(self, item: BatchFileItem) -> None:
        if (
            item.item_id not in self._timed_item_ids
            and item.elapsed_ms is not None
            and item.status in {BatchItemStatus.SUCCESS, BatchItemStatus.FAILED}
        ):
            self._timed_item_ids.add(item.item_id)
            self._elapsed_item_total_ms += item.elapsed_ms

    def _record_ui_refresh(self, started: float) -> None:
        elapsed_ms = (perf_counter() - started) * 1000.0
        self.main_thread_longest_refresh_ms = max(
            self.main_thread_longest_refresh_ms,
            elapsed_ms,
        )

    def _refresh_dirty_ui(self) -> None:
        task = self.task
        if task is None:
            return
        running = task.status in RUNNING_BATCH_STATUSES
        now = perf_counter()
        if self._summary_dirty or running:
            self.refresh_summary()
        if (
            self._result_view_dirty
            and self.tabs.currentIndex() == 1
            and now - self._last_result_view_refresh >= 1.0
        ):
            self._refresh_visible_result_view()
        if (
            self._statistics_dirty
            and self.tabs.currentIndex() == 2
            and (not running or now - self._last_chart_refresh >= 2.0)
        ):
            self.refresh_statistics()

    def _tab_changed(self, index: int) -> None:
        if index == 1:
            self._refresh_visible_result_view()
        elif index == 2:
            self.refresh_statistics()

    def _refresh_visible_result_view(self) -> None:
        self._update_filtered_count()
        self._result_view_dirty = False
        self._last_result_view_refresh = perf_counter()

    def _collect_result_filter_options(self, item: BatchFileItem) -> None:
        combination = item.predicted_combination
        if combination and combination not in self._known_combinations:
            self._known_combinations.add(combination)
            self.combination_filter.blockSignals(True)
            self.combination_filter.addItem(combination, combination)
            self.combination_filter.blockSignals(False)
        for source in item.predicted_sources:
            if source in self._known_sources:
                continue
            self._known_sources.add(source)
            option = QListWidgetItem(source)
            option.setFlags(option.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            option.setCheckState(Qt.CheckState.Unchecked)
            self.source_filter.blockSignals(True)
            self.source_filter.addItem(option)
            self.source_filter.blockSignals(False)

    def refresh_summary(self, *_: Any) -> None:
        started = perf_counter()
        task = self.task
        if task is None:
            return
        self.progress_bar.setValue(round(task.progress * 10))
        self.progress_bar.setFormat(
            f"{task.completed_count} / {task.total_count}  ({task.progress:.1f}%)"
        )
        average = (
            self._elapsed_item_total_ms / len(self._timed_item_ids) / 1000.0
            if self._timed_item_ids
            else None
        )
        eta = (
            max(0.0, average * (task.pending_count + task.running_count))
            if average is not None and len(self._timed_item_ids) >= 2
            else None
        )
        values = {
            "total": str(task.total_count),
            "success": str(task.success_count),
            "failed": str(task.failed_count),
            "skipped": str(task.skipped_count),
            "stopped": str(task.stopped_count),
            "pending": str(task.pending_count),
            "elapsed": self._duration(task.elapsed_seconds),
            "average": (
                self._duration(average) if average is not None else "计算中"
            ),
            "eta": self._duration(eta) if eta is not None else "计算中",
        }
        titles = {
            "total": "总数",
            "success": "成功",
            "failed": "失败",
            "skipped": "跳过",
            "stopped": "停止",
            "pending": "待处理",
            "elapsed": "已用时",
            "average": "平均",
            "eta": "预计剩余",
        }
        for key, value in values.items():
            self.stats_labels[key].setText(f"{titles[key]}：{value}")
        self.state_label.setText(
            {
                BatchStatus.CREATED: "等待开始",
                BatchStatus.RUNNING: "批量推理中",
                BatchStatus.PAUSED: "已暂停",
                BatchStatus.STOPPING: "正在停止",
                BatchStatus.STOPPED: "已停止",
                BatchStatus.COMPLETED: "已完成",
                BatchStatus.COMPLETED_WITH_ERRORS: "完成（有失败）",
                BatchStatus.FAILED: "任务失败",
            }[task.status]
        )
        self._summary_dirty = False
        self.summary_refresh_count += 1
        self._record_ui_refresh(started)

    def refresh_statistics(self) -> None:
        started = perf_counter()
        if self.task is None:
            return
        stats = BatchPredictionService.statistics(self.task)
        self.analysis_metrics["total"].setText(f"总文件数\n{stats['total']}")
        self.analysis_metrics["success"].setText(f"成功数\n{stats['success']}")
        self.analysis_metrics["failed"].setText(f"失败数\n{stats['failed']}")
        self.analysis_metrics["low"].setText(f"低置信度\n{stats['low_confidence']}")
        self.analysis_metrics["elapsed"].setText(
            f"总耗时\n{self._duration(stats['elapsed_seconds'])}"
        )
        average = stats["average_item_seconds"]
        self.analysis_metrics["average"].setText(
            f"平均耗时\n{self._duration(average) if average is not None else '—'}"
        )
        self._set_chart("combination", stats["combinations"])
        self._set_chart("source", stats["sources"])
        self._set_chart("confidence", stats["confidence_buckets"])
        self._set_chart("error", stats["errors"])
        self._statistics_dirty = False
        self._last_chart_refresh = perf_counter()
        self.chart_refresh_count += 1
        self._record_ui_refresh(started)

    def show_results_tab(self) -> None:
        """Switch to results after a completed, stopped or historical batch."""
        self.tabs.setCurrentIndex(1)
        self.result_model.refresh()
        self._rebuild_result_filter_options()
        self._apply_result_filters()

    def show_scan_summary(self, summary: BatchScanSummary) -> None:
        self.show_feedback(
            f"扫描 {summary.scanned_count} 项，新增 {summary.added_count} 个 CSV，"
            f"忽略重复 {summary.duplicate_count} 个，无效 {summary.invalid_count} 个。"
        )
        self.refresh_table()

    def show_export_result(self, result: Any) -> None:
        self.show_feedback(f"结果已导出：{result.output_directory}")
        self.refresh_summary()

    def show_filtered_export_result(self, path: Path) -> None:
        self.show_feedback(f"当前筛选结果已导出：{path}")

    def show_history(self, task: BatchPredictionTask) -> None:
        self.set_task(task)
        self.current_file_label.setText(f"历史批次：{task.name}")
        self.show_feedback(f"已打开历史结果：{task.output_directory}")
        self.show_results_tab()

    def show_result_preview(self, preview: SignalPreview) -> None:
        if (
            self._selected_result_item is None
            or preview.source_path != self._selected_result_item.file_path
        ):
            return
        self.result_waveform.clear()
        self.result_waveform.plot(
            list(preview.display_values),
            pen=pg.mkPen(color="#2463a8", width=1.1),
        )
        self.waveform_status.setText(
            f"已按需读取 {preview.original_point_count:,} 点；显示抽样 "
            f"{len(preview.display_values):,} 点。"
        )

    def show_result_preview_error(self, message: str) -> None:
        self.result_waveform.clear()
        self.waveform_status.setText(f"波形读取失败：{message}")

    def show_pause_requested(self) -> None:
        self.state_label.setText("正在暂停")
        self.show_feedback("当前文件完成后暂停，不会启动下一个文件。")

    def show_feedback(self, message: str, *, error: bool = False) -> None:
        self.feedback_label.setText(message)
        self.feedback_label.setProperty("error", error)
        self.feedback_label.style().unpolish(self.feedback_label)
        self.feedback_label.style().polish(self.feedback_label)
        self.feedback_label.show()

    def apply_statistics_filter(self, kind: str, value: str) -> None:
        """Public chart-link contract used by plot clicks and tests."""
        self.tabs.setCurrentIndex(1)
        if kind == "combination":
            index = self.combination_filter.findData(value)
            self.combination_filter.setCurrentIndex(max(0, index))
        elif kind == "source":
            for row in range(self.source_filter.count()):
                item = self.source_filter.item(row)
                item.setCheckState(
                    Qt.CheckState.Checked if item.text() == value else Qt.CheckState.Unchecked
                )
        elif kind == "confidence":
            minimum, maximum = CONFIDENCE_RANGES[value]
            self.confidence_min.setValue(minimum)
            self.confidence_max.setValue(maximum)
        elif kind == "error":
            self.error_only_checkbox.setChecked(True)
            self.result_proxy.set_filters(error_type=value)
        self._apply_result_filters()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.paths_added.emit(paths, self.recursive_checkbox.isChecked())
            event.acceptProposedAction()

    def _apply_result_filters(self, *_: Any) -> None:
        sources = {
            self.source_filter.item(row).text()
            for row in range(self.source_filter.count())
            if self.source_filter.item(row).checkState() == Qt.CheckState.Checked
        }
        self.result_proxy.set_filters(
            keyword=self.result_search.text(),
            status=self.result_status_filter.currentData(),
            combination=str(self.combination_filter.currentData() or ""),
            sources=sources,
            confidence_min=self.confidence_min.value(),
            confidence_max=self.confidence_max.value(),
            low_confidence_only=self.low_confidence_checkbox.isChecked(),
            errors_only=self.error_only_checkbox.isChecked(),
            error_type=(
                self.result_proxy.error_type if self.error_only_checkbox.isChecked() else ""
            ),
        )
        self._update_filtered_count()

    def _update_filtered_count(self) -> None:
        count = self.result_proxy.rowCount()
        total = self.result_model.rowCount()
        self.filtered_count_label.setText(f"{count:,} / {total:,} 条结果")
        self.export_filtered_button.setEnabled(count > 0)
        self._update_result_actions()

    def _rebuild_result_filter_options(self) -> None:
        task = self.task
        combinations = (
            sorted(
                {item.predicted_combination for item in task.items if item.predicted_combination}
            )
            if task
            else []
        )
        self._known_combinations = set(combinations)
        selected_combination = self.combination_filter.currentData()
        self.combination_filter.blockSignals(True)
        self.combination_filter.clear()
        self.combination_filter.addItem("全部预测组合", "")
        for combination in combinations:
            self.combination_filter.addItem(combination, combination)
        index = self.combination_filter.findData(selected_combination)
        self.combination_filter.setCurrentIndex(max(0, index))
        self.combination_filter.blockSignals(False)

        selected_sources = {
            self.source_filter.item(row).text()
            for row in range(self.source_filter.count())
            if self.source_filter.item(row).checkState() == Qt.CheckState.Checked
        }
        sources = sorted(
            {source for item in (task.items if task else []) for source in item.predicted_sources}
        )
        self._known_sources = set(sources)
        self.source_filter.blockSignals(True)
        self.source_filter.clear()
        for source in sources:
            item = QListWidgetItem(source)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if source in selected_sources else Qt.CheckState.Unchecked
            )
            self.source_filter.addItem(item)
        self.source_filter.blockSignals(False)

    def _show_selected_result_detail(self, *_: Any) -> None:
        indexes = self.result_table.selectionModel().selectedRows()
        if not indexes:
            self._selected_result_item = None
            self.result_metadata.clear()
            self.result_waveform.clear()
            self._clear_probability_bars()
            self._update_result_actions()
            return
        source_index = self.result_proxy.mapToSource(indexes[0])
        item = self.result_model.item_at(source_index.row())
        if item is None:
            return
        self._selected_result_item = item
        payload = item.result or {}
        candidates = combination_candidates(payload)
        metadata = {
            "file": {
                "name": item.file_name,
                "path": str(item.file_path),
                "size": item.file_size,
                "modified_at": item.modified_at,
                "status": item.status.value,
            },
            "model": {
                "name": self.task.model_name if self.task else "",
                "version": self.task.model_version if self.task else "",
                "runtime_version": self.task.runtime_version if self.task else "",
                "device": self.task.device if self.task else "",
            },
            "prediction": {
                "decision_mode": item.decision_mode,
                "input_shape": payload.get("input_shape", []),
                "predicted_combination": item.predicted_combination,
                "predicted_sources": item.predicted_sources,
                "decoded_label_vector": item.decoded_label_vector,
                "combination_probabilities": payload.get("combination_probabilities"),
                "first_candidate": candidates[0] if candidates else None,
                "second_candidate": candidates[1] if len(candidates) > 1 else None,
                "candidate_probability_margin": candidate_probability_margin(payload),
                "multilabel_probabilities": payload.get("multilabel_probabilities", []),
                "thresholds": payload.get("thresholds", []),
                "thresholds_applicable": payload.get("thresholds_applicable"),
                "elapsed_ms": item.elapsed_ms,
                "low_confidence": is_low_confidence(payload),
            },
            "error": {
                "type": item.error_type,
                "message": item.error_message,
            },
        }
        self.result_metadata.setPlainText(json.dumps(metadata, ensure_ascii=False, indent=2))
        self._set_probability_bars(item.labels, item.display_probabilities)
        self.result_waveform.clear()
        if item.file_path.is_file():
            self.waveform_status.setText("正在后台按需读取原始 CSV 波形…")
            self.waveform_requested.emit(item.file_path)
        else:
            self.waveform_status.setText("原始 CSV 已不可用；历史预测结果仍可查看。")
        self._update_result_actions()

    def _set_probability_bars(self, labels: list[str], probabilities: list[float]) -> None:
        self._clear_probability_bars()
        for label, probability in zip(labels, probabilities, strict=False):
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            name = QLabel(str(label))
            name.setMinimumWidth(110)
            bar = QProgressBar()
            bar.setRange(0, 10000)
            bar.setValue(round(float(probability) * 10000))
            bar.setFormat(f"{float(probability) * 100:.2f}%")
            layout.addWidget(name)
            layout.addWidget(bar, 1)
            self.probability_layout.addWidget(row)
        self.probability_layout.addStretch()

    def _clear_probability_bars(self) -> None:
        while self.probability_layout.count():
            item = self.probability_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _set_chart(self, key: str, distribution: dict[str, int]) -> None:
        plot = self.chart_plots[key]
        plot.clear()
        labels = list(distribution)
        values = [distribution[label] for label in labels]
        if not labels:
            plot.setTitle("暂无数据", color="#7a8493", size="10pt")
            plot._result_labels = []  # type: ignore[attr-defined]
            return
        plot.setTitle("")
        positions = list(range(len(labels)))
        bars = pg.BarGraphItem(x=positions, height=values, width=0.65, brush="#3f78b6")
        plot.addItem(bars)
        plot.getAxis("bottom").setTicks([list(zip(positions, labels, strict=True))])
        plot.setYRange(0, max(values) * 1.2 + 1)
        plot._result_labels = labels  # type: ignore[attr-defined]
        if not getattr(plot, "_result_click_connected", False):
            plot.scene().sigMouseClicked.connect(
                lambda event, chart=key, target=plot: self._chart_clicked(chart, target, event)
            )
            plot._result_click_connected = True  # type: ignore[attr-defined]

    def _chart_clicked(self, key: str, plot: pg.PlotWidget, event: Any) -> None:
        labels = getattr(plot, "_result_labels", [])
        if not labels:
            return
        point = plot.plotItem.vb.mapSceneToView(event.scenePos())
        index = round(point.x())
        if 0 <= index < len(labels):
            self.apply_statistics_filter(key, labels[index])

    def _compare_selected_results(self) -> None:
        items = self._selected_result_items()
        if not 2 <= len(items) <= 5 or any(
            item.status != BatchItemStatus.SUCCESS for item in items
        ):
            self.show_feedback("请选择 2～5 条成功结果进行对比。", error=True)
            return
        labels = sorted({label for item in items for label in item.labels})
        rows = [
            ("文件名", lambda item: item.file_name),
            ("预测组合", lambda item: item.predicted_combination),
            ("识别噪声源", lambda item: ", ".join(item.predicted_sources)),
            *[
                (
                    f"标签概率：{label}",
                    lambda item, target=label: self._label_probability(item, target),
                )
                for label in labels
            ],
            (
                "最高组合概率",
                lambda item: self._percent(highest_combination_probability(item.result or {})),
            ),
            (
                "第二组合概率",
                lambda item: self._percent(
                    combination_candidates(item.result or {})[1][1]
                    if len(combination_candidates(item.result or {})) > 1
                    else None
                ),
            ),
            (
                "候选概率差",
                lambda item: self._percent(candidate_probability_margin(item.result or {})),
            ),
            (
                "推理耗时",
                lambda item: f"{item.elapsed_ms:.1f} ms" if item.elapsed_ms is not None else "—",
            ),
        ]
        dialog = QDialog(self)
        dialog.setWindowTitle("批量结果对比")
        dialog.resize(900, 520)
        layout = QVBoxLayout(dialog)
        table = QTableWidget(len(rows), len(items), dialog)
        table.setHorizontalHeaderLabels([item.file_name for item in items])
        table.setVerticalHeaderLabels([title for title, _ in rows])
        for row_index, (_, getter) in enumerate(rows):
            for column, item in enumerate(items):
                table.setItem(row_index, column, QTableWidgetItem(str(getter(item))))
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.last_comparison_dialog = dialog
        dialog.show()

    def _selected_result_items(self) -> list[BatchFileItem]:
        items: list[BatchFileItem] = []
        for index in self.result_table.selectionModel().selectedRows():
            source = self.result_proxy.mapToSource(index)
            item = self.result_model.item_at(source.row())
            if item is not None:
                items.append(item)
        return items

    def _choose_filtered_export(self) -> None:
        if self.task is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出当前筛选结果", "filtered_predictions.csv", "CSV 文件 (*.csv)"
        )
        if path:
            self.filtered_export_requested.emit(Path(path), self.result_proxy.filtered_items())

    def _choose_history(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "打开历史批量结果目录")
        if path:
            self.history_requested.emit(Path(path))

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
        selected: list[str] = []
        for proxy_index in self.task_table.selectionModel().selectedRows():
            source_index = self.task_proxy.mapToSource(proxy_index)
            item = self.task_model.item_at(source_index.row())
            if item is not None:
                selected.append(item.item_id)
        return selected

    def _selected_task_item(self) -> BatchFileItem | None:
        if self.task is None:
            return None
        selected = self._selected_ids()
        return next(
            (item for item in self.task.items if selected and item.item_id == selected[0]),
            None,
        )

    def _show_task_detail(self) -> None:
        item = self._selected_task_item()
        if item is None:
            self.detail_view.clear()
            self._update_actions()
            return
        payload = {
            "task_id": self.task.task_id if self.task else "",
            "model_name": self.task.model_name if self.task else "",
            "model_version": self.task.model_version if self.task else "",
            "runtime_version": self.task.runtime_version if self.task else "",
            "device": self.task.device if self.task else "",
            "sequence": item.sequence,
            "file_path": str(item.file_path),
            "status": item.status.value,
            "status_message": item.status_message,
            "elapsed_ms": item.elapsed_ms,
            "predicted_combination": item.predicted_combination,
            "predicted_sources": item.predicted_sources,
            "error_type": item.error_type,
            "error_message": item.error_message,
            "retry_count": item.retry_count,
            "runtime_result": item.result,
        }
        self.detail_view.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))
        self._update_actions()

    def _apply_task_filter(self, *_: Any) -> None:
        self.task_proxy.set_filters(
            self.search_edit.text(),
            self.status_filter.currentData(),
        )

    def _sort_task_table(self, column: int) -> None:
        order = (
            Qt.SortOrder.DescendingOrder
            if self._task_sort_column == column
            and self.task_table.horizontalHeader().sortIndicatorOrder()
            == Qt.SortOrder.AscendingOrder
            else Qt.SortOrder.AscendingOrder
        )
        self._task_sort_column = column
        self.task_table.horizontalHeader().setSortIndicator(column, order)
        self.task_proxy.sort(column, order)

    def _sort_result_table(self, column: int) -> None:
        order = (
            Qt.SortOrder.DescendingOrder
            if self._result_sort_column == column
            and self.result_table.horizontalHeader().sortIndicatorOrder()
            == Qt.SortOrder.AscendingOrder
            else Qt.SortOrder.AscendingOrder
        )
        self._result_sort_column = column
        self.result_table.horizontalHeader().setSortIndicator(column, order)
        self.result_proxy.sort(column, order)

    def _remove_selected(self) -> None:
        if selected := self._selected_ids():
            self.remove_requested.emit(selected)

    def _retry_selected(self) -> None:
        if selected := self._selected_ids():
            self.retry_requested.emit(selected)

    def _open_selected_folder(self) -> None:
        item = self._selected_task_item()
        if item is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(item.file_path.parent)))

    def _update_actions(self) -> None:
        task = self.task
        running = task is not None and task.status in RUNNING_BATCH_STATUSES
        paused = task is not None and task.status == BatchStatus.PAUSED
        editable = not running
        has_items = bool(task and task.items)
        has_failed = bool(task and task.failed_count)
        selected_ids = set(self._selected_ids()) if task is not None else set()
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
        self.history_button.setEnabled(editable)
        self.remove_button.setEnabled(editable and bool(selected_ids))
        self.clear_button.setEnabled(editable and has_items)
        self.deduplicate_button.setEnabled(editable and has_items)
        self.start_button.setEnabled(editable and startable and self.model_available)
        self.pause_button.setEnabled(running and not paused)
        self.resume_button.setEnabled(paused)
        self.stop_button.setEnabled(running)
        self.retry_selected_button.setEnabled(editable and selected_failed)
        self.retry_all_button.setEnabled(editable and has_failed)
        self.export_button.setEnabled(bool(task and task.completed_count and not running))
        self.open_folder_button.setEnabled(bool(selected_ids))
        self.open_logs_button.setEnabled(bool(selected_ids))
        self._update_result_actions()

    def _update_result_actions(self) -> None:
        selected = self._selected_result_items() if hasattr(self, "result_table") else []
        self.compare_button.setEnabled(
            2 <= len(selected) <= 5
            and all(item.status == BatchItemStatus.SUCCESS for item in selected)
        )

    def _button(
        self,
        text: str,
        icon: QStyle.StandardPixmap | None = None,
        callback: Callable[..., Any] | None = None,
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

    @staticmethod
    def _label_probability(item: BatchFileItem, label: str) -> str:
        try:
            index = item.labels.index(label)
            return f"{item.display_probabilities[index] * 100:.2f}%"
        except (ValueError, IndexError):
            return "—"

    @staticmethod
    def _percent(value: float | None) -> str:
        return f"{value * 100:.2f}%" if value is not None else "—"

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
