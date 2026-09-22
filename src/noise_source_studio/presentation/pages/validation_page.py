"""Complete manifest-driven model validation center."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import QSortFilterProxyModel, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from noise_source_studio.domain.models import LoadedModel, SignalPreview
from noise_source_studio.domain.validation import (
    RUNNING_VALIDATION_STATUSES,
    ManifestValidationReport,
    ValidationSample,
    ValidationStatus,
    ValidationTask,
)
from noise_source_studio.presentation.models import (
    ConfusionMatrixTableModel,
    DictTableModel,
    ManifestPreviewTableModel,
    ValidationSampleFilterProxyModel,
    ValidationSampleTableModel,
)
from noise_source_studio.presentation.widgets import (
    PAGE_CONTENT_MARGINS,
    PAGE_CONTENT_SPACING,
    MetricCard,
    PageHeader,
    SectionCard,
)

LABEL_COLUMNS = [
    ("label", "标签"),
    ("tp", "TP"),
    ("fp", "FP"),
    ("tn", "TN"),
    ("fn", "FN"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("specificity", "Specificity"),
    ("false_positive_rate", "FPR"),
    ("false_negative_rate", "FNR"),
    ("support", "Support"),
    ("predicted_positive_count", "预测阳性数"),
]
COMBINATION_COLUMNS = [
    ("true_combination", "真实组合"),
    ("support", "样本数"),
    ("exact_count", "完全正确数"),
    ("exact_accuracy", "完全正确率"),
    ("most_common_mistake", "最常见误判"),
    ("most_common_mistake_count", "误判次数"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("average_confidence", "平均最高概率"),
    ("average_confidence_margin", "平均置信度差"),
]
GROUP_COLUMNS = [
    ("value", "分组值"),
    ("sample_count", "样本数"),
    ("exact_match", "Exact Match"),
    ("micro_f1", "Micro F1"),
    ("macro_f1", "Macro F1"),
    ("overprediction_rate", "过预测率"),
    ("underprediction_rate", "欠预测率"),
    ("inference_failed_count", "推理失败"),
    ("average_confidence", "平均置信度"),
    ("average_elapsed_ms", "平均耗时"),
]


class ValidationPage(QWidget):
    """Six-tab validation workspace with linked error analysis."""

    manifest_check_requested = Signal(object, object, str)
    start_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    stop_requested = Signal()
    history_requested = Signal(object)
    batch_reuse_requested = Signal(object)
    waveform_requested = Signal(object)
    export_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.loaded_model: LoadedModel | None = None
        self.report: ManifestValidationReport | None = None
        self.task: ValidationTask | None = None
        self._probability_widgets: list[QWidget] = []
        self._chart_values: dict[str, list[str]] = {}
        self._confidence_range = (0.0, 1.0)
        self._error_type_filter = ""
        self._group_values: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(*PAGE_CONTENT_MARGINS)
        layout.setSpacing(PAGE_CONTENT_SPACING)
        header = PageHeader(
            "模型验证",
            "导入带真实标签的清单，复用当前模型会话计算总体、标签、组合和分组指标。",
        )
        header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(header)
        toolbar = self._build_toolbar()
        toolbar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(toolbar)
        self.tabs = QTabWidget()
        self.tabs.setMinimumSize(0, 0)
        self.tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Ignored,
        )
        self.tabs.addTab(self._build_configuration_tab(), "验证配置")
        self.tabs.addTab(self._build_overall_tab(), "总体结果")
        self.tabs.addTab(self._build_label_tab(), "标签分析")
        self.tabs.addTab(self._build_combination_tab(), "组合分析")
        self.tabs.addTab(self._build_group_tab(), "分组分析")
        self.tabs.addTab(self._build_samples_tab(), "样本明细")
        layout.addWidget(self.tabs, 1)
        self._update_buttons()

    def _build_toolbar(self) -> QWidget:
        card = SectionCard("验证任务")
        row = QHBoxLayout()
        self.current_model_label = QLabel("当前模型：未配置")
        self.current_model_label.setObjectName("sectionDescription")
        self.open_history_button = QPushButton("打开历史验证结果")
        self.open_history_button.clicked.connect(self._choose_history)
        self.reuse_batch_button = QPushButton("使用已有批量结果验证")
        self.reuse_batch_button.clicked.connect(self._choose_batch_result)
        self.export_button = QPushButton("导出验证报告副本")
        self.export_button.clicked.connect(self._choose_export)
        self.open_output_button = QPushButton("打开输出目录")
        self.open_output_button.clicked.connect(self._open_output)
        row.addWidget(self.current_model_label)
        row.addStretch()
        row.addWidget(self.open_history_button)
        row.addWidget(self.reuse_batch_button)
        row.addWidget(self.export_button)
        row.addWidget(self.open_output_button)
        card.content_layout.addLayout(row)
        return card

    def _build_configuration_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(12)

        config = SectionCard("验证清单")
        form = QGridLayout()
        self.manifest_edit = QLineEdit()
        self.manifest_edit.setPlaceholderText("选择或拖入 validation_manifest.csv")
        self.manifest_edit.textChanged.connect(self._manifest_changed)
        self.manifest_button = QPushButton("选择清单")
        self.manifest_button.clicked.connect(self._choose_manifest)
        self.data_root_edit = QLineEdit()
        self.data_root_edit.setPlaceholderText("可选；默认使用 manifest 所在目录")
        self.root_button = QPushButton("选择数据根目录")
        self.root_button.clicked.connect(self._choose_data_root)
        self.missing_policy_combo = QComboBox()
        self.missing_policy_combo.addItem("缺失文件时停止", "stop")
        self.missing_policy_combo.addItem("跳过缺失文件并记录", "skip")
        self.check_button = QPushButton("执行数据检查")
        self.check_button.clicked.connect(self._request_manifest_check)
        self.start_button = QPushButton("开始验证")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.start_requested)
        self.pause_button = QPushButton("暂停")
        self.pause_button.clicked.connect(self.pause_requested)
        self.resume_button = QPushButton("继续")
        self.resume_button.clicked.connect(self.resume_requested)
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_requested)
        form.addWidget(QLabel("验证清单"), 0, 0)
        form.addWidget(self.manifest_edit, 0, 1, 1, 4)
        form.addWidget(self.manifest_button, 0, 5)
        form.addWidget(QLabel("数据根目录"), 1, 0)
        form.addWidget(self.data_root_edit, 1, 1, 1, 4)
        form.addWidget(self.root_button, 1, 5)
        form.addWidget(QLabel("缺失策略"), 2, 0)
        form.addWidget(self.missing_policy_combo, 2, 1)
        form.addWidget(self.check_button, 2, 2)
        form.addWidget(self.start_button, 2, 3)
        form.addWidget(self.pause_button, 2, 4)
        controls = QHBoxLayout()
        controls.addWidget(self.resume_button)
        controls.addWidget(self.stop_button)
        form.addLayout(controls, 2, 5)
        config.content_layout.addLayout(form)
        self.label_order_label = QLabel("模型标签顺序：—")
        self.label_order_label.setWordWrap(True)
        config.content_layout.addWidget(self.label_order_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress_label = QLabel("尚未执行验证")
        progress_row = QHBoxLayout()
        progress_row.addWidget(self.progress, 1)
        progress_row.addWidget(self.progress_label)
        config.content_layout.addLayout(progress_row)
        layout.addWidget(config)

        metrics = QGridLayout()
        names = (
            ("total", "样本总数"),
            ("combinations", "组合数量"),
            ("missing", "缺失文件"),
            ("duplicates", "重复文件"),
            ("invalid", "无效标签"),
            ("metadata", "元数据字段"),
        )
        self.manifest_cards: dict[str, MetricCard] = {}
        for index, (key, title) in enumerate(names):
            card = MetricCard(title, "—")
            self.manifest_cards[key] = card
            metrics.addWidget(card, index // 3, index % 3)
        layout.addLayout(metrics)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.preview_model = ManifestPreviewTableModel()
        self.preview_table = self._table(self.preview_model)
        preview_card = SectionCard("清单预览（前 100 条）")
        preview_card.content_layout.addWidget(self.preview_table)
        report_card = SectionCard("数据检查报告")
        self.check_report = QPlainTextEdit()
        self.check_report.setReadOnly(True)
        report_card.content_layout.addWidget(self.check_report)
        splitter.addWidget(preview_card)
        splitter.addWidget(report_card)
        splitter.setSizes([720, 420])
        layout.addWidget(splitter, 1)
        return content

    def _build_overall_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 12, 0, 12)
        layout.setSpacing(12)
        card_grid = QGridLayout()
        definitions = (
            ("valid_sample_count", "有效样本数"),
            ("inference_success_count", "推理成功数"),
            ("inference_failed_count", "推理失败数"),
            ("exact_match_accuracy", "Exact Match"),
            ("micro_precision", "Micro Precision"),
            ("micro_recall", "Micro Recall"),
            ("micro_f1", "Micro F1"),
            ("macro_precision", "Macro Precision"),
            ("macro_recall", "Macro Recall"),
            ("macro_f1", "Macro F1"),
            ("weighted_f1", "Weighted F1"),
            ("hamming_loss", "Hamming Loss"),
            ("average_wrong_labels_per_sample", "平均错误标签数"),
            ("source_count_accuracy", "Source Count Accuracy"),
            ("overprediction_rate", "过预测率"),
            ("underprediction_rate", "欠预测率"),
            ("inference_failure_rate", "推理失败率"),
            ("low_confidence_rate", "低置信度比例"),
            ("average_inference_ms", "平均推理耗时"),
            ("p95_inference_ms", "P95 推理耗时"),
        )
        self.overall_cards: dict[str, MetricCard] = {}
        for index, (key, title) in enumerate(definitions):
            card = MetricCard(title, "—")
            self.overall_cards[key] = card
            card_grid.addWidget(card, index // 4, index % 4)
        layout.addLayout(card_grid)
        chart_grid = QGridLayout()
        self.charts: dict[str, pg.PlotWidget] = {}
        chart_specs = (
            ("label_f1", "各标签 F1"),
            ("label_fpr", "各标签 FPR"),
            ("combination", "组合准确率"),
            ("confidence", "置信度分布"),
            ("errors", "错误类型分布"),
        )
        for index, (key, title) in enumerate(chart_specs):
            chart = pg.PlotWidget()
            chart.setMinimumHeight(220)
            chart.setBackground("w")
            chart.setTitle(title)
            chart.showGrid(y=True, alpha=0.2)
            chart.scene().sigMouseClicked.connect(
                lambda event, chart_key=key: self._chart_clicked(chart_key, event)
            )
            self.charts[key] = chart
            chart_grid.addWidget(chart, index // 2, index % 2)
        layout.addLayout(chart_grid)
        scroll.setWidget(content)
        return scroll

    def _build_label_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 12, 0, 0)
        controls = QHBoxLayout()
        self.label_filter_combo = QComboBox()
        self.label_relation_combo = QComboBox()
        for text, value in (
            ("该标签全部样本", "all"),
            ("假阳性", "false_positive"),
            ("假阴性", "false_negative"),
            ("预测正确", "correct"),
        ):
            self.label_relation_combo.addItem(text, value)
        apply_button = QPushButton("在样本明细中查看")
        apply_button.clicked.connect(self._apply_label_filter)
        controls.addWidget(QLabel("标签"))
        controls.addWidget(self.label_filter_combo)
        controls.addWidget(self.label_relation_combo)
        controls.addWidget(apply_button)
        controls.addStretch()
        layout.addLayout(controls)
        self.label_model = DictTableModel(LABEL_COLUMNS)
        self.label_proxy = QSortFilterProxyModel()
        self.label_proxy.setSourceModel(self.label_model)
        self.label_proxy.setSortRole(DictTableModel.SortRole)
        self.label_table = self._table(self.label_proxy)
        self.label_table.clicked.connect(self._label_row_clicked)
        layout.addWidget(self.label_table, 1)
        return content

    def _build_combination_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 12, 0, 0)
        splitter = QSplitter(Qt.Orientation.Vertical)
        metrics_card = SectionCard("组合指标")
        self.top_k_label = QLabel("Top-K 组合指标：尚无结果")
        self.top_k_label.setWordWrap(True)
        metrics_card.content_layout.addWidget(self.top_k_label)
        self.combination_model = DictTableModel(COMBINATION_COLUMNS)
        self.combination_proxy = QSortFilterProxyModel()
        self.combination_proxy.setSourceModel(self.combination_model)
        self.combination_proxy.setSortRole(DictTableModel.SortRole)
        self.combination_table = self._table(self.combination_proxy)
        self.combination_table.clicked.connect(self._combination_row_clicked)
        metrics_card.content_layout.addWidget(self.combination_table)
        matrix_card = SectionCard("组合混淆矩阵（纵轴真实，横轴预测）")
        matrix_controls = QHBoxLayout()
        self.matrix_percent_checkbox = QCheckBox("显示行百分比")
        self.matrix_percent_checkbox.toggled.connect(self._toggle_matrix_percent)
        matrix_controls.addWidget(self.matrix_percent_checkbox)
        matrix_controls.addStretch()
        matrix_card.content_layout.addLayout(matrix_controls)
        self.matrix_model = ConfusionMatrixTableModel()
        self.matrix_table = self._table(self.matrix_model)
        self.matrix_table.verticalHeader().setVisible(True)
        self.matrix_table.horizontalHeader().setStretchLastSection(False)
        self.matrix_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.matrix_table.clicked.connect(self._matrix_clicked)
        matrix_card.content_layout.addWidget(self.matrix_table)
        splitter.addWidget(metrics_card)
        splitter.addWidget(matrix_card)
        splitter.setSizes([340, 360])
        layout.addWidget(splitter, 1)
        return content

    def _build_group_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 12, 0, 0)
        controls = QHBoxLayout()
        self.group_field_combo = QComboBox()
        self.group_field_combo.currentIndexChanged.connect(self._refresh_group_view)
        controls.addWidget(QLabel("分组字段"))
        controls.addWidget(self.group_field_combo)
        controls.addStretch()
        layout.addLayout(controls)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.group_model = DictTableModel(GROUP_COLUMNS)
        self.group_proxy = QSortFilterProxyModel()
        self.group_proxy.setSourceModel(self.group_model)
        self.group_proxy.setSortRole(DictTableModel.SortRole)
        self.group_table = self._table(self.group_proxy)
        self.group_table.clicked.connect(self._group_row_clicked)
        self.group_chart = pg.PlotWidget()
        self.group_chart.setBackground("w")
        self.group_chart.setTitle("分组 Exact Match")
        self.group_chart.showGrid(y=True, alpha=0.2)
        self.group_chart.scene().sigMouseClicked.connect(self._group_chart_clicked)
        splitter.addWidget(self.group_table)
        splitter.addWidget(self.group_chart)
        splitter.setSizes([760, 430])
        layout.addWidget(splitter, 1)
        return content

    def _build_samples_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(8)
        filters = QGridLayout()
        self.sample_search = QLineEdit()
        self.sample_search.setPlaceholderText("搜索文件名或路径")
        self.sample_outcome_combo = QComboBox()
        for text, value in (
            ("全部样本", "all"),
            ("预测正确", "correct"),
            ("预测错误", "wrong"),
            ("推理失败", "failed"),
            ("低置信度", "low_confidence"),
            ("过预测", "overprediction"),
            ("欠预测", "underprediction"),
            ("Source Count 错误", "source_count_error"),
        ):
            self.sample_outcome_combo.addItem(text, value)
        self.true_combo_filter = QComboBox()
        self.predicted_combo_filter = QComboBox()
        self.sample_label_filter = QComboBox()
        self.sample_label_relation = QComboBox()
        for text, value in (
            ("标签全部样本", "all"),
            ("标签假阳性", "false_positive"),
            ("标签假阴性", "false_negative"),
            ("标签预测正确", "correct"),
        ):
            self.sample_label_relation.addItem(text, value)
        self.metadata_field_filter = QComboBox()
        self.metadata_value_filter = QLineEdit()
        self.metadata_value_filter.setPlaceholderText("元数据值")
        self.filtered_count_label = QLabel("0 / 0 条")
        filters.addWidget(self.sample_search, 0, 0, 1, 2)
        filters.addWidget(self.sample_outcome_combo, 0, 2)
        filters.addWidget(self.true_combo_filter, 0, 3)
        filters.addWidget(self.predicted_combo_filter, 0, 4)
        filters.addWidget(self.filtered_count_label, 0, 5)
        filters.addWidget(self.sample_label_filter, 1, 0)
        filters.addWidget(self.sample_label_relation, 1, 1)
        filters.addWidget(self.metadata_field_filter, 1, 2)
        filters.addWidget(self.metadata_value_filter, 1, 3, 1, 2)
        clear_button = QPushButton("清除筛选")
        clear_button.clicked.connect(self.clear_sample_filters)
        filters.addWidget(clear_button, 1, 5)
        layout.addLayout(filters)
        for widget in (
            self.sample_search,
            self.sample_outcome_combo,
            self.true_combo_filter,
            self.predicted_combo_filter,
            self.sample_label_filter,
            self.sample_label_relation,
            self.metadata_field_filter,
            self.metadata_value_filter,
        ):
            if isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._apply_sample_filters)
            else:
                widget.currentIndexChanged.connect(self._apply_sample_filters)

        quick = QHBoxLayout()
        for text, outcome, order_column, descending in (
            ("双源预测成三源", "double_to_triple", -1, False),
            ("三源预测成双源", "triple_to_double", -1, False),
            ("单源预测成多源", "single_to_multi", -1, False),
            ("最高置信度误判", "wrong", 6, True),
            ("最低置信度样本", "all", 6, False),
            ("候选差距最小", "all", 9, False),
        ):
            button = QPushButton(text)
            button.clicked.connect(
                lambda _checked=False, value=outcome, column=order_column, desc=descending: (
                    self._quick_filter(value, column, desc)
                )
            )
            quick.addWidget(button)
        most_fp_button = QPushButton("假阳性最多标签")
        most_fp_button.clicked.connect(lambda: self._most_error_label("fp"))
        most_fn_button = QPushButton("假阴性最多标签")
        most_fn_button.clicked.connect(lambda: self._most_error_label("fn"))
        quick.addWidget(most_fp_button)
        quick.addWidget(most_fn_button)
        quick.addStretch()
        layout.addLayout(quick)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.sample_model = ValidationSampleTableModel()
        self.sample_proxy = ValidationSampleFilterProxyModel()
        self.sample_proxy.setSourceModel(self.sample_model)
        self.sample_table = self._table(self.sample_proxy)
        self.sample_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sample_table.clicked.connect(self._sample_selected)
        splitter.addWidget(self.sample_table)

        detail_scroll = QScrollArea()
        detail_scroll.setWidgetResizable(True)
        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(10, 6, 10, 6)
        title = QLabel("样本详情")
        title.setObjectName("sectionTitle")
        detail_layout.addWidget(title)
        self.sample_detail = QPlainTextEdit()
        self.sample_detail.setReadOnly(True)
        self.sample_detail.setMinimumHeight(190)
        detail_layout.addWidget(self.sample_detail)
        detail_layout.addSpacing(8)
        probability_title = QLabel("动态标签概率")
        probability_title.setObjectName("sectionTitle")
        probability_title.setMinimumHeight(24)
        detail_layout.addWidget(probability_title)
        self.probability_layout = QVBoxLayout()
        detail_layout.addLayout(self.probability_layout)
        detail_layout.addSpacing(8)
        waveform_title = QLabel("按需原始信号波形")
        waveform_title.setObjectName("sectionTitle")
        waveform_title.setMinimumHeight(24)
        detail_layout.addWidget(waveform_title)
        self.waveform = pg.PlotWidget()
        self.waveform.setBackground("w")
        self.waveform.setMinimumHeight(220)
        self.waveform.showGrid(x=True, y=True, alpha=0.2)
        detail_layout.addWidget(self.waveform)
        self.waveform_status = QLabel("选择样本后按需读取 CSV。")
        detail_layout.addWidget(self.waveform_status)
        detail_layout.addStretch()
        detail_scroll.setWidget(detail)
        splitter.addWidget(detail_scroll)
        splitter.setSizes([850, 430])
        layout.addWidget(splitter, 1)
        return content

    @staticmethod
    def _table(model: Any) -> QTableView:
        table = QTableView()
        table.setModel(model)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(True)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def set_model(self, model: LoadedModel | None) -> None:
        self.loaded_model = model
        if model is None:
            self.current_model_label.setText("当前模型：未配置")
            self.label_order_label.setText("模型标签顺序：—")
        else:
            self.current_model_label.setText(
                f"当前模型：{model.record.display_name} · {model.device} · {model.prediction_mode}"
            )
            self.label_order_label.setText("模型标签顺序：" + " → ".join(model.labels))
        self._update_buttons()

    def show_manifest_report(self, report: ManifestValidationReport) -> None:
        self.report = report
        self.task = None
        self.preview_model.set_report(report)
        self.manifest_cards["total"].set_status(str(report.source_row_count))
        self.manifest_cards["combinations"].set_status(str(len(report.combination_counts)))
        self.manifest_cards["missing"].set_status(str(report.missing_count))
        self.manifest_cards["duplicates"].set_status(str(report.duplicate_count))
        self.manifest_cards["invalid"].set_status(str(report.invalid_label_count))
        self.manifest_cards["metadata"].set_status(str(len(report.metadata_fields)))
        lines = [
            f"检查结论：{'通过' if report.can_start else '未通过'}",
            f"可执行样本：{report.runnable_count}",
            f"标签顺序：{', '.join(report.labels)}",
            f"组合分布：{json.dumps(report.combination_counts, ensure_ascii=False)}",
            f"标签阳性数：{json.dumps(report.label_positive_counts, ensure_ascii=False)}",
            "",
        ]
        lines.extend(
            f"[{issue.severity.upper()}] 行 {issue.row or '—'} "
            f"{issue.code}: {issue.message} {issue.file_path}"
            for issue in report.issues
        )
        if not report.issues:
            lines.append("未发现缺失、冲突或无效记录。")
        self.check_report.setPlainText("\n".join(lines))
        self.progress_label.setText("清单检查通过" if report.can_start else "清单检查未通过")
        self._update_buttons()

    def set_task(self, task: ValidationTask) -> None:
        self.task = task
        self.report = task.manifest_report or self.report
        self.sample_model.set_task(task)
        self._populate_result_filters()
        self.refresh_task()

    def refresh_task(self) -> None:
        if self.task is None:
            self._update_buttons()
            return
        self.task.refresh_counts()
        self.progress.setValue(round(self.task.progress * 10))
        self.progress_label.setText(
            f"{self.task.status.value} · {self.task.success_count} 成功 · "
            f"{self.task.inference_failed_count} 失败 · {self.task.skipped_count} 跳过"
        )
        self.sample_model.refresh()
        self._apply_sample_filters()
        if self.task.metrics:
            self._render_results()
        self._update_buttons()

    def refresh_summary(self) -> None:
        if self.task is None:
            return
        self.task.refresh_counts()
        self.progress.setValue(round(self.task.progress * 10))
        self.progress_label.setText(
            f"{self.task.status.value} · {self.task.success_count} 成功 · "
            f"{self.task.inference_failed_count} 失败 · {self.task.skipped_count} 跳过"
        )
        self.filtered_count_label.setText(
            f"{self.sample_proxy.rowCount()} / {len(self.task.samples)} 条"
        )
        self._update_buttons()

    def refresh_sample(self, sample: ValidationSample) -> None:
        self.sample_model.refresh_sample(sample)
        self.refresh_summary()

    def show_history(self, task: ValidationTask) -> None:
        self.manifest_edit.setText(str(task.manifest_path))
        self.data_root_edit.setText(str(task.data_root))
        self.set_task(task)
        self.tabs.setCurrentIndex(1)

    def show_validation_complete(self, task: ValidationTask) -> None:
        self.set_task(task)
        self.tabs.setCurrentIndex(1)

    def show_error(self, message: str) -> None:
        QMessageBox.warning(self, "模型验证", message)
        self.progress_label.setText(message)
        self._update_buttons()

    def show_export_result(self, result: Any) -> None:
        self.progress_label.setText(f"验证结果已保存：{result.output_directory}")
        self._update_buttons()

    def show_preview(self, preview: SignalPreview) -> None:
        self.waveform.clear()
        self.waveform.plot(list(preview.display_values), pen=pg.mkPen("#2f78b7", width=1.2))
        self.waveform_status.setText(
            f"{preview.source_path.name} · 原始 {preview.original_point_count} 点 · "
            f"显示 {len(preview.display_values)} 点"
        )

    def show_preview_error(self, message: str) -> None:
        self.waveform.clear()
        self.waveform_status.setText(f"波形读取失败：{message}")

    def apply_sample_filter(
        self,
        *,
        outcome: str = "all",
        true_combination: str = "",
        predicted_combination: str = "",
        label: str = "",
        label_relation: str = "",
        metadata_field: str = "",
        metadata_value: str = "",
        confidence_range: tuple[float, float] | None = None,
        error_type: str = "",
    ) -> None:
        self.clear_sample_filters()
        self._confidence_range = confidence_range or (0.0, 1.0)
        self._error_type_filter = error_type
        self.tabs.setCurrentIndex(5)
        self._set_combo_data(self.sample_outcome_combo, outcome)
        self._set_combo_text(self.true_combo_filter, true_combination)
        self._set_combo_text(self.predicted_combo_filter, predicted_combination)
        self._set_combo_text(self.sample_label_filter, label)
        self._set_combo_data(self.sample_label_relation, label_relation or "all")
        self._set_combo_text(self.metadata_field_filter, metadata_field)
        self.metadata_value_filter.setText(metadata_value)
        self._apply_sample_filters()

    def apply_chart_filter(self, kind: str, value: str) -> None:
        """Public chart-link API used by scene clicks and tests."""
        if kind == "label":
            self.tabs.setCurrentIndex(2)
            self._set_combo_text(self.label_filter_combo, value)
        elif kind == "combination":
            self.apply_sample_filter(true_combination=value)
        elif kind == "confidence":
            ranges = {
                "<40%": (0.0, 0.4),
                "40–60%": (0.4, 0.6),
                "60–80%": (0.6, 0.8),
                "≥80%": (0.8, 1.0),
            }
            self.apply_sample_filter(confidence_range=ranges.get(value, (0.0, 1.0)))
        elif kind == "error":
            self.apply_sample_filter(outcome="all", error_type=value)

    def clear_sample_filters(self) -> None:
        self.sample_search.clear()
        self.sample_outcome_combo.setCurrentIndex(0)
        self.true_combo_filter.setCurrentIndex(0)
        self.predicted_combo_filter.setCurrentIndex(0)
        self.sample_label_filter.setCurrentIndex(0)
        self.sample_label_relation.setCurrentIndex(0)
        self.metadata_field_filter.setCurrentIndex(0)
        self.metadata_value_filter.clear()
        self.sample_proxy.special = ""  # type: ignore[attr-defined]
        self._confidence_range = (0.0, 1.0)
        self._error_type_filter = ""
        self._apply_sample_filters()

    def _render_results(self) -> None:
        assert self.task is not None
        metrics = self.task.metrics
        overall = metrics.get("overall", {})
        for key, card in self.overall_cards.items():
            value = overall.get(key)
            if value is None:
                text = "—"
            elif key.endswith("_count"):
                text = str(int(value))
            elif key.endswith("_ms"):
                text = f"{float(value):.1f} ms"
            elif key == "average_wrong_labels_per_sample":
                text = f"{float(value):.3f}"
            else:
                text = f"{float(value) * 100:.2f}%"
            card.set_status(text)
        label_rows = list(metrics.get("labels", []))
        combination_rows = list(metrics.get("combinations", []))
        self.label_model.set_rows(label_rows)
        self.combination_model.set_rows(combination_rows)
        self.matrix_model.set_matrix(metrics.get("confusion_matrix", {}))
        top_k = metrics.get("top_k")
        if top_k:
            self.top_k_label.setText(
                "Top-K 组合指标："
                f"Top-1 {top_k['top_1_accuracy'] * 100:.2f}% · "
                f"Top-2 {top_k['top_2_accuracy'] * 100:.2f}% · "
                f"真实组合平均排名 {top_k['average_true_combination_rank']:.3f} · "
                f"平均概率 {top_k['average_true_combination_probability'] * 100:.2f}% · "
                f"NLL {top_k['negative_log_likelihood']:.4f}"
            )
        else:
            self.top_k_label.setText("Top-K 组合指标：不适用（当前模式没有合法组合概率）。")
        groups = metrics.get("groups", {})
        current = self.group_field_combo.currentText()
        self.group_field_combo.blockSignals(True)
        self.group_field_combo.clear()
        self.group_field_combo.addItems(groups.keys())
        self._set_combo_text(self.group_field_combo, current)
        self.group_field_combo.blockSignals(False)
        self._refresh_group_view()
        self._set_chart(
            "label_f1",
            [row["label"] for row in label_rows],
            [float(row.get("f1") or 0.0) for row in label_rows],
        )
        self._set_chart(
            "label_fpr",
            [row["label"] for row in label_rows],
            [float(row.get("false_positive_rate") or 0.0) for row in label_rows],
        )
        self._set_chart(
            "combination",
            [row["true_combination"] for row in combination_rows],
            [float(row.get("exact_accuracy") or 0.0) for row in combination_rows],
        )
        successful = [sample for sample in self.task.samples if sample.confidence is not None]
        bucket_counts: Counter[str] = Counter()
        for sample in successful:
            score = float(sample.confidence or 0.0)
            bucket_counts[
                "<40%"
                if score < 0.4
                else "40–60%"
                if score < 0.6
                else "60–80%"
                if score < 0.8
                else "≥80%"
            ] += 1
        self._set_chart(
            "confidence",
            ["<40%", "40–60%", "60–80%", "≥80%"],
            [bucket_counts[key] for key in ("<40%", "40–60%", "60–80%", "≥80%")],
        )
        error_counts = Counter(
            sample.error_type for sample in self.task.samples if sample.error_type
        )
        self._set_chart("errors", list(error_counts), list(error_counts.values()))

    def _set_chart(self, key: str, labels: list[str], values: list[float | int]) -> None:
        plot = self.charts[key]
        plot.clear()
        self._chart_values[key] = labels
        if not labels:
            plot.setTitle("暂无数据")
            return
        plot.setTitle("")
        x = list(range(len(labels)))
        plot.addItem(
            pg.BarGraphItem(
                x=x,
                height=[float(value) for value in values],
                width=0.72,
                brush="#3d7fb8",
            )
        )
        plot.getAxis("bottom").setTicks([list(zip(x, labels, strict=True))])

    def _chart_clicked(self, key: str, event: Any) -> None:
        labels = self._chart_values.get(key, [])
        if not labels:
            return
        point = self.charts[key].plotItem.vb.mapSceneToView(event.scenePos())
        index = round(point.x())
        if not 0 <= index < len(labels):
            return
        kind = "label" if key.startswith("label") else key
        self.apply_chart_filter(kind, labels[index])

    def _refresh_group_view(self) -> None:
        if self.task is None:
            return
        field = self.group_field_combo.currentText()
        rows = list(self.task.metrics.get("groups", {}).get(field, []))
        self.group_model.set_rows(rows)
        self.group_chart.clear()
        self._group_values = [str(row["value"]) for row in rows]
        if not rows:
            return
        x = list(range(len(rows)))
        self.group_chart.addItem(
            pg.BarGraphItem(
                x=x,
                height=[float(row.get("exact_match") or 0.0) for row in rows],
                width=0.7,
                brush="#3d7fb8",
            )
        )
        self.group_chart.getAxis("bottom").setTicks(
            [list(zip(x, [str(row["value"]) for row in rows], strict=True))]
        )

    def _populate_result_filters(self) -> None:
        if self.task is None:
            return
        combinations = sorted(
            {
                combination
                for sample in self.task.samples
                for combination in (sample.true_combination, sample.predicted_combination)
                if combination
            }
        )
        for combo, title in (
            (self.true_combo_filter, "全部真实组合"),
            (self.predicted_combo_filter, "全部预测组合"),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(title, "")
            combo.addItems(combinations)
            combo.blockSignals(False)
        for combo, title, values in (
            (self.sample_label_filter, "全部标签", self.task.labels),
            (self.label_filter_combo, "选择标签", self.task.labels),
            (self.metadata_field_filter, "全部元数据", self.task.metadata_fields),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(title, "")
            combo.addItems(values)
            combo.blockSignals(False)

    def _apply_sample_filters(self, *_: Any) -> None:
        self.sample_proxy.set_filters(
            keyword=self.sample_search.text(),
            outcome=str(self.sample_outcome_combo.currentData() or "all"),
            true_combination=str(
                self.true_combo_filter.currentData() or self.true_combo_filter.currentText()
                if self.true_combo_filter.currentIndex() > 0
                else ""
            ),
            predicted_combination=str(
                self.predicted_combo_filter.currentData()
                or self.predicted_combo_filter.currentText()
                if self.predicted_combo_filter.currentIndex() > 0
                else ""
            ),
            label=str(
                self.sample_label_filter.currentData() or self.sample_label_filter.currentText()
                if self.sample_label_filter.currentIndex() > 0
                else ""
            ),
            label_relation=str(self.sample_label_relation.currentData() or "all"),
            metadata_field=str(
                self.metadata_field_filter.currentData() or self.metadata_field_filter.currentText()
                if self.metadata_field_filter.currentIndex() > 0
                else ""
            ),
            metadata_value=self.metadata_value_filter.text().strip(),
            confidence_min=self._confidence_range[0],
            confidence_max=self._confidence_range[1],
            error_type=self._error_type_filter,
        )
        total = len(self.task.samples) if self.task else 0
        self.filtered_count_label.setText(f"{self.sample_proxy.rowCount()} / {total} 条")

    def _sample_selected(self, index: Any) -> None:
        source = self.sample_proxy.mapToSource(index)
        sample = self.sample_model.sample_at(source.row())
        if sample is None:
            return
        candidates = sorted(
            zip(sample.combination_labels, sample.combination_probabilities, strict=False),
            key=lambda value: value[1],
            reverse=True,
        )
        detail = {
            "file": str(sample.file_path),
            "status": sample.status.value,
            "true_label_vector": sample.true_label_vector,
            "true_combination": sample.true_combination,
            "predicted_label_vector": sample.predicted_label_vector,
            "predicted_combination": sample.predicted_combination,
            "predicted_sources": sample.predicted_sources,
            "false_positive_labels": sample.false_positive_labels,
            "false_negative_labels": sample.false_negative_labels,
            "first_candidate": candidates[0] if candidates else None,
            "second_candidate": candidates[1] if len(candidates) > 1 else None,
            "confidence_margin": sample.confidence_margin,
            "model_version": self.task.model_version if self.task else "",
            "input_shape": sample.input_shape,
            "metadata": sample.metadata,
            "combination_probabilities": dict(candidates),
            "error": sample.error_message,
        }
        self.sample_detail.setPlainText(json.dumps(detail, ensure_ascii=False, indent=2))
        self._show_probabilities(sample)
        self.waveform.clear()
        if sample.file_path.is_file():
            self.waveform_status.setText("正在后台按需读取原始 CSV 波形…")
            self.waveform_requested.emit(sample.file_path)
        else:
            self.waveform_status.setText("原始 CSV 文件当前不可用。")

    def _show_probabilities(self, sample: ValidationSample) -> None:
        for widget in self._probability_widgets:
            widget.setParent(None)
            widget.deleteLater()
        self._probability_widgets.clear()
        for label, probability in zip(sample.labels, sample.display_probabilities, strict=False):
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            name = QLabel(label)
            name.setMinimumWidth(120)
            bar = QProgressBar()
            bar.setRange(0, 10000)
            bar.setValue(round(probability * 10000))
            bar.setFormat(f"{probability * 100:.2f}%")
            layout.addWidget(name)
            layout.addWidget(bar, 1)
            self.probability_layout.addWidget(row)
            self._probability_widgets.append(row)

    def _label_row_clicked(self, index: Any) -> None:
        source = self.label_proxy.mapToSource(index)
        row = self.label_model.row_at(source.row())
        if row:
            self._set_combo_text(self.label_filter_combo, str(row["label"]))

    def _combination_row_clicked(self, index: Any) -> None:
        source = self.combination_proxy.mapToSource(index)
        row = self.combination_model.row_at(source.row())
        if row:
            self.apply_sample_filter(true_combination=str(row["true_combination"]))

    def _matrix_clicked(self, index: Any) -> None:
        true_combination, predicted_combination = index.data(ConfusionMatrixTableModel.CellRole)
        self.apply_sample_filter(
            true_combination=true_combination,
            predicted_combination=predicted_combination,
        )

    def _group_row_clicked(self, index: Any) -> None:
        source = self.group_proxy.mapToSource(index)
        row = self.group_model.row_at(source.row())
        if row:
            self.apply_sample_filter(
                metadata_field=self.group_field_combo.currentText(),
                metadata_value=str(row["value"]),
            )

    def _group_chart_clicked(self, event: Any) -> None:
        if not self._group_values:
            return
        point = self.group_chart.plotItem.vb.mapSceneToView(event.scenePos())
        index = round(point.x())
        if 0 <= index < len(self._group_values):
            self.apply_sample_filter(
                metadata_field=self.group_field_combo.currentText(),
                metadata_value=self._group_values[index],
            )

    def _apply_label_filter(self) -> None:
        label = (
            self.label_filter_combo.currentText() if self.label_filter_combo.currentIndex() else ""
        )
        self.apply_sample_filter(
            label=label,
            label_relation=str(self.label_relation_combo.currentData()),
        )

    def _toggle_matrix_percent(self, checked: bool) -> None:
        self.matrix_model.set_percentages(checked)

    def _quick_filter(self, value: str, column: int, descending: bool) -> None:
        self.clear_sample_filters()
        self.sample_proxy.special = (
            value if value.endswith("_to_triple") or value == "single_to_multi" else ""
        )  # type: ignore[attr-defined]
        self._set_combo_data(
            self.sample_outcome_combo,
            value if value in {"wrong", "all"} else "all",
        )
        self._apply_sample_filters()
        if column >= 0:
            self.sample_table.sortByColumn(
                column,
                Qt.SortOrder.DescendingOrder if descending else Qt.SortOrder.AscendingOrder,
            )

    def _most_error_label(self, field: str) -> None:
        if not self.task:
            return
        rows = self.task.metrics.get("labels", [])
        if not rows:
            return
        row = max(rows, key=lambda value: int(value.get(field, 0)))
        self.apply_sample_filter(
            label=str(row["label"]),
            label_relation="false_positive" if field == "fp" else "false_negative",
        )

    def _manifest_changed(self) -> None:
        self.report = None
        self.task = None
        self.preview_model.set_report(None)
        self.check_report.clear()
        self.progress_label.setText("清单已更改，请重新执行数据检查")
        self._update_buttons()

    def _request_manifest_check(self) -> None:
        path = Path(self.manifest_edit.text().strip())
        root_text = self.data_root_edit.text().strip()
        root = Path(root_text) if root_text else None
        self.manifest_check_requested.emit(
            path,
            root,
            str(self.missing_policy_combo.currentData()),
        )

    def _choose_manifest(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择验证清单",
            "",
            "CSV 文件 (*.csv)",
        )
        if path:
            self.manifest_edit.setText(path)

    def _choose_data_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择验证数据根目录")
        if path:
            self.data_root_edit.setText(path)
            self._manifest_changed()

    def _choose_history(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择历史验证结果目录")
        if path:
            self.history_requested.emit(Path(path))

    def _choose_batch_result(self) -> None:
        if self.report is None or not self.report.can_start:
            self.show_error("请先选择并检查 validation_manifest.csv。")
            return
        path = QFileDialog.getExistingDirectory(self, "选择已有批量预测结果目录")
        if path:
            self.batch_reuse_requested.emit(Path(path))

    def _choose_export(self) -> None:
        if self.task is None or not self.task.metrics:
            return
        path = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if path:
            self.export_requested.emit(Path(path))

    def _open_output(self) -> None:
        if self.task and self.task.output_directory.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.task.output_directory)))

    def _update_buttons(self) -> None:
        running = bool(self.task is not None and self.task.status in RUNNING_VALIDATION_STATUSES)
        paused = bool(self.task is not None and self.task.status == ValidationStatus.PAUSED)
        self.manifest_edit.setEnabled(not running)
        self.manifest_button.setEnabled(not running)
        self.data_root_edit.setEnabled(not running)
        self.root_button.setEnabled(not running)
        self.missing_policy_combo.setEnabled(not running)
        self.check_button.setEnabled(
            not running and self.loaded_model is not None and bool(self.manifest_edit.text())
        )
        self.start_button.setEnabled(
            not running
            and self.loaded_model is not None
            and self.report is not None
            and self.report.can_start
            and not running
        )
        self.pause_button.setEnabled(running and not paused)
        self.resume_button.setEnabled(paused)
        self.stop_button.setEnabled(running)
        has_metrics = bool(self.task and self.task.metrics)
        self.export_button.setEnabled(has_metrics and not running)
        self.open_output_button.setEnabled(bool(self.task and self.task.output_directory.is_dir()))
        self.reuse_batch_button.setEnabled(
            not running
            and self.loaded_model is not None
            and bool(self.report and self.report.can_start)
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if self.task and self.task.status in RUNNING_VALIDATION_STATUSES:
            return
        if any(
            url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() == ".csv"
            for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        if self.task and self.task.status in RUNNING_VALIDATION_STATUSES:
            return
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if url.isLocalFile() and path.suffix.lower() == ".csv":
                self.manifest_edit.setText(str(path))
                event.acceptProposedAction()
                return

    @staticmethod
    def _set_combo_text(combo: QComboBox, value: str) -> None:
        if not value:
            combo.setCurrentIndex(0)
            return
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        combo.setCurrentIndex(index if index >= 0 else 0)


__all__ = ["ValidationPage"]
