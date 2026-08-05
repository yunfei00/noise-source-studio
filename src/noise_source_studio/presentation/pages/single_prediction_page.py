"""Single-file runtime prediction workflow page."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from noise_source_studio.domain.models import (
    LoadedModel,
    PredictionOutcome,
    SignalPreview,
)
from noise_source_studio.presentation.widgets import (
    PAGE_CONTENT_MARGINS,
    PAGE_CONTENT_SPACING,
    EmptyState,
    FileDropZone,
    PageHeader,
    SectionCard,
)
from noise_source_studio.services.result_adapter import (
    normalize_prediction_result,
    result_value,
)


class SinglePredictionPage(QWidget):
    """Select, preview, predict and explicitly export one strict DATA CSV."""

    preview_requested = Signal(object)
    prediction_requested = Signal(object)
    export_requested = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.selected_file: Path | None = None
        self.loaded_model: LoadedModel | None = None
        self.current_outcome: PredictionOutcome | None = None
        self._preview_valid = False

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
                "单文件预测",
                "选择包含严格 DATA 数据段的 CSV，并使用已激活模型执行真实推理。",
            )
        )

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)

        input_card = SectionCard("输入文件")
        self.drop_zone = FileDropZone(
            "拖放 CSV 文件到此处",
            "仅接受 .csv 文件，解析和推理均在后台执行",
            accepted_extensions=(".csv",),
        )
        self.drop_zone.browse_requested.connect(self._choose_file)
        self.drop_zone.file_dropped.connect(self.set_selected_file)
        self.drop_zone.file_rejected.connect(
            lambda path: self._show_message("仅支持 CSV 文件。", error=True)
        )
        input_card.content_layout.addWidget(self.drop_zone)

        info_widget = QWidget()
        info_layout = QFormLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        self.file_name_label = QLabel("未选择")
        self.file_path_label = QLabel("—")
        self.file_size_label = QLabel("—")
        self.file_modified_label = QLabel("—")
        self.file_path_label.setWordWrap(True)
        info_layout.addRow("文件名称", self.file_name_label)
        info_layout.addRow("文件路径", self.file_path_label)
        info_layout.addRow("文件大小", self.file_size_label)
        info_layout.addRow("修改时间", self.file_modified_label)
        input_card.content_layout.addWidget(info_widget)

        preview_card = SectionCard(
            "信号预览",
            "曲线来自 runtime 正式 CSV 解析结果；仅显示数据可能抽样，推理输入不变。",
        )
        self.preview_plot = pg.PlotWidget()
        self.preview_plot.setObjectName("signalPreview")
        self.preview_plot.setMinimumHeight(235)
        self.preview_plot.setBackground("#ffffff")
        self.preview_plot.showGrid(x=True, y=True, alpha=0.18)
        self.preview_plot.setLabel("bottom", "采样点")
        self.preview_plot.setLabel("left", "幅值")
        preview_card.content_layout.addWidget(self.preview_plot)
        preview_info = QWidget()
        preview_form = QFormLayout(preview_info)
        preview_form.setContentsMargins(0, 0, 0, 0)
        self.preview_labels: dict[str, QLabel] = {}
        preview_fields = (
            ("points", "有效数据点"),
            ("minimum", "原始最小值"),
            ("maximum", "原始最大值"),
            ("mean", "平均值"),
            ("std", "标准差"),
            ("mode", "解析模式"),
            ("start", "DATA 起始行"),
            ("columns", "数据列"),
            ("encoding", "编码"),
            ("delimiter", "分隔符"),
        )
        for key, title in preview_fields:
            value_label = QLabel("—")
            self.preview_labels[key] = value_label
            preview_form.addRow(title, value_label)
        preview_card.content_layout.addWidget(preview_info)

        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.setSpacing(16)
        left_layout.addWidget(input_card)
        left_layout.addWidget(preview_card)

        model_card = SectionCard("推理配置")
        config_form = QFormLayout()
        self.current_model_label = QLabel("未配置")
        self.current_device_label = QLabel("待检测")
        self.current_mode_label = QLabel("—")
        config_form.addRow("当前模型", self.current_model_label)
        config_form.addRow("运行设备", self.current_device_label)
        config_form.addRow("预测模式", self.current_mode_label)
        model_card.content_layout.addLayout(config_form)
        self.predict_button = QPushButton("开始预测")
        self.predict_button.setObjectName("primaryButton")
        self.predict_button.setEnabled(False)
        self.predict_button.clicked.connect(self._request_prediction)
        model_card.content_layout.addWidget(self.predict_button)
        self.engine_message = QLabel("请先在模型管理中导入并激活兼容模型。")
        self.engine_message.setObjectName("warningBanner")
        self.engine_message.setWordWrap(True)
        model_card.content_layout.addWidget(self.engine_message)

        self.result_card = SectionCard("标签概率结果")
        self.result_content = QWidget()
        self.result_layout = QVBoxLayout(self.result_content)
        self.result_layout.setContentsMargins(0, 0, 0, 0)
        self.result_layout.addWidget(
            EmptyState("暂无预测结果", "完成真实推理后将在此动态显示全部标签。")
        )
        self.result_card.content_layout.addWidget(self.result_content)

        final_card = SectionCard("最终结果")
        final_form = QFormLayout()
        self.final_combination_label = QLabel("—")
        self.final_vector_label = QLabel("—")
        self.final_sources_label = QLabel("—")
        self.highest_combination_label = QLabel("—")
        for label in (
            self.final_combination_label,
            self.final_vector_label,
            self.final_sources_label,
            self.highest_combination_label,
        ):
            label.setWordWrap(True)
        final_form.addRow("预测组合", self.final_combination_label)
        final_form.addRow("解码向量", self.final_vector_label)
        final_form.addRow("预测噪声源", self.final_sources_label)
        final_form.addRow("最高组合", self.highest_combination_label)
        final_card.content_layout.addLayout(final_form)

        detail_card = SectionCard("详细结果")
        self.detail_view = QPlainTextEdit()
        self.detail_view.setObjectName("predictionDetails")
        self.detail_view.setReadOnly(True)
        self.detail_view.setPlaceholderText("推理后显示概率、阈值、设备、耗时和输入张量信息。")
        self.detail_view.setMinimumHeight(230)
        detail_card.content_layout.addWidget(self.detail_view)
        export_row = QHBoxLayout()
        self.contract_checkbox = QCheckBox("同时导出 inference contract")
        self.export_button = QPushButton("导出结果")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(
            lambda: self.export_requested.emit(self.contract_checkbox.isChecked())
        )
        export_row.addWidget(self.contract_checkbox)
        export_row.addStretch()
        export_row.addWidget(self.export_button)
        detail_card.content_layout.addLayout(export_row)
        self.export_path_label = QLabel()
        self.export_path_label.setObjectName("operationFeedback")
        self.export_path_label.setWordWrap(True)
        self.export_path_label.hide()
        detail_card.content_layout.addWidget(self.export_path_label)

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(16)
        right_layout.addWidget(model_card)
        right_layout.addWidget(self.result_card)
        right_layout.addWidget(final_card)
        right_layout.addWidget(detail_card)
        right_layout.addStretch()

        splitter.addWidget(left_column)
        splitter.addWidget(right_column)
        splitter.setSizes([650, 430])
        layout.addWidget(splitter)
        layout.addStretch()
        scroll.setWidget(content)
        root_layout.addWidget(scroll)

    def set_model(self, loaded_model: LoadedModel | None) -> None:
        """Reflect the retained runtime session and prediction availability."""
        self.loaded_model = loaded_model
        if loaded_model is None:
            self.current_model_label.setText("未配置")
            self.current_device_label.setText("待检测")
            self.current_mode_label.setText("—")
            self._show_message("请先在模型管理中导入并激活兼容模型。", error=False)
        else:
            self.current_model_label.setText(loaded_model.record.display_name)
            self.current_device_label.setText(loaded_model.device)
            self.current_mode_label.setText(loaded_model.prediction_mode)
            self.engine_message.hide()
        self._update_predict_enabled()

    def set_selected_file(self, path: str | Path) -> None:
        """Validate and display a selected CSV, clearing all stale results."""
        selected = Path(path)
        if selected.suffix.lower() != ".csv":
            self._show_message("仅支持 .csv 文件。", error=True)
            return
        if not selected.is_file():
            self._show_message("所选 CSV 文件不存在或无法读取。", error=True)
            return
        stat = selected.stat()
        self.selected_file = selected
        self.file_name_label.setText(selected.name)
        self.file_path_label.setText(str(selected))
        self.file_size_label.setText(self._format_file_size(stat.st_size))
        self.file_modified_label.setText(
            datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        )
        self._preview_valid = False
        self._clear_prediction()
        self._clear_preview()
        self._show_message("正在后台解析信号预览…", error=False)
        self._update_predict_enabled()
        self.preview_requested.emit(selected)

    def show_preview(self, preview: SignalPreview) -> None:
        """Plot sampled display data and show exact runtime parser statistics."""
        if self.selected_file != preview.source_path:
            return
        self.preview_plot.clear()
        self.preview_plot.plot(
            list(preview.display_values),
            pen=pg.mkPen(color="#2463a8", width=1.2),
        )
        values = {
            "points": f"{preview.original_point_count:,}",
            "minimum": f"{preview.raw_minimum:.8g}",
            "maximum": f"{preview.raw_maximum:.8g}",
            "mean": f"{preview.raw_mean:.8g}",
            "std": f"{preview.raw_standard_deviation:.8g}",
            "mode": preview.parser_mode,
            "start": str(preview.data_start_line),
            "columns": ", ".join(str(value) for value in preview.selected_columns),
            "encoding": preview.encoding,
            "delimiter": preview.delimiter,
        }
        for key, value in values.items():
            self.preview_labels[key].setText(value)
        self._preview_valid = True
        if self.loaded_model is not None:
            self.engine_message.hide()
        self._update_predict_enabled()

    def show_preview_error(self, message: str) -> None:
        """Restore controls after parsing failed."""
        self._preview_valid = False
        self._show_message(message, error=True)
        self._update_predict_enabled()

    def begin_prediction(self) -> None:
        """Lock controls and expose the running state."""
        self.predict_button.setEnabled(False)
        self.predict_button.setText("推理中…")
        self.drop_zone.setEnabled(False)
        self._show_message("正在后台执行真实模型推理，请稍候。", error=False)

    def show_prediction(self, outcome: PredictionOutcome) -> None:
        """Render dynamic probabilities and authoritative runtime decisions."""
        if self.selected_file != outcome.source_path:
            return
        self.current_outcome = outcome
        result = normalize_prediction_result(outcome.result)
        labels = list(self._result_value(result, "labels", []))
        decision_mode = str(self._result_value(result, "decision_mode", "unknown"))
        probability_key = (
            "label_marginal_probabilities"
            if decision_mode == "structured"
            else "multilabel_probabilities"
        )
        probabilities = list(self._result_value(result, probability_key, []))
        self._clear_layout(self.result_layout)
        semantics = QLabel(
            "结构化组合边缘概率" if decision_mode == "structured" else "多标签 sigmoid 概率"
        )
        semantics.setObjectName("sectionDescription")
        self.result_layout.addWidget(semantics)
        for label, probability in zip(labels, probabilities, strict=False):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            name_label = QLabel(str(label))
            name_label.setMinimumWidth(120)
            bar = QProgressBar()
            bar.setRange(0, 10000)
            bar.setValue(round(float(probability) * 10000))
            bar.setFormat(f"{float(probability) * 100:.2f}%")
            row_layout.addWidget(name_label)
            row_layout.addWidget(bar, 1)
            self.result_layout.addWidget(row)

        combination = self._result_value(result, "predicted_combination", "—")
        vector = self._result_value(result, "decoded_label_vector", [])
        sources = self._result_value(result, "predicted_sources", [])
        self.final_combination_label.setText(str(combination))
        self.final_vector_label.setText(json.dumps(vector, ensure_ascii=False))
        self.final_sources_label.setText(", ".join(str(value) for value in sources) or "无")
        self.highest_combination_label.setText(self._highest_combination_text(result))
        self.detail_view.setPlainText(
            json.dumps(self._detail_payload(outcome), ensure_ascii=False, indent=2, default=str)
        )
        self.predict_button.setText("开始预测")
        self.drop_zone.setEnabled(True)
        self.export_button.setEnabled(True)
        self.engine_message.hide()
        self._update_predict_enabled()

    def show_history(self, outcome: PredictionOutcome) -> None:
        """Restore a saved result without parsing the source or running inference."""
        self.selected_file = outcome.source_path
        self.file_name_label.setText(outcome.source_path.name)
        self.file_path_label.setText(str(outcome.source_path))
        try:
            stat = outcome.source_path.stat()
        except OSError:
            self.file_size_label.setText("源文件当前不可用")
            self.file_modified_label.setText("—")
        else:
            self.file_size_label.setText(self._format_file_size(stat.st_size))
            self.file_modified_label.setText(
                datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            )
        self._preview_valid = False
        self._clear_preview()
        self.show_prediction(outcome)
        self._show_message("已从历史记录恢复结果，未重新执行模型推理。", error=False)

    def show_prediction_error(self, message: str) -> None:
        """Restore all controls after an inference exception."""
        self.predict_button.setText("开始预测")
        self.drop_zone.setEnabled(True)
        self._show_message(message, error=True)
        self._update_predict_enabled()

    def show_export_result(self, json_path: Path, contract_path: Path | None) -> None:
        """Display actual runtime export locations."""
        message = f"JSON 已导出：{json_path}"
        if contract_path is not None:
            message += f"\n推理契约：{contract_path}"
        self.export_path_label.setText(message)
        self.export_path_label.setProperty("error", False)
        self.export_path_label.show()

    def show_export_error(self, message: str) -> None:
        """Display an export-specific failure."""
        self.export_path_label.setText(message)
        self.export_path_label.setProperty("error", True)
        self.export_path_label.show()

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择信号 CSV",
            "",
            "CSV 文件 (*.csv)",
        )
        if path:
            self.set_selected_file(path)

    def _request_prediction(self) -> None:
        if self.loaded_model is None:
            self._show_message("请先激活模型。", error=True)
        elif self.selected_file is None:
            self._show_message("请先选择 CSV 文件。", error=True)
        elif not self._preview_valid:
            self._show_message("信号解析尚未成功，无法开始预测。", error=True)
        else:
            self.prediction_requested.emit(self.selected_file)

    def _update_predict_enabled(self) -> None:
        running = self.predict_button.text() != "开始预测"
        self.predict_button.setEnabled(
            not running
            and self.loaded_model is not None
            and self.selected_file is not None
            and self._preview_valid
        )

    def _show_message(self, message: str, *, error: bool) -> None:
        self.engine_message.setText(message)
        self.engine_message.setProperty("error", error)
        self.engine_message.style().unpolish(self.engine_message)
        self.engine_message.style().polish(self.engine_message)
        self.engine_message.show()

    def _clear_preview(self) -> None:
        self.preview_plot.clear()
        for label in self.preview_labels.values():
            label.setText("—")

    def _clear_prediction(self) -> None:
        self.current_outcome = None
        self.export_button.setEnabled(False)
        self.export_path_label.hide()
        self.detail_view.clear()
        self.final_combination_label.setText("—")
        self.final_vector_label.setText("—")
        self.final_sources_label.setText("—")
        self.highest_combination_label.setText("—")
        self._clear_layout(self.result_layout)
        self.result_layout.addWidget(
            EmptyState("暂无预测结果", "完成真实推理后将在此动态显示全部标签。")
        )

    @staticmethod
    def _clear_layout(layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    @staticmethod
    def _result_value(result: Any, key: str, default: Any) -> Any:
        return result_value(result, key, default)

    @classmethod
    def _highest_combination_text(cls, result: Any) -> str:
        labels = list(cls._result_value(result, "combination_labels", []))
        probabilities = cls._result_value(result, "combination_probabilities", None)
        if not labels or probabilities is None:
            return "multilabel 模式不使用组合 softmax"
        values = list(probabilities)
        if not values:
            return "—"
        index = max(range(len(values)), key=values.__getitem__)
        return f"{labels[index]} · {float(values[index]) * 100:.2f}%"

    @classmethod
    def _detail_payload(cls, outcome: PredictionOutcome) -> dict[str, Any]:
        result = normalize_prediction_result(outcome.result)
        decision_mode = str(cls._result_value(result, "decision_mode", "unknown"))
        thresholds_applicable = bool(cls._result_value(result, "thresholds_applicable", False))
        decision = (
            "最终组合由结构化组合概率 argmax 决定，阈值不参与最终结果。"
            if decision_mode == "structured"
            else "最终标签由 sigmoid 概率大于等于对应阈值决定。"
        )
        return {
            "task_id": outcome.task_id,
            "started_at": outcome.started_at.isoformat(),
            "completed_at": outcome.completed_at.isoformat(),
            "duration_seconds": outcome.duration_seconds,
            "model_name": outcome.model.model_name,
            "model_version": outcome.model.model_version,
            "runtime_version": cls._result_value(result, "runtime_version", "unknown"),
            "device": cls._result_value(result, "device", "unknown"),
            "input_shape": cls._result_value(result, "input_shape", []),
            "prediction_mode": decision_mode,
            "labels": cls._result_value(result, "labels", []),
            "multilabel_probabilities": cls._result_value(result, "multilabel_probabilities", []),
            "combination_labels": cls._result_value(result, "combination_labels", []),
            "combination_probabilities": cls._result_value(
                result, "combination_probabilities", None
            ),
            "label_marginal_probabilities": cls._result_value(
                result, "label_marginal_probabilities", []
            ),
            "auxiliary_logits": cls._result_value(result, "auxiliary_logits", None),
            "thresholds": cls._result_value(result, "thresholds", []),
            "thresholds_applicable": thresholds_applicable,
            "decoded_label_vector": cls._result_value(result, "decoded_label_vector", []),
            "predicted_combination": cls._result_value(result, "predicted_combination", ""),
            "predicted_sources": cls._result_value(result, "predicted_sources", []),
            "decision_explanation": decision,
        }

    @staticmethod
    def _format_file_size(size: int) -> str:
        if size < 1024:
            return f"{size} B"
        if size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size / (1024 * 1024):.1f} MB"
