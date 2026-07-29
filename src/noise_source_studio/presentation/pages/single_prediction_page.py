"""Single-file prediction workflow page."""

from __future__ import annotations

from pathlib import Path

import pyqtgraph as pg
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from noise_source_studio.presentation.widgets import (
    EmptyState,
    FileDropZone,
    PageHeader,
    SectionCard,
)


class SinglePredictionPage(QWidget):
    """File selection, signal preview and empty prediction result skeleton."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(28, 26, 28, 28)
        layout.setSpacing(20)
        layout.addWidget(
            PageHeader(
                "单文件预测",
                "选择一个信号文件并在已配置模型后执行多标签识别。",
            )
        )

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)

        input_card = SectionCard("输入文件")
        self.drop_zone = FileDropZone()
        self.drop_zone.browse_requested.connect(self._choose_file)
        self.drop_zone.file_dropped.connect(self._set_selected_file)
        input_card.content_layout.addWidget(self.drop_zone)

        info_widget = QWidget()
        info_layout = QFormLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        self.file_name_label = QLabel("未选择")
        self.file_path_label = QLabel("—")
        self.file_path_label.setWordWrap(True)
        info_layout.addRow("文件名称", self.file_name_label)
        info_layout.addRow("文件路径", self.file_path_label)
        input_card.content_layout.addWidget(info_widget)

        preview_card = SectionCard("信号预览", "选择文件后，由后续数据适配器解析并显示信号。")
        self.preview_plot = pg.PlotWidget()
        self.preview_plot.setObjectName("signalPreview")
        self.preview_plot.setMinimumHeight(235)
        self.preview_plot.setBackground("#ffffff")
        self.preview_plot.showGrid(x=True, y=True, alpha=0.18)
        self.preview_plot.setLabel("bottom", "采样点")
        self.preview_plot.setLabel("left", "幅值")
        preview_card.content_layout.addWidget(self.preview_plot)

        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.setSpacing(16)
        left_layout.addWidget(input_card)
        left_layout.addWidget(preview_card)

        model_card = SectionCard("推理配置")
        config_form = QFormLayout()
        config_form.addRow("当前模型", QLabel("未配置"))
        config_form.addRow("运行设备", QLabel("自动选择（待检测）"))
        model_card.content_layout.addLayout(config_form)
        self.predict_button = QPushButton("开始预测")
        self.predict_button.setObjectName("primaryButton")
        self.predict_button.clicked.connect(self._show_unconfigured_message)
        model_card.content_layout.addWidget(self.predict_button)
        self.engine_message = QLabel()
        self.engine_message.setObjectName("warningBanner")
        self.engine_message.setWordWrap(True)
        self.engine_message.hide()
        model_card.content_layout.addWidget(self.engine_message)

        result_card = SectionCard("标签概率结果")
        result_card.content_layout.addWidget(
            EmptyState("暂无预测结果", "平台不会生成随机概率；请先配置兼容推理引擎。")
        )
        detail_card = SectionCard("推理详情")
        detail_card.content_layout.addWidget(
            EmptyState("暂无推理详情", "任务执行后将显示模型、设备和耗时等信息。")
        )

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(6, 0, 0, 0)
        right_layout.setSpacing(16)
        right_layout.addWidget(model_card)
        right_layout.addWidget(result_card)
        right_layout.addWidget(detail_card)
        right_layout.addStretch()

        splitter.addWidget(left_column)
        splitter.addWidget(right_column)
        splitter.setSizes([650, 430])
        layout.addWidget(splitter)
        layout.addStretch()

        scroll.setWidget(content)
        root_layout.addWidget(scroll)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择信号文件")
        if path:
            self._set_selected_file(path)

    def _set_selected_file(self, path: str) -> None:
        selected = Path(path)
        self.file_name_label.setText(selected.name)
        self.file_path_label.setText(str(selected))

    def _show_unconfigured_message(self) -> None:
        self.engine_message.setText("推理引擎尚未配置，请先导入兼容模型。")
        self.engine_message.show()
