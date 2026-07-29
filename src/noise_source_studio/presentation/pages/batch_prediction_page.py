"""Batch prediction workflow page."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from noise_source_studio.presentation.widgets import PageHeader, SectionCard, create_table


class BatchPredictionPage(QWidget):
    """Batch controls, progress state and file task table."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 26, 28, 28)
        layout.setSpacing(20)
        layout.addWidget(
            PageHeader(
                "批量预测",
                "建立待处理文件队列；真实任务执行将在推理服务接入后启用。",
            )
        )

        toolbar = SectionCard("任务操作")
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        for text in ("添加文件", "添加文件夹", "清空列表"):
            button = QPushButton(text)
            button.setObjectName("secondaryButton")
            button_layout.addWidget(button)
        button_layout.addStretch()
        for text in ("开始", "暂停", "停止"):
            button = QPushButton(text)
            button.setEnabled(False)
            if text == "开始":
                button.setObjectName("primaryButton")
            button_layout.addWidget(button)
        toolbar.content_layout.addLayout(button_layout)

        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(QLabel("总进度"))
        progress_layout.addWidget(self.progress_bar, 1)
        state_label = QLabel("等待添加任务")
        state_label.setObjectName("statusNeutral")
        state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(state_label)
        toolbar.content_layout.addLayout(progress_layout)
        layout.addWidget(toolbar)

        table_card = SectionCard("文件任务")
        self.task_table = create_table(
            ("序号", "文件名称", "文件路径", "状态", "预测结果", "推理耗时", "错误信息"),
            minimum_height=320,
        )
        self.task_table.setObjectName("batchTaskTable")
        table_card.content_layout.addWidget(self.task_table)
        empty_hint = QLabel("尚未添加文件。任务列表不会预填充虚构数据。")
        empty_hint.setObjectName("tableEmptyHint")
        table_card.content_layout.addWidget(empty_hint)
        layout.addWidget(table_card, 1)
