"""Workspace overview page."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from noise_source_studio.presentation.widgets import EmptyState, MetricCard, PageHeader, SectionCard


class DashboardPage(QWidget):
    """Landing page with current status and workflow entry points."""

    navigation_requested = Signal(int)

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
                "工作台",
                "欢迎使用噪声源智能识别平台。完成模型配置后即可开始分析任务。",
            )
        )

        status_layout = QHBoxLayout()
        status_layout.setSpacing(12)
        status_layout.addWidget(MetricCard("当前模型", "未配置", "请前往模型管理导入兼容模型"))
        status_layout.addWidget(MetricCard("运行设备", "自动选择", "设备将在推理引擎接入后检测"))
        status_layout.addWidget(MetricCard("系统状态", "就绪", "基础服务运行正常"))
        layout.addLayout(status_layout)

        quick_card = SectionCard("快速开始", "选择一个工作流进入对应页面。")
        quick_layout = QGridLayout()
        quick_layout.setHorizontalSpacing(12)
        quick_layout.setVerticalSpacing(12)
        actions = (
            ("单文件预测", "分析单个信号文件", 1),
            ("批量预测", "管理多个文件任务", 2),
            ("模型验证", "评估兼容模型性能", 3),
        )
        for column, (title, description, page_index) in enumerate(actions):
            button = QPushButton(f"{title}\n{description}")
            button.setObjectName("quickActionButton")
            button.setMinimumHeight(76)
            button.clicked.connect(
                lambda checked=False, index=page_index: self.navigation_requested.emit(index)
            )
            quick_layout.addWidget(button, 0, column)
        quick_card.content_layout.addLayout(quick_layout)
        layout.addWidget(quick_card)

        recent_card = SectionCard("最近任务")
        recent_card.content_layout.addWidget(
            EmptyState("暂无任务记录", "完成预测或验证任务后，最近活动会显示在这里。")
        )
        layout.addWidget(recent_card)
        layout.addStretch()

        scroll.setWidget(content)
        root_layout.addWidget(scroll)
