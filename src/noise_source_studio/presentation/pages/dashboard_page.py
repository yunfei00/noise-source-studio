"""Workspace overview page."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from noise_source_studio.presentation.navigation import navigation_icon
from noise_source_studio.presentation.widgets import (
    PAGE_CONTENT_MARGINS,
    PAGE_CONTENT_SPACING,
    EmptyState,
    MetricCard,
    PageHeader,
    SectionCard,
)


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
        layout.setContentsMargins(*PAGE_CONTENT_MARGINS)
        layout.setSpacing(PAGE_CONTENT_SPACING)

        layout.addWidget(
            PageHeader(
                "工作台",
                "欢迎使用噪声源智能识别平台。完成模型配置后即可开始分析任务。",
            )
        )

        status_layout = QHBoxLayout()
        status_layout.setSpacing(12)
        self.status_cards = {
            "model": MetricCard("当前模型", "未配置", "请前往模型管理导入兼容模型"),
            "device": MetricCard("计算设备", "待检测", "设备将在推理引擎接入后检测"),
            "application": MetricCard("应用状态", "正常", "基础服务运行正常"),
        }
        for card in self.status_cards.values():
            status_layout.addWidget(card)
        layout.addLayout(status_layout)

        quick_card = SectionCard("快速开始", "选择一个工作流进入对应页面。")
        quick_layout = QGridLayout()
        quick_layout.setHorizontalSpacing(12)
        quick_layout.setVerticalSpacing(12)
        actions = (
            ("single_prediction", "单文件预测", "分析单个信号文件", 1),
            ("batch_prediction", "批量预测", "管理多个文件任务", 2),
            ("validation", "模型验证", "评估兼容模型性能", 3),
        )
        self.quick_action_buttons: list[QPushButton] = []
        for column, (key, title, description, page_index) in enumerate(actions):
            button = QPushButton(f"{title}\n{description}  ·  进入 →")
            button.setObjectName("quickActionButton")
            button.setIcon(navigation_icon(key))
            button.setIconSize(QSize(22, 22))
            button.setMinimumHeight(70)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            button.setAccessibleName(f"进入{title}")
            button.setToolTip(f"进入{title}")
            button.clicked.connect(
                lambda checked=False, index=page_index: self.navigation_requested.emit(index)
            )
            quick_layout.addWidget(button, 0, column)
            self.quick_action_buttons.append(button)
        quick_card.content_layout.addLayout(quick_layout)
        layout.addWidget(quick_card)

        recent_card = SectionCard("最近任务")
        recent_card.content_layout.addWidget(
            EmptyState(
                "暂无任务记录",
                "完成预测或模型验证后，最近任务将显示在这里。",
                max_text_width=480,
            )
        )
        layout.addWidget(recent_card)
        layout.addStretch()

        scroll.setWidget(content)
        root_layout.addWidget(scroll)
