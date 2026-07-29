"""Task history page."""

from __future__ import annotations

from PySide6.QtCore import QDate
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from noise_source_studio.presentation.widgets import PageHeader, SectionCard, create_table


class HistoryPage(QWidget):
    """Task filters and empty historical task table."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 26, 28, 28)
        layout.setSpacing(20)
        layout.addWidget(
            PageHeader(
                "任务历史",
                "查询预测与验证任务。持久化任务记录将在业务数据结构确认后接入。",
            )
        )

        filters = SectionCard("筛选条件")
        filter_layout = QHBoxLayout()
        task_type = QComboBox()
        task_type.addItems(("全部任务类型", "单文件预测", "批量预测", "模型验证"))
        status = QComboBox()
        status.addItems(("全部状态", "已完成", "已失败", "已取消"))
        start_date = QDateEdit(QDate.currentDate().addMonths(-1))
        start_date.setCalendarPopup(True)
        end_date = QDateEdit(QDate.currentDate())
        end_date.setCalendarPopup(True)
        keyword = QLineEdit()
        keyword.setPlaceholderText("任务名称或文件名")
        search = QPushButton("搜索")
        search.setObjectName("primaryButton")
        filter_layout.addWidget(task_type)
        filter_layout.addWidget(start_date)
        filter_layout.addWidget(end_date)
        filter_layout.addWidget(status)
        filter_layout.addWidget(keyword, 1)
        filter_layout.addWidget(search)
        filters.content_layout.addLayout(filter_layout)
        layout.addWidget(filters)

        history_card = SectionCard("历史任务")
        self.history_table = create_table(
            ("任务编号", "任务类型", "创建时间", "完成时间", "状态", "摘要"),
            minimum_height=310,
        )
        history_card.content_layout.addWidget(self.history_table)
        action_row = QHBoxLayout()
        action_row.addStretch()
        details_button = QPushButton("查看详情")
        details_button.setEnabled(False)
        export_button = QPushButton("导出结果")
        export_button.setEnabled(False)
        action_row.addWidget(details_button)
        action_row.addWidget(export_button)
        history_card.content_layout.addLayout(action_row)
        layout.addWidget(history_card, 1)
