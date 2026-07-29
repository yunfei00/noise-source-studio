"""Model management page."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from noise_source_studio.presentation.widgets import (
    PAGE_CONTENT_MARGINS,
    PAGE_CONTENT_SPACING,
    EmptyState,
    FileDropZone,
    PageHeader,
    SectionCard,
    create_table,
)


class ModelManagementPage(QWidget):
    """Installed model list, import area and empty model details."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*PAGE_CONTENT_MARGINS)
        layout.setSpacing(PAGE_CONTENT_SPACING)
        layout.addWidget(
            PageHeader(
                "模型管理",
                "导入、校验并激活兼容模型。模型格式将在后续阶段定义。",
            )
        )

        current_card = SectionCard("当前模型")
        current_row = QHBoxLayout()
        current_row.addWidget(QLabel("尚未配置活动模型"))
        current_row.addStretch()
        state = QLabel("未配置")
        state.setObjectName("statusWarning")
        current_row.addWidget(state)
        current_card.content_layout.addLayout(current_row)
        layout.addWidget(current_card)

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)

        list_card = SectionCard("已安装模型")
        self.model_table = create_table(("模型名称", "版本", "标签数量", "状态", "导入时间"))
        self.model_table.setObjectName("modelTable")
        list_card.content_layout.addWidget(self.model_table)
        action_row = QHBoxLayout()
        for text in ("激活模型", "删除模型", "完整性校验"):
            button = QPushButton(text)
            button.setEnabled(False)
            action_row.addWidget(button)
        action_row.addStretch()
        list_card.content_layout.addLayout(action_row)
        list_card.content_layout.addWidget(
            EmptyState("没有已安装模型", "导入功能会在模型包契约确认后启用。")
        )

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(8, 0, 0, 0)
        side_layout.setSpacing(16)
        import_card = SectionCard("导入模型")
        drop_zone = FileDropZone(
            "拖放模型包到此处",
            "当前阶段仅展示入口，不会尝试解析未知模型。",
        )
        drop_zone.browse_button.setEnabled(False)
        import_card.content_layout.addWidget(drop_zone)
        detail_card = SectionCard("模型详情")
        detail_card.content_layout.addWidget(
            EmptyState("请选择一个模型", "模型元数据、标签定义和校验信息将在这里显示。")
        )
        side_layout.addWidget(import_card)
        side_layout.addWidget(detail_card)
        side_layout.addStretch()

        splitter.addWidget(list_card)
        splitter.addWidget(side)
        splitter.setSizes([690, 390])
        layout.addWidget(splitter, 1)
