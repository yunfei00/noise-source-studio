"""Model validation workflow page."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from noise_source_studio.presentation.widgets import (
    MetricCard,
    PageHeader,
    SectionCard,
    create_table,
)


class ValidationPage(QWidget):
    """Validation setup and empty metrics/report structure."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 26, 28, 28)
        layout.setSpacing(20)
        layout.addWidget(
            PageHeader(
                "模型验证",
                "使用经确认的验证清单评估兼容模型；当前版本不计算真实指标。",
            )
        )

        setup_card = SectionCard("验证配置")
        setup_layout = QHBoxLayout()
        manifest = QLineEdit()
        manifest.setPlaceholderText("选择验证清单")
        model = QComboBox()
        model.addItem("未安装可用模型")
        start_button = QPushButton("开始验证")
        start_button.setObjectName("primaryButton")
        start_button.setEnabled(False)
        export_button = QPushButton("导出验证报告")
        export_button.setEnabled(False)
        setup_layout.addWidget(manifest, 2)
        setup_layout.addWidget(QPushButton("选择清单"))
        setup_layout.addWidget(model, 1)
        setup_layout.addWidget(start_button)
        setup_layout.addWidget(export_button)
        setup_card.content_layout.addLayout(setup_layout)
        layout.addWidget(setup_card)

        metrics_layout = QGridLayout()
        metrics_layout.setHorizontalSpacing(12)
        for column, label in enumerate(("样本数量", "宏平均 F1", "微平均 F1", "完全匹配率")):
            metrics_layout.addWidget(MetricCard(label, "—", "尚未执行验证"), 0, column)
        layout.addLayout(metrics_layout)

        tabs = QTabWidget()
        tabs.addTab(
            create_table(("标签", "精确率", "召回率", "F1", "支持样本数")),
            "标签指标",
        )
        tabs.addTab(
            create_table(("标签组合", "样本数", "正确数", "匹配率")),
            "组合标签统计",
        )
        tabs.addTab(
            create_table(("文件", "真实标签", "预测标签", "错误类型")),
            "错误样本",
        )
        layout.addWidget(tabs, 1)
