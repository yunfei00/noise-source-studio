"""Real model-package management page."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from noise_source_studio.domain.models import ModelRecord, PackageInspection
from noise_source_studio.presentation.widgets import (
    PAGE_CONTENT_MARGINS,
    PAGE_CONTENT_SPACING,
    EmptyState,
    PageHeader,
    SectionCard,
    create_table,
)


class ModelManagementPage(QWidget):
    """Import, register, activate, verify and remove runtime model packages."""

    package_selected = Signal(object)
    package_import_requested = Signal(object)
    activation_requested = Signal(str)
    deletion_requested = Signal(str)
    integrity_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._candidate: PackageInspection | None = None
        self._records: dict[str, ModelRecord] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(*PAGE_CONTENT_MARGINS)
        layout.setSpacing(PAGE_CONTENT_SPACING)
        layout.addWidget(
            PageHeader(
                "模型管理",
                "导入经过正式运行时校验的模型包，并管理当前活动模型。",
            )
        )

        current_card = SectionCard("当前模型")
        current_row = QHBoxLayout()
        self.current_model_label = QLabel("尚未配置活动模型")
        self.current_model_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        current_row.addWidget(self.current_model_label)
        current_row.addStretch()
        self.current_state_label = QLabel("未配置")
        self.current_state_label.setObjectName("statusWarning")
        current_row.addWidget(self.current_state_label)
        current_card.content_layout.addLayout(current_row)
        layout.addWidget(current_card)

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)

        list_card = SectionCard("已安装模型")
        self.model_table = create_table(("模型名称", "版本", "标签数量", "状态", "完整性"))
        self.model_table.setObjectName("modelTable")
        self.model_table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.model_table.itemSelectionChanged.connect(self._update_action_state)
        list_card.content_layout.addWidget(self.model_table)
        action_row = QHBoxLayout()
        self.activate_button = QPushButton("激活模型")
        self.delete_button = QPushButton("删除模型")
        self.integrity_button = QPushButton("完整性校验")
        self.activate_button.clicked.connect(self._request_activation)
        self.delete_button.clicked.connect(self._request_deletion)
        self.integrity_button.clicked.connect(self._request_integrity)
        for button in (
            self.activate_button,
            self.delete_button,
            self.integrity_button,
        ):
            button.setEnabled(False)
            action_row.addWidget(button)
        action_row.addStretch()
        list_card.content_layout.addLayout(action_row)
        self.empty_models = EmptyState(
            "没有已安装模型",
            "选择正式模型包目录，校验通过后即可导入。",
        )
        list_card.content_layout.addWidget(self.empty_models)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(8, 0, 0, 0)
        side_layout.setSpacing(16)
        import_card = SectionCard(
            "导入模型",
            "模型包将复制到当前用户应用数据目录，不会写入源码目录。",
        )
        self.package_path_label = QLabel("尚未选择模型包目录")
        self.package_path_label.setObjectName("pathValue")
        self.package_path_label.setWordWrap(True)
        select_button = QPushButton("选择模型包目录")
        select_button.clicked.connect(self._choose_package_directory)
        self.import_button = QPushButton("导入已校验模型")
        self.import_button.setObjectName("primaryButton")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self._request_import)
        import_card.content_layout.addWidget(self.package_path_label)
        import_card.content_layout.addWidget(select_button)
        import_card.content_layout.addWidget(self.import_button)

        detail_card = SectionCard("模型详情")
        self.manifest_view = QPlainTextEdit()
        self.manifest_view.setObjectName("manifestView")
        self.manifest_view.setReadOnly(True)
        self.manifest_view.setPlaceholderText("校验候选模型包或选择已安装模型后显示 manifest。")
        self.manifest_view.setMinimumHeight(220)
        detail_card.content_layout.addWidget(self.manifest_view)
        self.feedback_label = QLabel()
        self.feedback_label.setObjectName("operationFeedback")
        self.feedback_label.setWordWrap(True)
        self.feedback_label.hide()
        detail_card.content_layout.addWidget(self.feedback_label)

        side_layout.addWidget(import_card)
        side_layout.addWidget(detail_card, 1)

        splitter.addWidget(list_card)
        splitter.addWidget(side)
        splitter.setSizes([690, 390])
        layout.addWidget(splitter, 1)

    def set_models(self, records: tuple[ModelRecord, ...]) -> None:
        """Refresh the installed model table from the registry."""
        self._records = {record.identifier: record for record in records}
        self.model_table.setRowCount(len(records))
        for row, record in enumerate(records):
            values = (
                record.model_name,
                record.model_version,
                str(len(record.manifest.get("labels", []))),
                "活动" if record.is_active else "未激活",
                "完整" if record.integrity_status == "valid" else "异常",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, record.identifier)
                self.model_table.setItem(row, column, item)
        self.empty_models.setVisible(not records)
        self._update_action_state()
        active = next((record for record in records if record.is_active), None)
        self.set_active_model(active)

    def set_active_model(self, record: ModelRecord | None, device: str = "") -> None:
        """Update the existing current-model card without changing its layout."""
        if record is None:
            self.current_model_label.setText("尚未配置活动模型")
            self.current_state_label.setText("未配置")
            self.current_state_label.setObjectName("statusWarning")
        else:
            mode = record.manifest.get("prediction_mode", "unknown")
            device_text = f" · {device}" if device else ""
            self.current_model_label.setText(
                f"{record.model_name} · {record.model_version} · {mode}{device_text}"
            )
            self.current_state_label.setText("已激活")
            self.current_state_label.setObjectName("statusSuccess")
        self.current_state_label.style().unpolish(self.current_state_label)
        self.current_state_label.style().polish(self.current_state_label)

    def show_package_inspection(self, inspection: PackageInspection) -> None:
        """Display the verified manifest and enable explicit import."""
        self._candidate = inspection
        self.package_path_label.setText(str(inspection.package_path))
        self.manifest_view.setPlainText(
            json.dumps(inspection.manifest, ensure_ascii=False, indent=2)
        )
        self.import_button.setEnabled(True)
        self.show_feedback(
            f"校验通过 · runtime {inspection.runtime_version} · "
            f"SHA256 {inspection.verification.get('checkpoint_sha256', '—')}",
            error=False,
        )

    def show_feedback(self, message: str, *, error: bool) -> None:
        """Show a concise operation result."""
        self.feedback_label.setText(message)
        self.feedback_label.setProperty("error", error)
        self.feedback_label.style().unpolish(self.feedback_label)
        self.feedback_label.style().polish(self.feedback_label)
        self.feedback_label.show()

    def set_busy(self, busy: bool, message: str = "") -> None:
        """Disable duplicate actions while a worker owns model state."""
        self.model_table.setEnabled(not busy)
        self.import_button.setEnabled(not busy and self._candidate is not None)
        if busy and message:
            self.show_feedback(message, error=False)
        self._update_action_state()

    def _selected_identifier(self) -> str | None:
        items = self.model_table.selectedItems()
        return str(items[0].data(Qt.ItemDataRole.UserRole)) if items else None

    def _selected_record(self) -> ModelRecord | None:
        identifier = self._selected_identifier()
        return self._records.get(identifier) if identifier else None

    def _update_action_state(self) -> None:
        record = self._selected_record()
        enabled = record is not None and self.model_table.isEnabled()
        self.activate_button.setEnabled(enabled and not record.is_active if record else False)
        self.delete_button.setEnabled(enabled and not record.is_active if record else False)
        self.integrity_button.setEnabled(enabled)
        if record is not None:
            self.manifest_view.setPlainText(
                json.dumps(record.manifest, ensure_ascii=False, indent=2)
            )

    def _choose_package_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择模型包目录")
        if not directory:
            return
        self._candidate = None
        self.import_button.setEnabled(False)
        self.package_path_label.setText(directory)
        self.package_selected.emit(Path(directory))

    def _request_import(self) -> None:
        if self._candidate is not None:
            self.package_import_requested.emit(self._candidate.package_path)

    def _request_activation(self) -> None:
        identifier = self._selected_identifier()
        if identifier:
            self.activation_requested.emit(identifier)

    def _request_deletion(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        answer = QMessageBox.question(
            self,
            "删除模型",
            f"确定删除未激活模型 {record.display_name}？",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.deletion_requested.emit(record.identifier)

    def _request_integrity(self) -> None:
        identifier = self._selected_identifier()
        if identifier:
            self.integrity_requested.emit(identifier)
