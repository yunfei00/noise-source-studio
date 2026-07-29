"""Application settings page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from noise_source_studio.common.exceptions import ConfigurationError
from noise_source_studio.infrastructure.config import AppSettings, SettingsManager
from noise_source_studio.presentation.widgets import PageHeader, SectionCard


class SettingsPage(QWidget):
    """Editable, non-sensitive application settings."""

    settings_saved = Signal(object)

    def __init__(
        self,
        settings_manager: SettingsManager,
        settings: AppSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.settings = settings

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
                "系统设置",
                "设置保存在当前用户的应用数据目录中，不存储密码、令牌或其他凭据。",
            )
        )

        directories_card = SectionCard("通用设置", "配置任务输出与本地运行数据目录。")
        directory_form = QFormLayout()
        self.output_directory_edit = self._directory_row(
            directory_form,
            "默认输出目录",
            settings.output_directory,
        )
        self.model_directory_edit = self._directory_row(
            directory_form,
            "模型目录",
            settings.model_directory,
        )
        self.log_directory_edit = self._directory_row(
            directory_form,
            "日志目录",
            settings.log_directory,
        )
        directories_card.content_layout.addLayout(directory_form)
        layout.addWidget(directories_card)

        interface_card = SectionCard("界面与运行")
        interface_form = QFormLayout()
        self.device_combo = QComboBox()
        self.device_combo.addItem("自动选择（推荐）", "auto")
        self.device_combo.addItem("CPU（占位选项）", "cpu")
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("浅色", "light")
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1180, 3840)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(720, 2160)
        interface_form.addRow("默认计算设备", self.device_combo)
        interface_form.addRow("界面主题", self.theme_combo)
        interface_form.addRow("启动宽度", self.width_spin)
        interface_form.addRow("启动高度", self.height_spin)
        interface_card.content_layout.addLayout(interface_form)
        layout.addWidget(interface_card)

        action_row = QHBoxLayout()
        self.feedback_label = QLabel()
        self.feedback_label.setObjectName("settingsFeedback")
        action_row.addWidget(self.feedback_label)
        action_row.addStretch()
        restore_button = QPushButton("恢复默认设置")
        restore_button.clicked.connect(self.restore_defaults)
        save_button = QPushButton("保存设置")
        save_button.setObjectName("primaryButton")
        save_button.clicked.connect(self.save_settings)
        action_row.addWidget(restore_button)
        action_row.addWidget(save_button)
        layout.addLayout(action_row)
        layout.addStretch()

        scroll.setWidget(content)
        root_layout.addWidget(scroll)
        self._populate(settings)

    def save_settings(self) -> None:
        """Validate and persist values currently shown in the form."""
        updated = self.settings.model_copy(
            update={
                "output_directory": Path(self.output_directory_edit.text().strip()),
                "model_directory": Path(self.model_directory_edit.text().strip()),
                "log_directory": Path(self.log_directory_edit.text().strip()),
                "default_device": self.device_combo.currentData(),
                "theme": self.theme_combo.currentData(),
                "window_width": self.width_spin.value(),
                "window_height": self.height_spin.value(),
            }
        )
        try:
            self.settings_manager.save(updated)
        except ConfigurationError as exc:
            self.feedback_label.setText(str(exc))
            self.feedback_label.setProperty("error", True)
            self.feedback_label.style().unpolish(self.feedback_label)
            self.feedback_label.style().polish(self.feedback_label)
            return

        self.settings = updated
        self.feedback_label.setProperty("error", False)
        self.feedback_label.setText("设置已保存")
        self.settings_saved.emit(updated)

    def restore_defaults(self) -> None:
        """Restore and persist platform defaults."""
        defaults = self.settings_manager.default_settings()
        self.settings_manager.save(defaults)
        self.settings = defaults
        self._populate(defaults)
        self.feedback_label.setText("已恢复默认设置")
        self.settings_saved.emit(defaults)

    def _directory_row(
        self,
        form: QFormLayout,
        label: str,
        value: Path,
    ) -> QLineEdit:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(str(value))
        browse = QPushButton("浏览")
        browse.clicked.connect(lambda checked=False, target=edit: self._browse_directory(target))
        row_layout.addWidget(edit, 1)
        row_layout.addWidget(browse)
        form.addRow(label, row)
        return edit

    def _browse_directory(self, target: QLineEdit) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择目录", target.text())
        if selected:
            target.setText(selected)

    def _populate(self, settings: AppSettings) -> None:
        self.output_directory_edit.setText(str(settings.output_directory))
        self.model_directory_edit.setText(str(settings.model_directory))
        self.log_directory_edit.setText(str(settings.log_directory))
        self._select_data(self.device_combo, settings.default_device)
        self._select_data(self.theme_combo, settings.theme)
        self.width_spin.setValue(settings.window_width)
        self.height_spin.setValue(settings.window_height)

    @staticmethod
    def _select_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
