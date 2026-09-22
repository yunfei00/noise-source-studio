"""Application settings page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from noise_source_studio.common.exceptions import ConfigurationError
from noise_source_studio.domain.device import DeviceProbeReport
from noise_source_studio.infrastructure.config import AppSettings, SettingsManager
from noise_source_studio.presentation.widgets import (
    PAGE_CONTENT_MARGINS,
    PAGE_CONTENT_SPACING,
    PageHeader,
    SectionCard,
)


class SettingsPage(QWidget):
    """Editable, non-sensitive application settings."""

    settings_saved = Signal(object)
    settings_save_requested = Signal(object)
    device_probe_requested = Signal()
    device_details_requested = Signal()

    def __init__(
        self,
        settings_manager: SettingsManager,
        settings: AppSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.settings = settings
        self.device_report: DeviceProbeReport | None = None
        self.resolved_device = ""

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

        device_card = SectionCard(
            "计算设备",
            "设备策略保存在配置中；实际设备来自当前模型会话，不会根据 CUDA 安装状态猜测。",
        )
        device_form = QFormLayout()
        self.device_combo = QComboBox()
        self.device_combo.addItem("自动选择（推荐）", "auto")
        self.device_combo.addItem("CPU", "cpu")
        if settings.device_preference.startswith("cuda:"):
            self.device_combo.addItem(
                f"{settings.device_preference.upper()}（等待检测）",
                settings.device_preference,
            )
        self.fallback_checkbox = QCheckBox("CUDA 失败时自动回退到 CPU")
        self.actual_device_label = QLabel("尚未加载模型")
        self.actual_device_label.setObjectName("settingsDeviceValue")
        self.cuda_status_label = QLabel("尚未检测")
        self.cuda_status_label.setWordWrap(True)
        device_form.addRow("设备策略", self.device_combo)
        device_form.addRow("", self.fallback_checkbox)
        device_form.addRow("实际设备", self.actual_device_label)
        device_form.addRow("CUDA 状态", self.cuda_status_label)
        device_card.content_layout.addLayout(device_form)
        device_actions = QHBoxLayout()
        self.probe_button = QPushButton("重新检测设备")
        self.probe_button.clicked.connect(
            lambda: self.device_probe_requested.emit()
        )
        details_button = QPushButton("查看设备详情")
        details_button.clicked.connect(
            lambda: self.device_details_requested.emit()
        )
        device_actions.addWidget(self.probe_button)
        device_actions.addWidget(details_button)
        device_actions.addStretch()
        device_card.content_layout.addLayout(device_actions)
        layout.addWidget(device_card)

        interface_card = SectionCard("界面设置")
        interface_form = QFormLayout()
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("浅色", "light")
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1180, 3840)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(720, 2160)
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
        self.save_button = QPushButton("保存设置")
        self.save_button.setObjectName("primaryButton")
        self.save_button.clicked.connect(self.save_settings)
        action_row.addWidget(restore_button)
        action_row.addWidget(self.save_button)
        layout.addLayout(action_row)
        layout.addStretch()

        scroll.setWidget(content)
        root_layout.addWidget(scroll)
        self._populate(settings)

    def save_settings(self) -> None:
        """Build a candidate configuration for the main-window device guard."""
        updated = self.settings.model_copy(
            update={
                "output_directory": Path(self.output_directory_edit.text().strip()),
                "model_directory": Path(self.model_directory_edit.text().strip()),
                "log_directory": Path(self.log_directory_edit.text().strip()),
                "device_preference": self.device_combo.currentData(),
                "allow_cpu_fallback": self.fallback_checkbox.isChecked(),
                "theme": self.theme_combo.currentData(),
                "window_width": self.width_spin.value(),
                "window_height": self.height_spin.value(),
            }
        )
        self.feedback_label.setText("正在应用设备设置…")
        self.settings_save_requested.emit(updated)

    def restore_defaults(self) -> None:
        """Request platform defaults through the same device-switch guard."""
        defaults = self.settings_manager.default_settings()
        self.settings_save_requested.emit(defaults)

    def commit_settings(self, settings: AppSettings, message: str = "设置已保存") -> None:
        """Persist settings after the main window authorizes the change."""
        try:
            self.settings_manager.save(settings)
        except ConfigurationError as exc:
            self.show_error(str(exc))
            raise
        self.settings = settings
        self._populate(settings)
        self.feedback_label.setProperty("error", False)
        self.feedback_label.setText(message)
        self.feedback_label.style().unpolish(self.feedback_label)
        self.feedback_label.style().polish(self.feedback_label)
        self.settings_saved.emit(settings)

    def apply_settings(self, settings: AppSettings) -> None:
        """Restore the visible form without persisting it."""
        self.settings = settings
        self._populate(settings)

    def show_error(self, message: str) -> None:
        """Show a concise, non-blocking settings error."""
        self.feedback_label.setText(message)
        self.feedback_label.setProperty("error", True)
        self.feedback_label.style().unpolish(self.feedback_label)
        self.feedback_label.style().polish(self.feedback_label)

    def set_probe_busy(self, busy: bool) -> None:
        """Lock device controls while background probing is active."""
        self.probe_button.setEnabled(not busy)
        self.save_button.setEnabled(not busy)
        self.device_combo.setEnabled(not busy)
        if busy:
            self.cuda_status_label.setText("正在后台执行 CUDA tensor 探测…")

    def set_device_switch_locked(self, locked: bool, message: str = "") -> None:
        """Prevent policy changes while a session or inference task owns the device."""
        self.device_combo.setEnabled(not locked)
        self.probe_button.setEnabled(not locked)
        self.save_button.setEnabled(not locked)
        if locked and message:
            self.feedback_label.setText(message)

    def set_device_report(
        self,
        report: DeviceProbeReport,
        *,
        resolved_device: str = "",
        cuda_status: str = "",
    ) -> None:
        """Populate detected CUDA choices while keeping CPU always selectable."""
        preference = self.device_combo.currentData() or self.settings.device_preference
        self.device_report = report
        self.resolved_device = resolved_device
        self.device_combo.clear()
        self.device_combo.addItem("自动选择（推荐）", "auto")
        self.device_combo.addItem("CPU", "cpu")
        known_ids = {"auto", "cpu"}
        model = self.device_combo.model()
        assert isinstance(model, QStandardItemModel)
        for device in report.devices:
            if device.type != "cuda":
                continue
            label = device.display_name
            if not device.available:
                label = f"{label} - 不可用"
            self.device_combo.addItem(label, device.device_id)
            known_ids.add(device.device_id)
            if not device.available:
                item = model.item(self.device_combo.count() - 1)
                if item is not None:
                    item.setEnabled(False)
        if preference.startswith("cuda:") and preference not in known_ids:
            self.device_combo.addItem(f"{preference.upper()} - 不可用", preference)
            item = model.item(self.device_combo.count() - 1)
            if item is not None:
                item.setEnabled(False)
        self._select_data(self.device_combo, preference)
        self.actual_device_label.setText(
            resolved_device if resolved_device else "尚未加载模型"
        )
        self.cuda_status_label.setText(cuda_status or report.error_message or "可用")
        self.set_probe_busy(False)

    def show_device_details(self, report: DeviceProbeReport) -> None:
        """Show safe diagnostics without exposing tracebacks."""
        lines = [
            f"PyTorch 版本：{report.torch_version or '未知'}",
            f"CUDA runtime：{report.cuda_runtime_version or '不可用'}",
            f"torch.cuda.is_available：{report.torch_cuda_available}",
            f"CUDA 设备数量：{report.cuda_device_count}",
            "",
        ]
        for device in report.devices:
            memory = (
                f"{device.total_memory / 1024**3:.2f} GiB"
                if device.total_memory is not None
                else "未知"
            )
            lines.extend(
                [
                    f"{device.display_name}",
                    f"  编号：{device.device_id}",
                    f"  状态：{'可用' if device.available else '不可用'}",
                    f"  完整探测：{'是' if device.tested else '否'}",
                    f"  显存：{memory if device.type == 'cuda' else '不适用'}",
                    f"  错误摘要：{device.error_message or '无'}",
                    "",
                ]
            )
        dialog = QDialog(self)
        dialog.setWindowTitle("计算设备详情")
        dialog.resize(620, 480)
        layout = QVBoxLayout(dialog)
        details = QPlainTextEdit()
        details.setReadOnly(True)
        details.setPlainText("\n".join(lines))
        close_button = QPushButton("关闭")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(details)
        layout.addWidget(close_button)
        dialog.exec()

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
        self._select_data(self.device_combo, settings.device_preference)
        self.fallback_checkbox.setChecked(settings.allow_cpu_fallback)
        self._select_data(self.theme_combo, settings.theme)
        self.width_spin.setValue(settings.window_width)
        self.height_spin.setValue(settings.window_height)

    @staticmethod
    def _select_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
