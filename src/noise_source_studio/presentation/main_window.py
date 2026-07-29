"""Top-level application window."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from noise_source_studio.infrastructure.config import AppSettings, SettingsManager
from noise_source_studio.presentation.navigation import NAVIGATION_ITEMS, NavigationSidebar
from noise_source_studio.presentation.pages import (
    BatchPredictionPage,
    DashboardPage,
    HistoryPage,
    LogPage,
    ModelManagementPage,
    SettingsPage,
    SinglePredictionPage,
    ValidationPage,
)
from noise_source_studio.version import APPLICATION_TITLE


class HeaderStatusUnit(QFrame):
    """Consistent top-bar status that supports runtime value updates."""

    def __init__(
        self,
        label: str,
        value: str,
        state: str = "neutral",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("headerStatus")
        self.setFixedHeight(44)
        self.setMinimumWidth(120)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 5, 12, 5)
        layout.setSpacing(0)

        label_widget = QLabel(label)
        label_widget.setObjectName("headerStatusLabel")
        self.value_label = QLabel()
        self.value_label.setObjectName("headerStatusValue")
        layout.addWidget(label_widget)
        layout.addWidget(self.value_label)
        self.set_status(value, state)

    def set_status(self, value: str, state: str = "neutral") -> None:
        """Update the displayed value and its semantic color state."""
        self.value_label.setText(value)
        self.value_label.setProperty("state", state)
        self.value_label.style().unpolish(self.value_label)
        self.value_label.style().polish(self.value_label)


class MainWindow(QMainWindow):
    """Commercial-style shell hosting all application workflows."""

    def __init__(
        self,
        settings: AppSettings,
        settings_manager: SettingsManager,
        log_file: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.header_statuses: dict[str, HeaderStatusUnit] = {}
        self.setObjectName("mainWindow")
        self.setWindowTitle(APPLICATION_TITLE)
        self.setMinimumSize(1180, 720)
        self.resize(settings.window_width, settings.window_height)

        central = QWidget()
        central.setObjectName("applicationShell")
        shell_layout = QVBoxLayout(central)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        shell_layout.addWidget(self._build_header())
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.navigation = NavigationSidebar()
        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("pageStack")
        self.pages = self._create_pages(settings_manager, log_file)
        for page in self.pages:
            self.page_stack.addWidget(page)

        body_layout.addWidget(self.navigation)
        body_layout.addWidget(self.page_stack, 1)
        shell_layout.addWidget(body, 1)
        self.setCentralWidget(central)

        self.navigation.page_selected.connect(self.set_current_page)
        dashboard = self.pages[0]
        if isinstance(dashboard, DashboardPage):
            dashboard.navigation_requested.connect(self.navigation.select_page)

        self._build_status_bar()
        self.set_current_page(0)

    @property
    def page_count(self) -> int:
        """Return the number of registered workflow pages."""
        return self.page_stack.count()

    def set_current_page(self, index: int) -> None:
        """Switch to a page while guarding invalid navigation indices."""
        if 0 <= index < self.page_stack.count():
            self.page_stack.setCurrentIndex(index)
            self.current_page_label.setText(NAVIGATION_ITEMS[index].label)

    def update_header_status(
        self,
        status_key: str,
        value: str,
        state: str = "neutral",
    ) -> None:
        """Update one top-bar status through the shared runtime interface."""
        try:
            status = self.header_statuses[status_key]
        except KeyError as exc:
            raise KeyError(f"Unknown header status: {status_key}") from exc
        status.set_status(value, state)

    def _create_pages(
        self,
        settings_manager: SettingsManager,
        log_file: Path,
    ) -> tuple[QWidget, ...]:
        return (
            DashboardPage(),
            SinglePredictionPage(),
            BatchPredictionPage(),
            ValidationPage(),
            ModelManagementPage(),
            HistoryPage(),
            LogPage(log_file),
            SettingsPage(settings_manager, self.settings),
        )

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("topHeader")
        header.setFixedHeight(64)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 22, 0)
        layout.setSpacing(12)

        brand_mark = QLabel("NS")
        brand_mark.setObjectName("brandMark")
        brand_mark.setFixedSize(34, 34)
        product = QWidget()
        product_layout = QVBoxLayout(product)
        product_layout.setContentsMargins(0, 0, 0, 0)
        product_layout.setSpacing(1)
        chinese_name = QLabel(self.settings.application_name)
        chinese_name.setObjectName("productName")
        english_name = QLabel("NOISE SOURCE STUDIO")
        english_name.setObjectName("productSubtitle")
        product_layout.addWidget(chinese_name)
        product_layout.addWidget(english_name)

        layout.addWidget(brand_mark)
        layout.addWidget(product)
        layout.addStretch()
        status_definitions = (
            ("model", "当前模型", "未配置", "warning"),
            ("device", "计算设备", "待检测", "neutral"),
            ("mode", "工作模式", "本地", "neutral"),
        )
        for key, label, value, state in status_definitions:
            status = HeaderStatusUnit(label, value, state)
            self.header_statuses[key] = status
            layout.addWidget(status)
        return header

    def _build_status_bar(self) -> None:
        status_bar = QStatusBar()
        status_bar.setObjectName("applicationStatusBar")
        status_bar.setSizeGripEnabled(False)
        self.setStatusBar(status_bar)

        version = QLabel(f"版本 {self.settings.application_version}")
        version.setObjectName("footerText")
        ready = QLabel("●  应用正常")
        ready.setObjectName("readyStatus")
        self.current_page_label = QLabel()
        self.current_page_label.setObjectName("footerText")
        self.clock_label = QLabel()
        self.clock_label.setObjectName("footerText")
        self.clock_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

        status_bar.addWidget(version)
        status_bar.addWidget(ready)
        status_bar.addPermanentWidget(self.current_page_label)
        status_bar.addPermanentWidget(self.clock_label)

        timer = QTimer(self)
        timer.timeout.connect(self._update_clock)
        timer.start(1000)
        self.clock_timer = timer
        self._update_clock()

    def _update_clock(self) -> None:
        self.clock_label.setText(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
