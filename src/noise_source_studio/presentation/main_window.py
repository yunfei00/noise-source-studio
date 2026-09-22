"""Top-level application window."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Any
from uuid import uuid4

from PySide6.QtCore import Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from noise_source_studio.common.exceptions import (
    ConfigurationError,
    DeviceSelectionError,
    ModelActivationError,
)
from noise_source_studio.domain.batch import (
    RUNNING_BATCH_STATUSES,
    BatchPredictionTask,
    BatchStatus,
)
from noise_source_studio.domain.device import DeviceProbeReport, DeviceResolution
from noise_source_studio.domain.history import HistoryQuery, TaskType
from noise_source_studio.domain.interfaces import InferenceEngine
from noise_source_studio.domain.models import LoadedModel, ModelRecord, PredictionOutcome
from noise_source_studio.domain.validation import (
    RUNNING_VALIDATION_STATUSES,
    ValidationTask,
)
from noise_source_studio.infrastructure.config import AppSettings, SettingsManager
from noise_source_studio.infrastructure.inference import (
    BatchWorker,
    RuntimeAdapter,
    ValidationWorker,
)
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
from noise_source_studio.services import (
    BatchPredictionService,
    DeviceService,
    HistoryService,
    ModelService,
    PredictionService,
    ValidationService,
)
from noise_source_studio.services.tasks import BackgroundTask, TaskFailure
from noise_source_studio.version import APPLICATION_TITLE

LOGGER = logging.getLogger("noise_source_studio.main_window")


class HeaderStatusUnit(QFrame):
    """Consistent top-bar status that supports runtime value updates."""

    clicked = Signal()

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
        self.setCursor(Qt.CursorShape.PointingHandCursor)
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

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Expose status units as keyboard-free navigation affordances."""
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.rect().contains(event.position().toPoint())
        ):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    """Commercial-style shell hosting all application workflows."""

    history_scan_progress = Signal(int, int, str)

    def __init__(
        self,
        settings: AppSettings,
        settings_manager: SettingsManager,
        log_file: Path,
        engine: InferenceEngine | None = None,
        model_service: ModelService | None = None,
        prediction_service: PredictionService | None = None,
        batch_prediction_service: BatchPredictionService | None = None,
        validation_service: ValidationService | None = None,
        device_service: DeviceService | None = None,
        history_service: HistoryService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.settings_manager = settings_manager
        self.header_statuses: dict[str, HeaderStatusUnit] = {}
        self.engine = engine or RuntimeAdapter()
        self.device_service = device_service or DeviceService(self.engine)
        self.model_service = model_service or ModelService(
            settings.model_directory,
            self.engine,
        )
        self.prediction_service = prediction_service or PredictionService(
            self.engine,
            settings.output_directory,
        )
        self.batch_prediction_service = batch_prediction_service or BatchPredictionService(
            settings.output_directory,
        )
        self.validation_service = validation_service or ValidationService(
            settings.output_directory,
        )
        self.history_service: HistoryService | None = history_service
        self.history_error = ""
        if self.history_service is None:
            try:
                self.history_service = HistoryService(
                    settings_manager.paths.history_database,
                    settings.output_directory,
                    model_root=settings.model_directory,
                )
            except Exception as exc:
                self.history_error = str(exc)
                LOGGER.exception("History index unavailable")
        self._history_scan_cancel: Event | None = None
        self.batch_task = self.batch_prediction_service.create_task()
        self.batch_worker: BatchWorker | None = None
        self.validation_task: ValidationTask | None = None
        self.validation_worker: ValidationWorker | None = None
        self.loaded_model: LoadedModel | None = None
        self.device_report: DeviceProbeReport | None = None
        self.device_resolution: DeviceResolution | None = None
        self._model_loading = False
        self._model_load_generation = 0
        self._single_prediction_running = False
        self.task_pool = QThreadPool(self)
        self.task_pool.setMaxThreadCount(1)
        self._tasks: set[BackgroundTask] = set()
        self._shutting_down = False
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

        self._connect_runtime_pages()
        self._build_status_bar()
        self._refresh_model_views()
        self.set_current_page(0)
        if self.history_service is not None:
            QTimer.singleShot(0, self._history_page.request_query)
        else:
            self._history_page.set_unavailable(self.history_error or "未知错误")
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)
        if self.model_service.active_model() is not None:
            QTimer.singleShot(0, self._restore_active_model)
        elif self.settings.device_preference != "cpu":
            QTimer.singleShot(0, self._probe_devices)
        else:
            self.device_report = self.device_service.cpu_report()
            self._update_settings_device_state()

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

    def shutdown(self) -> None:
        """Wait briefly for background work and always close the retained session."""
        if self._shutting_down:
            return
        self._shutting_down = True
        self.task_pool.waitForDone()
        self.engine.close()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Release runtime resources before the window is destroyed."""
        validation_running = bool(
            self.validation_task and self.validation_task.status in RUNNING_VALIDATION_STATUSES
        )
        if self.batch_task.status in RUNNING_BATCH_STATUSES or validation_running:
            answer = QMessageBox.question(
                self,
                "后台任务仍在运行",
                "退出将等待当前文件完成并安全停止任务。确定退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            if self.batch_worker is not None:
                self.batch_worker.request_stop()
            if self.validation_worker is not None:
                self.validation_worker.request_stop()
        self.shutdown()
        super().closeEvent(event)

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
            ("model", "当前模型", "未配置", "warning", 220),
            ("device", "计算设备", "待检测", "neutral", 220),
            ("mode", "工作模式", "本地", "neutral", 120),
        )
        for key, label, value, state, minimum_width in status_definitions:
            status = HeaderStatusUnit(label, value, state)
            status.setMinimumWidth(minimum_width)
            self.header_statuses[key] = status
            layout.addWidget(status)
        self.header_statuses["device"].setToolTip("点击打开系统设置中的计算设备配置。")
        return header

    def _build_status_bar(self) -> None:
        status_bar = QStatusBar()
        status_bar.setObjectName("applicationStatusBar")
        status_bar.setSizeGripEnabled(False)
        self.setStatusBar(status_bar)

        version = QLabel(f"版本 {self.settings.application_version}")
        version.setObjectName("footerText")
        self.application_state_label = QLabel("●  应用正常")
        self.application_state_label.setObjectName("readyStatus")
        self.current_page_label = QLabel()
        self.current_page_label.setObjectName("footerText")
        self.clock_label = QLabel()
        self.clock_label.setObjectName("footerText")
        self.clock_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

        status_bar.addWidget(version)
        status_bar.addWidget(self.application_state_label)
        status_bar.addPermanentWidget(self.current_page_label)
        status_bar.addPermanentWidget(self.clock_label)

        timer = QTimer(self)
        timer.timeout.connect(self._update_clock)
        timer.start(1000)
        self.clock_timer = timer
        self._update_clock()

    def _update_clock(self) -> None:
        self.clock_label.setText(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))

    def _connect_runtime_pages(self) -> None:
        model_page = self._model_page
        model_page.package_selected.connect(self._inspect_model_package)
        model_page.package_import_requested.connect(self._import_model_package)
        model_page.activation_requested.connect(self._activate_model)
        model_page.deletion_requested.connect(self._delete_model)
        model_page.integrity_requested.connect(self._check_model_integrity)

        prediction_page = self._single_prediction_page
        prediction_page.preview_requested.connect(self._preview_file)
        prediction_page.prediction_requested.connect(self._predict_file)
        prediction_page.export_requested.connect(self._export_prediction)

        batch_page = self._batch_prediction_page
        batch_page.set_task(self.batch_task)
        batch_page.paths_added.connect(self._batch_add_paths)
        batch_page.remove_requested.connect(self._batch_remove_items)
        batch_page.clear_requested.connect(self._batch_clear)
        batch_page.deduplicate_requested.connect(self._batch_deduplicate)
        batch_page.start_requested.connect(self._start_batch)
        batch_page.pause_requested.connect(self._pause_batch)
        batch_page.resume_requested.connect(self._resume_batch)
        batch_page.stop_requested.connect(self._stop_batch)
        batch_page.retry_requested.connect(self._retry_batch)
        batch_page.export_requested.connect(self._export_batch_copy)
        batch_page.filtered_export_requested.connect(self._export_filtered_batch)
        batch_page.history_requested.connect(self._open_batch_history)
        batch_page.waveform_requested.connect(self._preview_batch_result)
        batch_page.logs_requested.connect(lambda: self.navigation.select_page(6))

        validation_page = self._validation_page
        validation_page.manifest_check_requested.connect(self._check_validation_manifest)
        validation_page.start_requested.connect(self._start_validation)
        validation_page.pause_requested.connect(self._pause_validation)
        validation_page.resume_requested.connect(self._resume_validation)
        validation_page.stop_requested.connect(self._stop_validation)
        validation_page.history_requested.connect(self._open_validation_history)
        validation_page.batch_reuse_requested.connect(self._validate_existing_batch)
        validation_page.waveform_requested.connect(self._preview_validation_sample)
        validation_page.export_requested.connect(self._export_validation_copy)

        history_page = self._history_page
        history_page.query_requested.connect(self._query_history)
        history_page.open_task_requested.connect(self._open_history_task)
        history_page.open_directory_requested.connect(self._open_history_directory)
        history_page.verify_requested.connect(self._verify_history_task)
        history_page.notes_requested.connect(self._update_history_notes)
        history_page.delete_requested.connect(self._delete_history_task)
        history_page.export_requested.connect(self._export_history_summary)
        history_page.scan_requested.connect(self._scan_history_outputs)
        history_page.scan_cancel_requested.connect(self._cancel_history_scan)
        history_page.rebuild_requested.connect(self._rebuild_history_index)
        history_page.backup_requested.connect(self._backup_history_database)
        history_page.database_directory_requested.connect(
            self._open_history_database_directory
        )
        self.history_scan_progress.connect(history_page.show_scan_progress)

        settings_page = self._settings_page
        settings_page.settings_save_requested.connect(self._save_settings)
        settings_page.device_probe_requested.connect(self._probe_devices)
        settings_page.device_details_requested.connect(self._show_device_details)
        self.header_statuses["device"].clicked.connect(
            lambda: self.navigation.select_page(7)
        )

    @property
    def _dashboard_page(self) -> DashboardPage:
        page = self.pages[0]
        assert isinstance(page, DashboardPage)
        return page

    @property
    def _single_prediction_page(self) -> SinglePredictionPage:
        page = self.pages[1]
        assert isinstance(page, SinglePredictionPage)
        return page

    @property
    def _batch_prediction_page(self) -> BatchPredictionPage:
        page = self.pages[2]
        assert isinstance(page, BatchPredictionPage)
        return page

    @property
    def _validation_page(self) -> ValidationPage:
        page = self.pages[3]
        assert isinstance(page, ValidationPage)
        return page

    @property
    def _model_page(self) -> ModelManagementPage:
        page = self.pages[4]
        assert isinstance(page, ModelManagementPage)
        return page

    @property
    def _history_page(self) -> HistoryPage:
        page = self.pages[5]
        assert isinstance(page, HistoryPage)
        return page

    @property
    def _settings_page(self) -> SettingsPage:
        page = self.pages[7]
        assert isinstance(page, SettingsPage)
        return page

    def _query_history(self, query: HistoryQuery) -> None:
        service = self.history_service
        if service is None:
            self._history_page.set_unavailable(self.history_error or "历史服务未初始化")
            return
        self._run_task(
            lambda: (service.list_tasks(query), service.overview()),
            "历史任务查询",
            lambda payload: self._history_page.show_results(payload[0], payload[1]),
            self._history_page.show_error,
        )

    def _open_history_task(self, task_id: str) -> None:
        service = self.history_service
        if service is None:
            return
        page = self._history_page
        page.loading_label.setText("正在读取任务结果，不会重新执行模型推理…")

        def load() -> tuple[TaskType, object]:
            record = service.get_task(task_id)
            if record is None:
                raise ValueError("历史任务不存在或已被删除。")
            if record.integrity_status.value in {"missing", "corrupt"}:
                service.verify_task(task_id)
            if record.task_type == TaskType.SINGLE:
                payload: object = service.load_single_prediction(task_id)
            elif record.task_type == TaskType.BATCH:
                payload = self.batch_prediction_service.load_history(
                    Path(record.result_directory)
                )
            else:
                payload = self.validation_service.load_history(Path(record.result_directory))
            service.mark_opened(task_id)
            return record.task_type, payload

        def loaded(payload: tuple[TaskType, object]) -> None:
            task_type, result = payload
            if task_type == TaskType.SINGLE:
                assert isinstance(result, PredictionOutcome)
                self._single_prediction_page.show_history(result)
                self.navigation.select_page(1)
            elif task_type == TaskType.BATCH:
                assert isinstance(result, BatchPredictionTask)
                self.batch_task = result
                self._batch_prediction_page.show_history(result)
                self.navigation.select_page(2)
            else:
                assert isinstance(result, ValidationTask)
                self.validation_task = result
                self._validation_page.show_history(result)
                self.navigation.select_page(3)

        self._run_task(load, "打开历史任务", loaded, page.show_error)

    def _open_history_directory(self, task_id: str) -> None:
        service = self.history_service
        if service is None:
            return

        def opened(record: object) -> None:
            if record is None or not getattr(record, "result_directory", ""):
                self._history_page.show_error("该任务没有可打开的结果目录。")
                return
            path = Path(record.result_directory)
            if not path.is_dir():
                self._history_page.show_error("结果目录不存在，请先执行完整性校验。")
                return
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

        self._run_task(
            lambda: service.get_task(task_id),
            "读取历史任务目录",
            opened,
            self._history_page.show_error,
        )

    def _verify_history_task(self, task_id: str) -> None:
        service = self.history_service
        if service is None:
            return
        self._run_task(
            lambda: service.verify_task(task_id),
            "历史文件完整性校验",
            lambda status: self._history_page.show_operation_result(
                f"文件校验完成：{status.value}"
            ),
            self._history_page.show_error,
        )

    def _update_history_notes(
        self,
        task_id: str,
        notes: str,
        tags: object,
    ) -> None:
        service = self.history_service
        if service is None:
            return
        normalized_tags = tuple(str(value) for value in tags)
        self._run_task(
            lambda: service.update_notes(task_id, notes, normalized_tags),
            "保存历史备注",
            lambda _changed: self._history_page.show_operation_result("备注与标签已保存"),
            self._history_page.show_error,
        )

    def _delete_history_task(self, task_id: str, with_files: bool) -> None:
        service = self.history_service
        if service is None:
            return

        def confirm(record: object) -> None:
            if record is None:
                self._history_page.show_error("历史任务不存在或已被删除。")
                return
            result_directory = getattr(record, "result_directory", "")
            detail = (
                f"\n\n将同时永久删除：\n{result_directory}"
                if with_files
                else "\n\n结果文件将保留，可通过扫描重新建立索引。"
            )
            answer = QMessageBox.warning(
                self,
                "确认删除历史任务",
                f"确定删除任务 {task_id} 的索引吗？{detail}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            operation = (
                (lambda: service.delete_task_files(task_id))
                if with_files
                else (lambda: service.delete_index(task_id))
            )
            self._run_task(
                operation,
                "删除历史任务",
                lambda _deleted: self._history_page.show_operation_result("历史任务已删除"),
                self._history_page.show_error,
            )

        self._run_task(
            lambda: service.get_task(task_id),
            "读取待删除任务",
            confirm,
            self._history_page.show_error,
        )

    def _export_history_summary(self, query: HistoryQuery, destination: Path) -> None:
        service = self.history_service
        if service is None:
            return
        self._run_task(
            lambda: service.export_summary(query, destination),
            "导出历史任务摘要",
            lambda path: self._history_page.show_operation_result(
                f"筛选摘要已导出：{path}", refresh=False
            ),
            self._history_page.show_error,
        )

    def _scan_history_outputs(self) -> None:
        service = self.history_service
        if service is None:
            return
        self._history_scan_cancel = Event()
        cancel_event = self._history_scan_cancel
        self._history_page.loading_label.setText("正在后台扫描已有结果目录…")

        def scanned(report: object) -> None:
            self._history_scan_cancel = None
            message = (
                f"扫描完成：新增 {report.imported_tasks}，更新 {report.updated_tasks}，"
                f"损坏 {len(report.corrupted_directories)}"
            )
            self._history_page.show_operation_result(message)

        self._run_task(
            lambda: service.scan_outputs(
                cancel_event=cancel_event,
                progress=lambda current, total, path: self.history_scan_progress.emit(
                    current, total, path
                ),
            ),
            "扫描历史结果",
            scanned,
            self._history_page.show_error,
            lambda: setattr(self, "_history_scan_cancel", None),
        )

    def _cancel_history_scan(self) -> None:
        if self._history_scan_cancel is not None:
            self._history_scan_cancel.set()
            self._history_page.loading_label.setText("正在取消扫描…")

    def _rebuild_history_index(self) -> None:
        service = self.history_service
        if service is None:
            return
        answer = QMessageBox.question(
            self,
            "重建历史索引",
            "将先备份当前数据库，再从结果目录构建临时索引并原子替换。继续吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._history_scan_cancel = Event()
        cancel_event = self._history_scan_cancel
        self._run_task(
            lambda: service.rebuild_index(
                cancel_event=cancel_event,
                progress=lambda current, total, path: self.history_scan_progress.emit(
                    current, total, path
                ),
            ),
            "重建历史索引",
            lambda payload: self._history_page.show_operation_result(
                f"索引重建完成，原数据库备份至：{payload[0]}"
            ),
            self._history_page.show_error,
            lambda: setattr(self, "_history_scan_cancel", None),
        )

    def _backup_history_database(self) -> None:
        service = self.history_service
        if service is None:
            return
        self._run_task(
            service.backup_database,
            "备份历史数据库",
            lambda path: self._history_page.show_operation_result(
                f"数据库备份完成：{path}", refresh=False
            ),
            self._history_page.show_error,
        )

    def _open_history_database_directory(self) -> None:
        service = self.history_service
        if service is not None:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(service.database_path.parent.resolve()))
            )

    def _run_task(
        self,
        operation: Any,
        description: str,
        on_success: Any,
        on_error: Any,
        on_finished: Any | None = None,
    ) -> None:
        if self._shutting_down:
            return
        task = BackgroundTask(operation, description)
        self._tasks.add(task)
        task.signals.succeeded.connect(on_success)
        task.signals.failed.connect(
            lambda failure, context=description, task_id=task.task_id: self._dispatch_task_error(
                context,
                task_id,
                failure,
                on_error,
            )
        )

        def finish() -> None:
            self._tasks.discard(task)
            if on_finished is not None:
                on_finished()

        task.signals.finished.connect(finish)
        self.task_pool.start(task)

    def _dispatch_task_error(
        self,
        context: str,
        task_id: str,
        failure: TaskFailure,
        handler: Any,
    ) -> None:
        LOGGER.error(
            "Background task failed | task=%s | task_id=%s | exception=%s | traceback=%s",
            context,
            task_id,
            type(failure.exception).__name__,
            failure.traceback_text,
        )
        handler(str(failure.exception))

    def _inspect_model_package(self, package_path: Path) -> None:
        page = self._model_page
        page.set_busy(True, "正在后台校验模型包…")
        self._run_task(
            lambda: self.model_service.inspect_package(package_path),
            "模型包校验",
            page.show_package_inspection,
            lambda message: page.show_feedback(message, error=True),
            lambda: page.set_busy(False),
        )

    def _import_model_package(self, package_path: Path) -> None:
        page = self._model_page
        page.set_busy(True, "正在复制并二次校验模型包…")

        def imported(record: ModelRecord) -> None:
            self._refresh_model_views()
            page.show_feedback(f"模型已导入：{record.display_name}", error=False)

        self._run_task(
            lambda: self.model_service.import_package(package_path),
            "模型包导入",
            imported,
            lambda message: page.show_feedback(message, error=True),
            lambda: page.set_busy(False),
        )

    def _activate_model(self, identifier: str) -> None:
        if reason := self._device_switch_block_reason():
            self._model_page.show_feedback(reason, error=True)
            return
        self._begin_model_load(identifier, startup=False)

    def _restore_active_model(self) -> None:
        record = self.model_service.active_model()
        if record is None:
            return
        self._begin_model_load(record.identifier, startup=True)

    def _begin_model_load(
        self,
        identifier: str,
        *,
        startup: bool,
        device_preference: str | None = None,
    ) -> None:
        """Resolve policy and load a candidate model session in the background."""
        preference = device_preference or self.settings.device_preference
        allow_fallback = self.settings.allow_cpu_fallback
        page = self._model_page
        self._model_load_generation += 1
        generation = self._model_load_generation
        self._model_loading = True
        page.set_busy(True, "正在后台检测设备并加载模型…")
        self._settings_page.set_device_switch_locked(
            True,
            "模型正在加载，暂时不能切换计算设备。",
        )
        self._set_application_state("模型加载中")
        self.update_header_status("model", "加载中", "warning")

        def activated(result: tuple[LoadedModel, DeviceResolution]) -> None:
            loaded, resolution = result
            self.device_resolution = resolution
            self.device_report = resolution.report
            self._apply_loaded_model(loaded)
            self._refresh_model_views()
            page.show_feedback(f"模型已激活：{loaded.record.display_name}", error=False)
            if resolution.fallback_reason:
                message = "CUDA 不可用，当前已使用 CPU。"
                self.statusBar().showMessage(message, 10000)
                self.header_statuses["device"].setToolTip(
                    f"{message}\n{resolution.fallback_reason}"
                )
                self._settings_page.feedback_label.setText(
                    f"{message} {resolution.fallback_reason}"
                )

        def failed(message: str) -> None:
            report = self.device_service.last_report
            if report is not None:
                self.device_report = report
            self._refresh_model_views()
            page.show_feedback(message, error=True)
            self._show_preserved_or_failed_session(preference, message)
            if preference.startswith("cuda:") and allow_fallback:
                self._offer_explicit_cpu_fallback(identifier, preference, message)
            elif not startup and preference != "auto":
                QMessageBox.warning(self, "模型加载失败", message)

        def finished() -> None:
            if generation != self._model_load_generation:
                return
            self._model_loading = False
            page.set_busy(False)
            self._settings_page.set_device_switch_locked(False)
            self._update_settings_device_state()

        self._run_task(
            lambda: self._load_model_with_policy(
                identifier,
                preference,
                allow_cpu_fallback=allow_fallback,
            ),
            "启动恢复活动模型" if startup else "模型激活",
            activated,
            failed,
            finished,
        )

    def _load_model_with_policy(
        self,
        identifier: str,
        preference: str,
        *,
        allow_cpu_fallback: bool,
    ) -> tuple[LoadedModel, DeviceResolution]:
        """Resolve a concrete device and apply automatic CPU fallback."""
        resolution = self.device_service.resolve(preference)
        try:
            loaded = self.model_service.activate_model(
                identifier,
                device=resolution.resolved_device,
            )
        except (DeviceSelectionError, ModelActivationError) as exc:
            if (
                preference == "auto"
                and resolution.resolved_device.startswith("cuda:")
                and allow_cpu_fallback
            ):
                reason = f"CUDA 模型加载失败：{exc}"
                LOGGER.exception(
                    "Automatic CUDA model load failed; retrying on CPU | device=%s",
                    resolution.resolved_device,
                )
                cpu_resolution = DeviceResolution(
                    device_preference="auto",
                    resolved_device="cpu",
                    report=resolution.report,
                    fallback_reason=reason,
                )
                loaded = self.model_service.activate_model(identifier, device="cpu")
                LOGGER.info("Automatic CPU fallback model load succeeded")
                return loaded, cpu_resolution
            raise
        actual_resolution = DeviceResolution(
            device_preference=resolution.device_preference,
            resolved_device=loaded.device,
            report=resolution.report,
            fallback_reason=resolution.fallback_reason,
        )
        LOGGER.info(
            "Model loaded on resolved device | preference=%s | resolved=%s",
            preference,
            loaded.device,
        )
        return loaded, actual_resolution

    def _show_preserved_or_failed_session(
        self,
        preference: str,
        message: str,
    ) -> None:
        """Keep an old usable session visible, or expose a truthful failed state."""
        if self.loaded_model is not None:
            self._apply_loaded_model(self.loaded_model)
            self.header_statuses["device"].setToolTip(
                f"新设备加载失败，仍在使用原会话。\n{message}"
            )
            self._set_application_state("正常")
            return
        self._clear_loaded_model(load_failed=True)
        if preference.startswith("cuda:"):
            self.update_header_status(
                "device",
                f"{preference.upper()} 不可用",
                "warning",
            )
        self.header_statuses["device"].setToolTip(message)

    def _offer_explicit_cpu_fallback(
        self,
        identifier: str,
        preference: str,
        message: str,
    ) -> None:
        choice = self._ask_cpu_fallback(preference, message)
        if choice == "logs":
            self.navigation.select_page(6)
            return
        if choice != "cpu":
            LOGGER.info(
                "User cancelled explicit CUDA CPU fallback | device=%s",
                preference,
            )
            return
        updated = self.settings.model_copy(update={"device_preference": "cpu"})
        try:
            self._settings_page.commit_settings(
                updated,
                "已切换设备策略为 CPU，正在重新加载模型…",
            )
        except ConfigurationError:
            return
        self.settings = updated
        LOGGER.info(
            "User confirmed CPU fallback | failed_device=%s | model=%s",
            preference,
            identifier,
        )
        QTimer.singleShot(
            0,
            lambda: self._begin_model_load(
                identifier,
                startup=False,
                device_preference="cpu",
            ),
        )

    def _ask_cpu_fallback(self, preference: str, message: str) -> str:
        """Return ``cpu``, ``cancel``, or ``logs`` from an explicit CUDA prompt."""
        dialog = QMessageBox(self)
        dialog.setWindowTitle("CUDA 设备加载失败")
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setText(
            f"{preference.upper()} 加载失败，是否切换到 CPU 重新加载当前模型？"
        )
        dialog.setInformativeText(message)
        cpu_button = dialog.addButton(
            "切换到 CPU",
            QMessageBox.ButtonRole.AcceptRole,
        )
        cancel_button = dialog.addButton(
            "取消",
            QMessageBox.ButtonRole.RejectRole,
        )
        logs_button = dialog.addButton(
            "查看日志",
            QMessageBox.ButtonRole.ActionRole,
        )
        dialog.setDefaultButton(cpu_button)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is cpu_button:
            return "cpu"
        if clicked is logs_button:
            return "logs"
        assert clicked is cancel_button or clicked is None
        return "cancel"

    def _save_settings(self, updated: AppSettings) -> None:
        """Persist settings only after device-switch task guards pass."""
        preference_changed = (
            updated.device_preference != self.settings.device_preference
        )
        if preference_changed and (reason := self._device_switch_block_reason()):
            self._settings_page.apply_settings(self.settings)
            self._settings_page.show_error(reason)
            return
        try:
            self._settings_page.commit_settings(updated)
        except ConfigurationError:
            return
        previous_preference = self.settings.device_preference
        self.settings = updated
        LOGGER.info(
            "User device preference saved | previous=%s | current=%s | "
            "allow_cpu_fallback=%s",
            previous_preference,
            updated.device_preference,
            updated.allow_cpu_fallback,
        )
        if not preference_changed:
            self._update_settings_device_state()
            return
        active = self.model_service.active_model()
        if active is not None and self.loaded_model is not None:
            self._begin_model_load(
                active.identifier,
                startup=False,
                device_preference=updated.device_preference,
            )
        elif updated.device_preference == "cpu":
            self.device_report = self.device_service.cpu_report()
            self._update_settings_device_state()
        else:
            self._probe_devices()

    def _probe_devices(self) -> None:
        """Detect CUDA devices in the background without changing the session."""
        if reason := self._device_switch_block_reason():
            self._settings_page.show_error(reason)
            return
        page = self._settings_page
        page.set_probe_busy(True)

        def completed(report: DeviceProbeReport) -> None:
            self.device_report = report
            self._update_settings_device_state()
            LOGGER.info("Device details refreshed in settings")

        self._run_task(
            self.device_service.probe_devices,
            "计算设备探测",
            completed,
            page.show_error,
            lambda: page.set_probe_busy(False),
        )

    def _show_device_details(self) -> None:
        report = self.device_report or self.device_service.cpu_report()
        self._settings_page.show_device_details(report)

    def _update_settings_device_state(self) -> None:
        report = self.device_report
        if report is None:
            return
        resolved_display = (
            self.device_service.display_name(self.loaded_model.device, report)
            if self.loaded_model is not None
            else ""
        )
        self._settings_page.set_device_report(
            report,
            resolved_device=resolved_display,
            cuda_status=self.device_service.cuda_status(report),
        )

    def _device_switch_block_reason(self) -> str:
        if self._model_loading:
            return "模型正在加载，请等待加载完成后再切换计算设备。"
        if self._single_prediction_running:
            return "当前推理任务正在使用计算设备，请完成或停止任务后再切换。"
        if self.batch_task.status in RUNNING_BATCH_STATUSES:
            return "当前批量任务正在使用计算设备，请完成或停止任务后再切换。"
        if (
            self.validation_task is not None
            and self.validation_task.status in RUNNING_VALIDATION_STATUSES
        ):
            return "当前模型验证正在使用计算设备，请完成或停止任务后再切换。"
        return ""

    def _delete_model(self, identifier: str) -> None:
        page = self._model_page
        page.set_busy(True, "正在删除未激活模型…")

        def deleted(_: object) -> None:
            self._refresh_model_views()
            page.show_feedback("模型已删除。", error=False)

        self._run_task(
            lambda: self.model_service.delete_model(identifier),
            "模型删除",
            deleted,
            lambda message: page.show_feedback(message, error=True),
            lambda: page.set_busy(False),
        )

    def _check_model_integrity(self, identifier: str) -> None:
        page = self._model_page
        page.set_busy(True, "正在后台执行完整性校验…")

        def checked(_: object) -> None:
            self._refresh_model_views()
            page.show_feedback("完整性校验通过。", error=False)

        self._run_task(
            lambda: self.model_service.check_integrity(identifier),
            "模型完整性校验",
            checked,
            lambda message: page.show_feedback(message, error=True),
            lambda: page.set_busy(False),
        )

    def _preview_file(self, source_path: Path) -> None:
        page = self._single_prediction_page
        self._run_task(
            lambda: self.prediction_service.preview_file(source_path),
            "CSV 信号解析",
            page.show_preview,
            page.show_preview_error,
        )

    def _predict_file(self, source_path: Path) -> None:
        record = self.model_service.active_model()
        if record is None or self.loaded_model is None:
            self._single_prediction_page.show_prediction_error("当前没有已加载的活动模型。")
            return
        locked_record = record
        task_id = uuid4().hex
        page = self._single_prediction_page
        self._single_prediction_running = True
        self._settings_page.set_device_switch_locked(
            True,
            "当前推理任务正在使用计算设备，请完成后再切换。",
        )
        page.begin_prediction()
        self._set_application_state("推理中")

        def completed(outcome: PredictionOutcome) -> None:
            page.show_prediction(outcome)
            self._set_application_state("正常")

        def failed(message: str) -> None:
            page.show_prediction_error(message)
            self._set_application_state("推理失败")

        def finished() -> None:
            self._single_prediction_running = False
            self._settings_page.set_device_switch_locked(False)

        def predict_and_record() -> PredictionOutcome:
            service = self.history_service
            if service is not None:
                self._record_history_safely(
                    lambda: service.register_single_running(
                        task_id,
                        source_path,
                        locked_record,
                    )
                )
            try:
                outcome = self.prediction_service.predict_file(
                    source_path,
                    locked_record,
                    task_id=task_id,
                )
            except Exception as exc:
                if service is not None:
                    self._record_history_safely(
                        lambda error=exc: service.fail_task(task_id, error)
                    )
                raise
            if service is not None:
                self._record_history_safely(lambda: service.complete_single(outcome))
            return outcome

        self._run_task(
            predict_and_record,
            "单文件推理",
            completed,
            failed,
            finished,
        )

    def _export_prediction(self, include_contract: bool) -> None:
        outcome = self._single_prediction_page.current_outcome
        if outcome is None:
            return
        page = self._single_prediction_page
        self._run_task(
            lambda: self.prediction_service.export_result(
                outcome,
                include_contract=include_contract,
            ),
            "结果导出",
            lambda paths: page.show_export_result(paths[0], paths[1]),
            page.show_export_error,
        )

    def _apply_loaded_model(self, loaded: LoadedModel) -> None:
        self.loaded_model = loaded
        device_display = self.device_service.display_name(
            loaded.device,
            self.device_report,
        )
        self.update_header_status("model", loaded.record.display_name, "active")
        self.update_header_status("device", device_display, "active")
        self.update_header_status("mode", "本地", "neutral")
        dashboard = self._dashboard_page
        dashboard.status_cards["model"].set_status(
            loaded.record.display_name,
            f"prediction mode：{loaded.prediction_mode}",
        )
        dashboard.status_cards["device"].set_status(
            device_display,
            f"runtime {loaded.runtime_version}",
        )
        self._single_prediction_page.set_model(loaded)
        self._batch_prediction_page.set_model_available(True)
        self._validation_page.set_model(loaded)
        self._model_page.set_active_model(loaded.record, device_display)
        self._update_settings_device_state()
        self._set_application_state("正常")

    def _clear_loaded_model(self, *, load_failed: bool = False) -> None:
        self.loaded_model = None
        self.engine.close()
        self.update_header_status(
            "model",
            "加载失败" if load_failed else "未配置",
            "warning",
        )
        self.update_header_status("device", "待检测", "neutral")
        dashboard = self._dashboard_page
        dashboard.status_cards["model"].set_status(
            "加载失败" if load_failed else "未配置",
            "请在模型管理中检查活动模型",
        )
        dashboard.status_cards["device"].set_status("待检测", "尚无可用模型会话")
        self._single_prediction_page.set_model(None)
        self._batch_prediction_page.set_model_available(False)
        self._validation_page.set_model(None)
        self._set_application_state("正常")

    def _set_application_state(self, state: str) -> None:
        notes = {
            "正常": "基础服务与模型会话状态正常",
            "模型加载中": "正在后台校验并创建模型会话",
            "推理中": "正在后台执行单文件推理",
            "批量推理中": "正在复用当前模型会话顺序执行批量任务",
            "批量已暂停": "当前文件已完成，等待用户继续或停止",
            "模型验证中": "正在复用当前模型会话执行清单验证与指标计算",
            "验证已暂停": "当前样本已完成，等待用户继续或停止",
            "推理失败": "最近一次推理失败，请查看系统日志",
        }
        self._dashboard_page.status_cards["application"].set_status(
            state,
            notes.get(state, ""),
        )
        self.application_state_label.setText(f"●  应用{state}")
        self.application_state_label.setProperty(
            "state",
            "error" if state == "推理失败" else "active",
        )
        self.application_state_label.style().unpolish(self.application_state_label)
        self.application_state_label.style().polish(self.application_state_label)

    def _refresh_model_views(self) -> None:
        records = self.model_service.list_models()
        self._model_page.set_models(records)

    def _batch_add_paths(self, paths: list[Path], recursive: bool) -> None:
        page = self._batch_prediction_page
        page.show_feedback("正在扫描文件…")
        self._run_task(
            lambda: self.batch_prediction_service.add_paths(
                self.batch_task,
                paths,
                recursive=recursive,
            ),
            "批量文件扫描",
            page.show_scan_summary,
            lambda message: page.show_feedback(message, error=True),
        )

    def _batch_remove_items(self, item_ids: list[str]) -> None:
        self.batch_prediction_service.remove_items(self.batch_task, item_ids)
        self._batch_prediction_page.refresh_table()

    def _batch_clear(self) -> None:
        self.batch_prediction_service.clear_items(self.batch_task)
        self._batch_prediction_page.refresh_table()

    def _batch_deduplicate(self) -> None:
        removed = self.batch_prediction_service.deduplicate(self.batch_task)
        page = self._batch_prediction_page
        page.show_feedback(f"已移除 {removed} 个重复文件。")
        page.refresh_table()

    def _start_batch(self) -> None:
        if self.validation_task and self.validation_task.status in RUNNING_VALIDATION_STATUSES:
            self._batch_prediction_page.show_feedback(
                "模型验证运行期间不能同时启动批量预测。",
                error=True,
            )
            return
        if self.loaded_model is None:
            self._batch_prediction_page.show_feedback(
                "请先在模型管理中激活模型。",
                error=True,
            )
            return
        if self.batch_task.status == BatchStatus.STOPPED:
            self.batch_prediction_service.prepare_remaining(self.batch_task)
        if not any(item.status.value == "pending" for item in self.batch_task.items):
            self._batch_prediction_page.show_feedback("队列中没有待处理文件。", error=True)
            return
        locked_identifier = (
            f"{self.batch_task.model_name}@{self.batch_task.model_version}"
            if self.batch_task.model_name
            else ""
        )
        if locked_identifier and locked_identifier != self.loaded_model.record.identifier:
            self._batch_prediction_page.show_feedback(
                f"该批次已锁定模型 {locked_identifier}，请重新激活此模型后继续；"
                "如需使用新模型，请先清空队列创建新批次。",
                error=True,
            )
            return
        if not locked_identifier:
            self.batch_prediction_service.lock_model(self.batch_task, self.loaded_model)
        if self.history_service is not None:
            self._record_history_safely(
                lambda: self.history_service.register_batch_running(self.batch_task)
            )
        worker = BatchWorker(self.engine, self.batch_task)
        self.batch_worker = worker
        page = self._batch_prediction_page
        worker.signals.batch_started.connect(page.batch_started)
        worker.signals.item_started.connect(page.refresh_item)
        worker.signals.item_progress.connect(lambda item, _message: page.refresh_item(item))
        worker.signals.item_succeeded.connect(page.refresh_item)
        worker.signals.item_failed.connect(page.refresh_item)
        worker.signals.batch_progress.connect(page.mark_summary_dirty)
        worker.signals.batch_paused.connect(self._batch_paused)
        worker.signals.batch_resumed.connect(self._batch_resumed)
        worker.signals.batch_stopped.connect(self._batch_terminal)
        worker.signals.batch_completed.connect(self._batch_terminal)
        worker.signals.fatal_error.connect(self._batch_fatal)
        worker.signals.finished.connect(self._batch_worker_finished)
        self._set_batch_ui_locked(True)
        self._set_application_state("批量推理中")
        self.update_header_status("mode", "批量推理中", "active")
        page.begin_batch_updates()
        self.task_pool.start(worker)

    def _pause_batch(self) -> None:
        if self.batch_worker is not None:
            self.batch_worker.request_pause()
            self._batch_prediction_page.show_pause_requested()

    def _resume_batch(self) -> None:
        if self.batch_worker is not None:
            self.batch_worker.request_resume()

    def _stop_batch(self) -> None:
        if self.batch_worker is not None:
            self.batch_worker.request_stop()
            self._batch_prediction_page.show_feedback("正在完成当前文件并安全停止…")
            self._batch_prediction_page.flush_pending_ui_updates()
            self._batch_prediction_page.refresh_summary()

    def _batch_paused(self, task: BatchPredictionTask) -> None:
        self._set_application_state("批量已暂停")
        self.update_header_status("mode", "批量已暂停", "warning")
        self._batch_prediction_page.flush_pending_ui_updates()
        self._batch_prediction_page.refresh_summary()
        self._batch_prediction_page.refresh_actions()

    def _batch_resumed(self, task: BatchPredictionTask) -> None:
        self._set_application_state("批量推理中")
        self.update_header_status("mode", "批量推理中", "active")
        self._batch_prediction_page.flush_pending_ui_updates()
        self._batch_prediction_page.refresh_summary()
        self._batch_prediction_page.refresh_actions()

    def _batch_terminal(self, task: BatchPredictionTask) -> None:
        page = self._batch_prediction_page
        page.current_file_label.setText(
            "已安全停止，可继续剩余任务"
            if task.status == BatchStatus.STOPPED
            else "全部文件处理完成"
        )
        page.finalize_batch_updates()
        page.tabs.setCurrentIndex(1)
        self._set_batch_ui_locked(False)
        self._set_application_state("正常")
        self.update_header_status("mode", "本地", "neutral")
        self._run_task(
            lambda: self._export_and_register_batch(task),
            "批量结果自动导出",
            page.show_export_result,
            lambda message: page.show_feedback(f"自动导出失败：{message}", error=True),
        )

    def _batch_fatal(self, message: str) -> None:
        page = self._batch_prediction_page
        page.current_file_label.setText("批量任务因不可恢复错误终止")
        page.show_feedback(f"批量任务已终止：{message}", error=True)
        page.finalize_batch_updates()
        self._set_batch_ui_locked(False)
        self._set_application_state("推理失败")
        self.update_header_status("mode", "本地", "neutral")
        self._run_task(
            lambda: self._export_and_register_batch(self.batch_task),
            "批量致命错误结果导出",
            page.show_export_result,
            lambda error: page.show_feedback(f"错误结果导出失败：{error}", error=True),
        )

    def _batch_worker_finished(self) -> None:
        self.batch_worker = None
        self._batch_prediction_page.flush_pending_ui_updates()
        self._batch_prediction_page.refresh_summary()
        self._batch_prediction_page.refresh_actions()

    def _retry_batch(self, item_ids: list[str] | None) -> None:
        count = self.batch_prediction_service.retry_failed(self.batch_task, item_ids)
        page = self._batch_prediction_page
        page.show_feedback(f"已将 {count} 个失败项放回待处理队列。")
        page.refresh_table()

    def _export_and_register_batch(self, task: BatchPredictionTask) -> object:
        exported = self.batch_prediction_service.export_results(task)
        if self.history_service is not None:
            self._record_history_safely(
                lambda: self.history_service.register_batch(task, exported)
            )
        return exported

    def _export_batch_copy(self, destination: Path) -> None:
        page = self._batch_prediction_page
        self._run_task(
            lambda: self.batch_prediction_service.export_results(
                self.batch_task,
                destination=destination,
            ),
            "批量结果导出副本",
            page.show_export_result,
            lambda message: page.show_feedback(message, error=True),
        )

    def _export_filtered_batch(
        self,
        destination: Path,
        items: list[Any],
    ) -> None:
        page = self._batch_prediction_page
        self._run_task(
            lambda: self.batch_prediction_service.export_filtered_results(
                self.batch_task,
                items,
                destination,
            ),
            "导出当前筛选结果",
            page.show_filtered_export_result,
            lambda message: page.show_feedback(message, error=True),
        )

    def _open_batch_history(self, directory: Path) -> None:
        page = self._batch_prediction_page
        page.show_feedback("正在读取历史批量结果…")

        def loaded(task: BatchPredictionTask) -> None:
            self.batch_task = task
            page.show_history(task)

        self._run_task(
            lambda: self.batch_prediction_service.load_history(directory),
            "打开历史批量结果",
            loaded,
            lambda message: page.show_feedback(message, error=True),
        )

    def _preview_batch_result(self, source_path: Path) -> None:
        page = self._batch_prediction_page
        self._run_task(
            lambda: self.prediction_service.preview_file(source_path),
            "批量结果波形按需读取",
            page.show_result_preview,
            page.show_result_preview_error,
        )

    def _set_batch_ui_locked(self, locked: bool) -> None:
        self._single_prediction_page.setEnabled(not locked)
        self._validation_page.setEnabled(not locked)
        self._model_page.set_busy(locked, "批量推理期间模型已锁定。" if locked else "")
        self._settings_page.set_device_switch_locked(
            locked,
            "当前批量任务正在使用计算设备，请完成或停止任务后再切换。"
            if locked
            else "",
        )
        self._batch_prediction_page.refresh_actions()

    def _check_validation_manifest(
        self,
        manifest_path: Path,
        data_root: Path | None,
        missing_policy: str,
    ) -> None:
        page = self._validation_page
        if self.loaded_model is None:
            page.show_error("请先在模型管理中激活模型。")
            return
        locked_model = self.loaded_model
        page.progress_label.setText("正在后台检查验证清单…")
        self._run_task(
            lambda: self.validation_service.inspect_manifest(
                manifest_path,
                locked_model,
                data_root=data_root,
                missing_policy=missing_policy,
            ),
            "验证清单检查",
            page.show_manifest_report,
            page.show_error,
        )

    def _start_validation(self) -> None:
        page = self._validation_page
        if self.loaded_model is None or page.report is None:
            page.show_error("请先激活模型并完成验证清单检查。")
            return
        if self.batch_task.status in RUNNING_BATCH_STATUSES:
            page.show_error("批量预测运行期间不能同时启动模型验证。")
            return
        task = self.validation_service.create_task(page.report, self.loaded_model)
        self.validation_task = task
        if self.history_service is not None:
            self._record_history_safely(
                lambda: self.history_service.register_validation_running(task)
            )
        page.set_task(task)
        worker = ValidationWorker(self.engine, task)
        self.validation_worker = worker
        worker.signals.validation_started.connect(page.refresh_summary)
        worker.signals.sample_started.connect(page.refresh_sample)
        worker.signals.sample_succeeded.connect(page.refresh_sample)
        worker.signals.sample_failed.connect(page.refresh_sample)
        worker.signals.validation_progress.connect(lambda _task: page.refresh_summary())
        worker.signals.validation_paused.connect(self._validation_paused)
        worker.signals.validation_resumed.connect(self._validation_resumed)
        worker.signals.validation_stopped.connect(self._validation_terminal)
        worker.signals.validation_completed.connect(self._validation_terminal)
        worker.signals.fatal_error.connect(self._validation_fatal)
        worker.signals.finished.connect(self._validation_worker_finished)
        self._set_validation_ui_locked(True)
        self._set_application_state("模型验证中")
        self.update_header_status("mode", "模型验证中", "active")
        self.task_pool.start(worker)

    def _pause_validation(self) -> None:
        if self.validation_worker is not None:
            self.validation_worker.request_pause()
            self._validation_page.progress_label.setText("将在当前样本结束后暂停…")

    def _resume_validation(self) -> None:
        if self.validation_worker is not None:
            self.validation_worker.request_resume()

    def _stop_validation(self) -> None:
        if self.validation_worker is not None:
            self.validation_worker.request_stop()
            self._validation_page.progress_label.setText("正在安全停止验证任务…")

    def _validation_paused(self, task: ValidationTask) -> None:
        self._validation_page.refresh_task()
        self._set_application_state("验证已暂停")
        self.update_header_status("mode", "验证已暂停", "warning")

    def _validation_resumed(self, task: ValidationTask) -> None:
        self._validation_page.refresh_task()
        self._set_application_state("模型验证中")
        self.update_header_status("mode", "模型验证中", "active")

    def _validation_terminal(self, task: ValidationTask) -> None:
        page = self._validation_page
        page.show_validation_complete(task)
        self._set_validation_ui_locked(False)
        self._set_application_state("正常")
        self.update_header_status("mode", "本地", "neutral")
        self._run_task(
            lambda: self._export_and_register_validation(task),
            "验证结果自动导出",
            page.show_export_result,
            page.show_error,
        )

    def _validation_fatal(self, message: str) -> None:
        page = self._validation_page
        page.show_error(f"验证任务已终止：{message}")
        self._set_validation_ui_locked(False)
        self._set_application_state("推理失败")
        self.update_header_status("mode", "本地", "neutral")
        if self.validation_task is not None:
            self._run_task(
                lambda: self._export_and_register_validation(self.validation_task),
                "验证失败结果导出",
                page.show_export_result,
                page.show_error,
            )

    def _validation_worker_finished(self) -> None:
        self.validation_worker = None
        self._validation_page.refresh_task()

    def _open_validation_history(self, directory: Path) -> None:
        page = self._validation_page
        page.progress_label.setText("正在读取历史验证结果…")

        def loaded(task: ValidationTask) -> None:
            self.validation_task = task
            page.show_history(task)

        self._run_task(
            lambda: self.validation_service.load_history(directory),
            "打开历史验证结果",
            loaded,
            page.show_error,
        )

    def _validate_existing_batch(self, directory: Path) -> None:
        page = self._validation_page
        if self.loaded_model is None or page.report is None:
            page.show_error("请先激活模型并检查验证清单。")
            return
        locked_model = self.loaded_model
        report = page.report
        page.progress_label.setText("正在关联已有批量结果与真实标签…")

        def completed(task: ValidationTask) -> None:
            self.validation_task = task
            page.show_validation_complete(task)
            self._run_task(
                lambda: self._export_and_register_validation(task),
                "已有批量结果验证导出",
                page.show_export_result,
                page.show_error,
            )

        self._run_task(
            lambda: self.validation_service.validate_existing_batch(
                directory,
                report,
                locked_model,
            ),
            "已有批量结果转验证",
            completed,
            page.show_error,
        )

    def _preview_validation_sample(self, source_path: Path) -> None:
        page = self._validation_page
        self._run_task(
            lambda: self.prediction_service.preview_file(source_path),
            "验证样本波形按需读取",
            page.show_preview,
            page.show_preview_error,
        )

    def _export_validation_copy(self, destination: Path) -> None:
        if self.validation_task is None:
            return
        page = self._validation_page
        self._run_task(
            lambda: self.validation_service.export_results(
                self.validation_task,
                destination=destination,
            ),
            "验证报告副本导出",
            page.show_export_result,
            page.show_error,
        )

    def _export_and_register_validation(self, task: ValidationTask) -> object:
        exported = self.validation_service.export_results(task)
        if self.history_service is not None:
            self._record_history_safely(
                lambda: self.history_service.register_validation(task, exported)
            )
        return exported

    @staticmethod
    def _record_history_safely(operation: Any) -> object | None:
        """Keep a history-index failure from changing inference behavior."""
        try:
            return operation()
        except Exception:
            LOGGER.exception("Task history update failed; runtime workflow continues")
            return None

    def _set_validation_ui_locked(self, locked: bool) -> None:
        self._single_prediction_page.setEnabled(not locked)
        self._batch_prediction_page.setEnabled(not locked)
        self._model_page.set_busy(locked, "模型验证期间当前模型已锁定。" if locked else "")
        self._settings_page.set_device_switch_locked(
            locked,
            "当前模型验证正在使用计算设备，请完成或停止任务后再切换。"
            if locked
            else "",
        )
        self._validation_page.refresh_task()
