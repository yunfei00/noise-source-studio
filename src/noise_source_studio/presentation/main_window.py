"""Top-level application window."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThreadPool, QTimer
from PySide6.QtGui import QCloseEvent
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

from noise_source_studio.domain.batch import (
    RUNNING_BATCH_STATUSES,
    BatchPredictionTask,
    BatchStatus,
)
from noise_source_studio.domain.interfaces import InferenceEngine
from noise_source_studio.domain.models import LoadedModel, ModelRecord, PredictionOutcome
from noise_source_studio.infrastructure.config import AppSettings, SettingsManager
from noise_source_studio.infrastructure.inference import BatchWorker, RuntimeAdapter
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
    ModelService,
    PredictionService,
)
from noise_source_studio.services.tasks import BackgroundTask, TaskFailure
from noise_source_studio.version import APPLICATION_TITLE

LOGGER = logging.getLogger("noise_source_studio.main_window")


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
        engine: InferenceEngine | None = None,
        model_service: ModelService | None = None,
        prediction_service: PredictionService | None = None,
        batch_prediction_service: BatchPredictionService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.header_statuses: dict[str, HeaderStatusUnit] = {}
        self.engine = engine or RuntimeAdapter()
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
        self.batch_task = self.batch_prediction_service.create_task()
        self.batch_worker: BatchWorker | None = None
        self.loaded_model: LoadedModel | None = None
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
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)
        if self.model_service.active_model() is not None:
            QTimer.singleShot(0, self._restore_active_model)

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
        if self.batch_task.status in RUNNING_BATCH_STATUSES:
            answer = QMessageBox.question(
                self,
                "批量任务仍在运行",
                "退出将等待当前文件完成并安全停止批量任务。确定退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            if self.batch_worker is not None:
                self.batch_worker.request_stop()
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
            ("device", "计算设备", "待检测", "neutral", 120),
            ("mode", "工作模式", "本地", "neutral", 120),
        )
        for key, label, value, state, minimum_width in status_definitions:
            status = HeaderStatusUnit(label, value, state)
            status.setMinimumWidth(minimum_width)
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
    def _model_page(self) -> ModelManagementPage:
        page = self.pages[4]
        assert isinstance(page, ModelManagementPage)
        return page

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
        page = self._model_page
        page.set_busy(True, "正在后台加载模型…")
        self._set_application_state("模型加载中")
        self.update_header_status("model", "加载中", "warning")

        def activated(loaded: LoadedModel) -> None:
            self._apply_loaded_model(loaded)
            self._refresh_model_views()
            page.show_feedback(f"模型已激活：{loaded.record.display_name}", error=False)

        def failed(message: str) -> None:
            self._clear_loaded_model(load_failed=True)
            self._refresh_model_views()
            page.show_feedback(message, error=True)
            QMessageBox.warning(self, "模型加载失败", message)

        self._run_task(
            lambda: self.model_service.activate_model(
                identifier,
                device=self.settings.default_device,
            ),
            "模型激活",
            activated,
            failed,
            lambda: page.set_busy(False),
        )

    def _restore_active_model(self) -> None:
        record = self.model_service.active_model()
        if record is None:
            return
        self._set_application_state("模型加载中")
        self.update_header_status("model", "加载中", "warning")

        def failed(message: str) -> None:
            self._clear_loaded_model(load_failed=True)
            self._refresh_model_views()
            QMessageBox.warning(
                self,
                "活动模型恢复失败",
                f"{message}\n\n应用已恢复到安全的未配置状态。",
            )

        self._run_task(
            lambda: self.model_service.activate_model(
                record.identifier,
                device=self.settings.default_device,
            ),
            "启动恢复活动模型",
            self._apply_loaded_model,
            failed,
        )

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
        page = self._single_prediction_page
        page.begin_prediction()
        self._set_application_state("推理中")

        def completed(outcome: PredictionOutcome) -> None:
            page.show_prediction(outcome)
            self._set_application_state("正常")

        def failed(message: str) -> None:
            page.show_prediction_error(message)
            self._set_application_state("推理失败")

        self._run_task(
            lambda: self.prediction_service.predict_file(
                source_path,
                locked_record,
            ),
            "单文件推理",
            completed,
            failed,
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
        self.update_header_status("model", loaded.record.display_name, "active")
        self.update_header_status("device", loaded.device, "active")
        self.update_header_status("mode", "本地", "neutral")
        dashboard = self._dashboard_page
        dashboard.status_cards["model"].set_status(
            loaded.record.display_name,
            f"prediction mode：{loaded.prediction_mode}",
        )
        dashboard.status_cards["device"].set_status(
            loaded.device,
            f"runtime {loaded.runtime_version}",
        )
        self._single_prediction_page.set_model(loaded)
        self._batch_prediction_page.set_model_available(True)
        self._model_page.set_active_model(loaded.record, loaded.device)
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
        self._set_application_state("正常")

    def _set_application_state(self, state: str) -> None:
        notes = {
            "正常": "基础服务与模型会话状态正常",
            "模型加载中": "正在后台校验并创建模型会话",
            "推理中": "正在后台执行单文件推理",
            "批量推理中": "正在复用当前模型会话顺序执行批量任务",
            "批量已暂停": "当前文件已完成，等待用户继续或停止",
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
        worker = BatchWorker(self.engine, self.batch_task)
        self.batch_worker = worker
        page = self._batch_prediction_page
        worker.signals.batch_started.connect(page.refresh_summary)
        worker.signals.item_started.connect(page.refresh_item)
        worker.signals.item_progress.connect(lambda item, _message: page.refresh_item(item))
        worker.signals.item_succeeded.connect(page.refresh_item)
        worker.signals.item_failed.connect(page.refresh_item)
        worker.signals.batch_progress.connect(page.refresh_summary)
        worker.signals.batch_paused.connect(self._batch_paused)
        worker.signals.batch_resumed.connect(self._batch_resumed)
        worker.signals.batch_stopped.connect(self._batch_terminal)
        worker.signals.batch_completed.connect(self._batch_terminal)
        worker.signals.fatal_error.connect(self._batch_fatal)
        worker.signals.finished.connect(self._batch_worker_finished)
        self._set_batch_ui_locked(True)
        self._set_application_state("批量推理中")
        self.update_header_status("mode", "批量推理中", "active")
        page.refresh_table()
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
            self._batch_prediction_page.refresh_summary()

    def _batch_paused(self, task: BatchPredictionTask) -> None:
        self._set_application_state("批量已暂停")
        self.update_header_status("mode", "批量已暂停", "warning")
        self._batch_prediction_page.refresh_table()

    def _batch_resumed(self, task: BatchPredictionTask) -> None:
        self._set_application_state("批量推理中")
        self.update_header_status("mode", "批量推理中", "active")
        self._batch_prediction_page.refresh_table()

    def _batch_terminal(self, task: BatchPredictionTask) -> None:
        page = self._batch_prediction_page
        page.current_file_label.setText(
            "已安全停止，可继续剩余任务"
            if task.status == BatchStatus.STOPPED
            else "全部文件处理完成"
        )
        page.refresh_table()
        page.show_results_tab()
        self._set_batch_ui_locked(False)
        self._set_application_state("正常")
        self.update_header_status("mode", "本地", "neutral")
        self._run_task(
            lambda: self.batch_prediction_service.export_results(task),
            "批量结果自动导出",
            page.show_export_result,
            lambda message: page.show_feedback(f"自动导出失败：{message}", error=True),
        )

    def _batch_fatal(self, message: str) -> None:
        page = self._batch_prediction_page
        page.current_file_label.setText("批量任务因不可恢复错误终止")
        page.show_feedback(f"批量任务已终止：{message}", error=True)
        self._set_batch_ui_locked(False)
        self._set_application_state("推理失败")
        self.update_header_status("mode", "本地", "neutral")
        self._run_task(
            lambda: self.batch_prediction_service.export_results(self.batch_task),
            "批量致命错误结果导出",
            page.show_export_result,
            lambda error: page.show_feedback(f"错误结果导出失败：{error}", error=True),
        )

    def _batch_worker_finished(self) -> None:
        self.batch_worker = None
        self._batch_prediction_page.refresh_table()

    def _retry_batch(self, item_ids: list[str] | None) -> None:
        count = self.batch_prediction_service.retry_failed(self.batch_task, item_ids)
        page = self._batch_prediction_page
        page.show_feedback(f"已将 {count} 个失败项放回待处理队列。")
        page.refresh_table()

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
        self._model_page.set_busy(locked, "批量推理期间模型已锁定。" if locked else "")
        self.pages[7].setEnabled(not locked)
        self._batch_prediction_page.refresh_table()
