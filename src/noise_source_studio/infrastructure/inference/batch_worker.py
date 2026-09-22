"""Sequential, cooperative batch inference worker."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from noise_source_studio.domain.batch import (
    BatchFileItem,
    BatchItemStatus,
    BatchPredictionTask,
    BatchStatus,
)
from noise_source_studio.domain.interfaces import InferenceEngine
from noise_source_studio.infrastructure.inference.worker_control import (
    CooperativeWorkerControl,
    is_fatal_inference_error,
)
from noise_source_studio.services.result_adapter import (
    display_probabilities,
    normalize_prediction_result,
)

LOGGER = logging.getLogger("noise_source_studio.batch_worker")


class BatchWorkerSignals(QObject):
    """All lifecycle notifications emitted by :class:`BatchWorker`."""

    batch_started = Signal(object)
    item_started = Signal(object)
    item_progress = Signal(object, str)
    item_succeeded = Signal(object)
    item_failed = Signal(object)
    batch_progress = Signal(object)
    batch_paused = Signal(object)
    batch_resumed = Signal(object)
    batch_stopped = Signal(object)
    batch_completed = Signal(object)
    fatal_error = Signal(str)
    finished = Signal()


class BatchWorker(QRunnable):
    """Process pending files in order against one already-loaded session."""

    def __init__(self, engine: InferenceEngine, task: BatchPredictionTask) -> None:
        super().__init__()
        self.engine = engine
        self.task = task
        self.signals = BatchWorkerSignals()
        self._control = CooperativeWorkerControl()

    def request_pause(self) -> None:
        """Pause at the next file boundary."""
        self._control.request_pause()

    def request_resume(self) -> None:
        """Wake a paused worker without resetting any successful item."""
        self._control.request_resume()

    def request_stop(self) -> None:
        """Cooperatively stop after the current inference returns."""
        self._control.request_stop()
        if self.task.status in {BatchStatus.RUNNING, BatchStatus.PAUSED}:
            self.task.status = BatchStatus.STOPPING

    @Slot()
    def run(self) -> None:
        """Execute pending items, isolating ordinary per-file errors."""
        try:
            self._run_batch()
        except Exception as exc:  # final safety boundary for Qt's thread pool
            self.task.status = BatchStatus.FAILED
            self.task.finished_at = datetime.now(UTC)
            self.task.refresh_counts()
            LOGGER.exception("Batch fatal error | task_id=%s", self.task.task_id)
            self.signals.fatal_error.emit(str(exc))
        finally:
            self.signals.finished.emit()

    def _run_batch(self) -> None:
        if not self.task.items:
            raise ValueError("批量队列为空。")
        self.task.started_at = self.task.started_at or datetime.now(UTC)
        self.task.finished_at = None
        self.task.status = BatchStatus.RUNNING
        self.task.refresh_counts()
        LOGGER.info(
            "Batch started | task_id=%s | model=%s@%s | device=%s | total=%d",
            self.task.task_id,
            self.task.model_name,
            self.task.model_version,
            self.task.device,
            self.task.total_count,
        )
        self.signals.batch_started.emit(self.task)

        for item in self.task.items:
            if item.status != BatchItemStatus.PENDING:
                continue
            if self._wait_at_boundary():
                self._mark_remaining_stopped()
                return
            if not self._run_item(item):
                if self.task.status == BatchStatus.FAILED:
                    self._mark_pending_after_fatal()
                else:
                    self._mark_remaining_stopped()
                return

        self.task.finished_at = datetime.now(UTC)
        self.task.refresh_counts()
        self.task.status = (
            BatchStatus.COMPLETED_WITH_ERRORS if self.task.failed_count else BatchStatus.COMPLETED
        )
        LOGGER.info(
            "Batch completed | task_id=%s | status=%s | success=%d | failed=%d | elapsed=%.3fs",
            self.task.task_id,
            self.task.status.value,
            self.task.success_count,
            self.task.failed_count,
            self.task.elapsed_seconds,
        )
        self.signals.batch_progress.emit(self.task)
        self.signals.batch_completed.emit(self.task)

    def _wait_at_boundary(self) -> bool:
        def paused() -> None:
            self.task.status = BatchStatus.PAUSED
            LOGGER.info("Batch paused | task_id=%s", self.task.task_id)
            self.signals.batch_paused.emit(self.task)

        def resumed() -> None:
            self.task.status = BatchStatus.RUNNING
            LOGGER.info("Batch resumed | task_id=%s", self.task.task_id)
            self.signals.batch_resumed.emit(self.task)

        return self._control.wait_at_boundary(paused, resumed)

    def _run_item(self, item: BatchFileItem) -> bool:
        item.started_at = datetime.now(UTC)
        started = perf_counter()
        self.task.transition_item_status(item, BatchItemStatus.VALIDATING)
        item.status_message = "校验文件"
        self.signals.item_started.emit(item)
        self.signals.item_progress.emit(item, "校验文件")
        LOGGER.info(
            "Batch item started | task_id=%s | item=%d | file=%s | model=%s@%s | device=%s",
            self.task.task_id,
            item.sequence,
            item.file_path,
            self.task.model_name,
            self.task.model_version,
            self.task.device,
        )
        try:
            self._validate_file(item.file_path)
            self.task.transition_item_status(item, BatchItemStatus.RUNNING)
            item.status_message = "推理中"
            self.signals.item_progress.emit(item, "推理中")
            runtime_result = self.engine.predict_file(item.file_path)
            payload = normalize_prediction_result(runtime_result)
            self._apply_result(item, payload)
            del runtime_result
        except Exception as exc:
            item.finished_at = datetime.now(UTC)
            item.elapsed_ms = (perf_counter() - started) * 1000.0
            if self._is_fatal(exc):
                self.task.transition_item_status(item, BatchItemStatus.FAILED)
                item.status_message = "致命错误"
                item.error_type = type(exc).__name__
                item.error_message = str(exc)
                self.task.status = BatchStatus.FAILED
                self.task.finished_at = item.finished_at
                self.signals.item_failed.emit(item)
                self.signals.fatal_error.emit(str(exc))
                LOGGER.exception(
                    "Batch fatal item | task_id=%s | item=%d | file=%s",
                    self.task.task_id,
                    item.sequence,
                    item.file_path,
                )
                return False
            self.task.transition_item_status(item, BatchItemStatus.FAILED)
            item.status_message = "失败"
            item.error_type = type(exc).__name__
            item.error_message = str(exc)
            LOGGER.warning(
                "Batch item failed | task_id=%s | item=%d | file=%s | error=%s",
                self.task.task_id,
                item.sequence,
                item.file_path,
                exc,
            )
            self.signals.item_failed.emit(item)
        else:
            item.finished_at = datetime.now(UTC)
            item.elapsed_ms = (perf_counter() - started) * 1000.0
            self.task.transition_item_status(item, BatchItemStatus.SUCCESS)
            item.status_message = "成功"
            self.signals.item_succeeded.emit(item)
            LOGGER.info(
                "Batch item completed | task_id=%s | item=%d | file=%s | elapsed_ms=%.2f",
                self.task.task_id,
                item.sequence,
                item.file_path,
                item.elapsed_ms,
            )
        self.signals.batch_progress.emit(self.task)
        return True

    def _mark_remaining_stopped(self) -> None:
        now = datetime.now(UTC)
        for item in self.task.items:
            if item.status == BatchItemStatus.PENDING:
                item.status = BatchItemStatus.STOPPED
                item.status_message = "已停止"
        self.task.status = BatchStatus.STOPPED
        self.task.finished_at = now
        self.task.refresh_counts()
        LOGGER.info(
            "Batch stopped | task_id=%s | completed=%d | stopped=%d",
            self.task.task_id,
            self.task.success_count + self.task.failed_count,
            self.task.stopped_count,
        )
        self.signals.batch_progress.emit(self.task)
        self.signals.batch_stopped.emit(self.task)

    def _mark_pending_after_fatal(self) -> None:
        for item in self.task.items:
            if item.status == BatchItemStatus.PENDING:
                item.status = BatchItemStatus.STOPPED
                item.status_message = "因任务致命错误未执行"
        self.task.finished_at = self.task.finished_at or datetime.now(UTC)
        self.task.refresh_counts()
        self.signals.batch_progress.emit(self.task)

    @staticmethod
    def _validate_file(path: Path) -> None:
        if path.suffix.lower() != ".csv":
            raise ValueError("仅支持 CSV 文件。")
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")
        with path.open("rb") as handle:
            handle.read(1)

    @staticmethod
    def _apply_result(item: BatchFileItem, payload: dict[str, Any]) -> None:
        item.result = payload
        item.labels = [str(value) for value in payload.get("labels", []) or []]
        item.decision_mode = str(payload.get("decision_mode", "unknown"))
        item.display_probabilities = display_probabilities(payload)
        item.predicted_combination = str(payload.get("predicted_combination", ""))
        item.predicted_sources = [
            str(value) for value in payload.get("predicted_sources", []) or []
        ]
        item.decoded_label_vector = [
            int(value) for value in payload.get("decoded_label_vector", []) or []
        ]
        item.error_type = ""
        item.error_message = ""

    @staticmethod
    def _is_fatal(exc: Exception) -> bool:
        return is_fatal_inference_error(exc)


__all__ = ["BatchWorker", "BatchWorkerSignals"]
