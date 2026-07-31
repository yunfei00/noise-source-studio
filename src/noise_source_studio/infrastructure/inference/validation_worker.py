"""Sequential validation worker using the retained inference session."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from time import perf_counter

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from noise_source_studio.domain.interfaces import InferenceEngine
from noise_source_studio.domain.validation import (
    ValidationSample,
    ValidationSampleStatus,
    ValidationStatus,
    ValidationTask,
)
from noise_source_studio.infrastructure.inference.worker_control import (
    CooperativeWorkerControl,
    is_fatal_inference_error,
)
from noise_source_studio.services.result_adapter import normalize_prediction_result
from noise_source_studio.services.validation_metrics import calculate_validation_metrics
from noise_source_studio.services.validation_result import apply_validation_result

LOGGER = logging.getLogger("noise_source_studio.validation_worker")


class ValidationWorkerSignals(QObject):
    """Validation lifecycle signals emitted back to the GUI thread."""

    validation_started = Signal(object)
    sample_started = Signal(object)
    sample_succeeded = Signal(object)
    sample_failed = Signal(object)
    validation_progress = Signal(object)
    validation_paused = Signal(object)
    validation_resumed = Signal(object)
    validation_stopped = Signal(object)
    validation_completed = Signal(object)
    fatal_error = Signal(str)
    finished = Signal()


class ValidationWorker(QRunnable):
    """Run manifest samples sequentially and calculate metrics off the GUI thread."""

    def __init__(self, engine: InferenceEngine, task: ValidationTask) -> None:
        super().__init__()
        self.engine = engine
        self.task = task
        self.signals = ValidationWorkerSignals()
        self._control = CooperativeWorkerControl()

    def request_pause(self) -> None:
        self._control.request_pause()

    def request_resume(self) -> None:
        self._control.request_resume()

    def request_stop(self) -> None:
        self._control.request_stop()
        if self.task.status in {ValidationStatus.RUNNING, ValidationStatus.PAUSED}:
            self.task.status = ValidationStatus.STOPPING

    @Slot()
    def run(self) -> None:
        try:
            self._run_validation()
        except Exception as exc:
            self.task.status = ValidationStatus.FAILED
            self.task.finished_at = datetime.now(UTC)
            self.task.refresh_counts()
            LOGGER.exception("Validation fatal error | task_id=%s", self.task.task_id)
            self.signals.fatal_error.emit(str(exc))
        finally:
            self.signals.finished.emit()

    def _run_validation(self) -> None:
        if not self.task.samples:
            raise ValueError("验证任务没有样本。")
        self.task.started_at = self.task.started_at or datetime.now(UTC)
        self.task.finished_at = None
        self.task.status = ValidationStatus.RUNNING
        self.task.refresh_counts()
        self.signals.validation_started.emit(self.task)
        for sample in self.task.samples:
            if sample.status != ValidationSampleStatus.PENDING:
                continue
            if self._wait_at_boundary():
                self._mark_remaining_stopped()
                return
            if not self._run_sample(sample):
                self._mark_remaining_stopped()
                return

        self.task.finished_at = datetime.now(UTC)
        self.task.refresh_counts()
        self.task.metrics = calculate_validation_metrics(
            self.task.samples,
            self.task.labels,
            decision_mode=self.task.decision_mode,
            group_fields=self.task.metadata_fields,
        )
        self.task.status = (
            ValidationStatus.COMPLETED_WITH_ERRORS
            if self.task.inference_failed_count
            else ValidationStatus.COMPLETED
        )
        self.signals.validation_progress.emit(self.task)
        self.signals.validation_completed.emit(self.task)

    def _wait_at_boundary(self) -> bool:
        def paused() -> None:
            self.task.status = ValidationStatus.PAUSED
            self.signals.validation_paused.emit(self.task)

        def resumed() -> None:
            self.task.status = ValidationStatus.RUNNING
            self.signals.validation_resumed.emit(self.task)

        return self._control.wait_at_boundary(paused, resumed)

    def _run_sample(self, sample: ValidationSample) -> bool:
        started = perf_counter()
        sample.status = ValidationSampleStatus.RUNNING
        sample.error_type = ""
        sample.error_message = ""
        self.task.refresh_counts()
        self.signals.sample_started.emit(sample)
        try:
            runtime_result = self.engine.predict_file(sample.file_path)
            payload = normalize_prediction_result(runtime_result)
            apply_validation_result(
                sample,
                payload,
                self.task.labels,
                self.task.decision_mode,
            )
            del runtime_result
        except Exception as exc:
            sample.elapsed_ms = (perf_counter() - started) * 1000.0
            sample.error_type = type(exc).__name__
            sample.error_message = str(exc)
            sample.status = ValidationSampleStatus.INFERENCE_FAILED
            self.task.refresh_counts()
            self.signals.sample_failed.emit(sample)
            if is_fatal_inference_error(exc):
                self.task.status = ValidationStatus.FAILED
                self.task.finished_at = datetime.now(UTC)
                self.signals.fatal_error.emit(str(exc))
                return False
        else:
            sample.elapsed_ms = (perf_counter() - started) * 1000.0
            sample.status = ValidationSampleStatus.SUCCESS
            self.signals.sample_succeeded.emit(sample)
        self.task.refresh_counts()
        self.signals.validation_progress.emit(self.task)
        return True

    def _mark_remaining_stopped(self) -> None:
        for sample in self.task.samples:
            if sample.status == ValidationSampleStatus.PENDING:
                sample.status = ValidationSampleStatus.STOPPED
                sample.error_type = "Stopped"
                sample.error_message = "验证任务已由用户停止。"
        if self.task.status != ValidationStatus.FAILED:
            self.task.status = ValidationStatus.STOPPED
        self.task.finished_at = self.task.finished_at or datetime.now(UTC)
        self.task.refresh_counts()
        self.task.metrics = calculate_validation_metrics(
            self.task.samples,
            self.task.labels,
            decision_mode=self.task.decision_mode,
            group_fields=self.task.metadata_fields,
        )
        self.signals.validation_progress.emit(self.task)
        if self.task.status != ValidationStatus.FAILED:
            self.signals.validation_stopped.emit(self.task)


__all__ = ["ValidationWorker", "ValidationWorkerSignals"]
