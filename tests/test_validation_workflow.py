"""Validation worker, persistence, history and batch reuse tests."""

from __future__ import annotations

import csv
from pathlib import Path
from threading import Event, Thread
from time import sleep
from typing import Any

from noise_source_studio.domain.batch import (
    BatchFileItem,
    BatchItemStatus,
    BatchPredictionTask,
    BatchStatus,
)
from noise_source_studio.domain.models import LoadedModel, ModelRecord
from noise_source_studio.domain.validation import (
    ManifestValidationReport,
    ValidationSample,
    ValidationStatus,
)
from noise_source_studio.infrastructure.inference import ValidationWorker
from noise_source_studio.services import BatchPredictionService, ValidationService

LABELS = ["fan", "motor", "switch_power"]


def _signal(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("DATA\n0,1\n", encoding="utf-8")
    return path


def _model(tmp_path: Path, mode: str = "structured") -> LoadedModel:
    record = ModelRecord(
        "noise-model",
        "1.0.0",
        tmp_path / "model",
        {
            "labels": LABELS,
            "prediction_mode": mode,
            "checkpoint_sha256": "abc123",
        },
        "2026-01-01T00:00:00Z",
        True,
        "valid",
    )
    return LoadedModel(record, "1.0.0", "cpu", mode, tuple(LABELS), {})


def _payload(mode: str = "structured") -> dict[str, Any]:
    return {
        "runtime_version": "1.0.0",
        "device": "cpu",
        "labels": LABELS,
        "decision_mode": mode,
        "multilabel_probabilities": [0.2, 0.9, 0.7],
        "label_marginal_probabilities": [0.8, 0.1, 0.9],
        "combination_labels": ["001", "010", "101"],
        "combination_probabilities": [0.1, 0.2, 0.7],
        "decoded_label_vector": [1, 0, 1],
        "predicted_combination": "101",
        "predicted_sources": ["fan", "switch_power"],
        "thresholds": [0.5, 0.8, 0.8],
        "thresholds_applicable": mode == "multilabel",
        "input_shape": [1, 1, 128, 64],
    }


def _report(tmp_path: Path, count: int = 3) -> ManifestValidationReport:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "manifest.csv"
    path.write_text("file_path,true_combination\n", encoding="utf-8")
    samples = [
        ValidationSample(
            index,
            _signal(tmp_path / f"file-{index}.csv"),
            [1, 0, 1],
            "101",
            metadata={"frequency_mhz": str(600 + index * 50)},
        )
        for index in range(1, count + 1)
    ]
    return ManifestValidationReport(
        path,
        tmp_path,
        list(LABELS),
        samples=samples,
        metadata_fields=["frequency_mhz"],
        source_row_count=count,
    )


class FakeEngine:
    runtime_version = "1.0.0"

    def __init__(
        self,
        *,
        mode: str = "structured",
        fail: set[str] | None = None,
        block_first: bool = False,
    ) -> None:
        self.mode = mode
        self.fail = fail or set()
        self.block_first = block_first
        self.calls: list[Path] = []
        self.first_started = Event()
        self.release_first = Event()

    def predict_file(self, path: Path) -> dict[str, Any]:
        self.calls.append(path)
        if self.block_first and len(self.calls) == 1:
            self.first_started.set()
            assert self.release_first.wait(3)
        if path.name in self.fail:
            raise ValueError("损坏的 CSV")
        return _payload(self.mode)


def _wait_for(predicate: Any, timeout: float = 3.0) -> None:
    elapsed = 0.0
    while not predicate():
        sleep(0.01)
        elapsed += 0.01
        if elapsed >= timeout:
            raise AssertionError("timed out")


def test_structured_uses_decoded_label_vector(tmp_path: Path) -> None:
    service = ValidationService(tmp_path / "out")
    task = service.create_task(_report(tmp_path, 1), _model(tmp_path))
    ValidationWorker(FakeEngine(), task).run()  # type: ignore[arg-type]
    sample = task.samples[0]
    assert sample.predicted_label_vector == [1, 0, 1]
    assert sample.display_probabilities == [0.8, 0.1, 0.9]


def test_multilabel_uses_probabilities_and_thresholds(tmp_path: Path) -> None:
    service = ValidationService(tmp_path / "out")
    task = service.create_task(_report(tmp_path, 1), _model(tmp_path, "multilabel"))
    ValidationWorker(FakeEngine(mode="multilabel"), task).run()  # type: ignore[arg-type]
    assert task.samples[0].predicted_label_vector == [0, 1, 0]
    assert task.samples[0].predicted_combination == "010"


def test_inference_failure_continues_and_is_counted(tmp_path: Path) -> None:
    service = ValidationService(tmp_path / "out")
    task = service.create_task(_report(tmp_path), _model(tmp_path))
    worker = ValidationWorker(FakeEngine(fail={"file-2.csv"}), task)  # type: ignore[arg-type]
    worker.run()
    assert task.success_count == 2
    assert task.inference_failed_count == 1
    assert task.status == ValidationStatus.COMPLETED_WITH_ERRORS
    assert task.metrics["denominator"]["sample_count"] == 2


def test_validation_pause_resume_and_stop(tmp_path: Path) -> None:
    service = ValidationService(tmp_path / "out")
    task = service.create_task(_report(tmp_path), _model(tmp_path))
    engine = FakeEngine(block_first=True)
    worker = ValidationWorker(engine, task)  # type: ignore[arg-type]
    thread = Thread(target=worker.run)
    thread.start()
    assert engine.first_started.wait(2)
    worker.request_pause()
    engine.release_first.set()
    _wait_for(lambda: task.status == ValidationStatus.PAUSED)
    assert len(engine.calls) == 1
    worker.request_resume()
    thread.join(3)
    assert task.status == ValidationStatus.COMPLETED

    stopped_task = service.create_task(_report(tmp_path / "stopped"), _model(tmp_path))
    stopped_engine = FakeEngine(block_first=True)
    stopped_worker = ValidationWorker(stopped_engine, stopped_task)  # type: ignore[arg-type]
    stopped_thread = Thread(target=stopped_worker.run)
    stopped_thread.start()
    assert stopped_engine.first_started.wait(2)
    stopped_worker.request_stop()
    stopped_engine.release_first.set()
    stopped_thread.join(3)
    assert stopped_task.status == ValidationStatus.STOPPED
    assert stopped_task.stopped_count == 2


def test_export_and_history_restore_without_inference(tmp_path: Path) -> None:
    service = ValidationService(tmp_path / "out")
    task = service.create_task(_report(tmp_path, 2), _model(tmp_path))
    engine = FakeEngine()
    ValidationWorker(engine, task).run()  # type: ignore[arg-type]
    exported = service.export_results(task)
    expected = {
        "task.json",
        "summary.json",
        "sample_results.csv",
        "label_metrics.csv",
        "combination_metrics.csv",
        "confusion_matrix.csv",
        "group_metrics.csv",
        "errors.csv",
        "report.html",
    }
    assert {path.name for path in exported.output_directory.iterdir()} == expected
    calls_before = list(engine.calls)
    restored = service.load_history(exported.output_directory)
    assert engine.calls == calls_before
    assert restored.task_id == task.task_id
    assert restored.metrics == task.metrics
    assert len(restored.samples) == 2


def test_html_report_is_utf8_and_self_contained(tmp_path: Path) -> None:
    service = ValidationService(tmp_path / "out")
    task = service.create_task(_report(tmp_path, 1), _model(tmp_path))
    ValidationWorker(FakeEngine(), task).run()  # type: ignore[arg-type]
    report = service.export_results(task).report_path.read_text(encoding="utf-8")
    assert "模型验证报告" in report
    assert "checkpoint" in report.casefold()
    assert "http://" not in report and "https://" not in report


def test_existing_batch_results_are_reused_without_inference(tmp_path: Path) -> None:
    batch_service = BatchPredictionService(tmp_path / "batch-out")
    report = _report(tmp_path, 2)
    batch = BatchPredictionTask("batch", tmp_path / "batch-out")
    batch.model_name = "noise-model"
    batch.model_version = "1.0.0"
    batch.runtime_version = "1.0.0"
    batch.device = "cpu"
    batch.status = BatchStatus.COMPLETED
    batch.items = []
    for sample in report.samples:
        item = BatchFileItem(sample.sequence, sample.file_path)
        item.status = BatchItemStatus.SUCCESS
        item.result = _payload()
        item.labels = list(LABELS)
        item.predicted_combination = "101"
        item.predicted_sources = ["fan", "switch_power"]
        batch.items.append(item)
    batch.refresh_counts()
    directory = batch_service.export_results(batch).output_directory
    task = ValidationService(tmp_path / "validation-out").validate_existing_batch(
        directory,
        report,
        _model(tmp_path),
    )
    assert task.success_count == 2
    assert task.status == ValidationStatus.COMPLETED
    assert all(sample.exact_match for sample in task.samples)


def test_exported_sample_csv_preserves_metadata(tmp_path: Path) -> None:
    service = ValidationService(tmp_path / "out")
    task = service.create_task(_report(tmp_path, 1), _model(tmp_path))
    ValidationWorker(FakeEngine(), task).run()  # type: ignore[arg-type]
    path = service.export_results(task).sample_results_path
    with path.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert "frequency_mhz" in row["metadata"]
