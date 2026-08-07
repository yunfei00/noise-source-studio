"""Batch queue, worker, page and export contract tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from threading import Event, Thread
from time import perf_counter, sleep
from types import SimpleNamespace
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from noise_source_studio.common.exceptions import InferenceEngineNotConfiguredError
from noise_source_studio.domain.batch import (
    BatchFileItem,
    BatchItemStatus,
    BatchPredictionTask,
    BatchStatus,
)
from noise_source_studio.domain.models import LoadedModel, ModelRecord
from noise_source_studio.infrastructure.inference import BatchWorker
from noise_source_studio.presentation.pages.batch_prediction_page import BatchPredictionPage
from noise_source_studio.services import BatchPredictionService
from noise_source_studio.services.result_adapter import (
    display_probabilities,
    normalize_prediction_result,
)


def _csv(path: Path, content: str = "DATA\n0,1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _result(mode: str = "structured", labels: list[str] | None = None) -> dict[str, Any]:
    labels = labels or ["风扇", "电机", "开关电源"]
    return {
        "runtime_version": "1.0.0",
        "device": "cpu",
        "labels": labels,
        "decision_mode": mode,
        "multilabel_probabilities": [0.9, 0.2, 0.8][: len(labels)],
        "label_marginal_probabilities": [0.8, 0.3, 0.7][: len(labels)],
        "combination_labels": ["100", "101"],
        "combination_probabilities": [0.25, 0.75],
        "decoded_label_vector": [1, 0, 1][: len(labels)],
        "predicted_combination": "101",
        "predicted_sources": [labels[0], labels[-1]],
        "thresholds": [0.5] * len(labels),
        "thresholds_applicable": mode == "multilabel",
        "input_shape": [1, 1, 128, 64],
    }


class FakeEngine:
    runtime_version = "1.0.0"

    def __init__(
        self,
        *,
        failures: set[str] | None = None,
        fatal: bool = False,
        block_first: bool = False,
        mode: str = "structured",
    ) -> None:
        self.failures = failures or set()
        self.fatal = fatal
        self.block_first = block_first
        self.mode = mode
        self.calls: list[Path] = []
        self.first_started = Event()
        self.release_first = Event()
        self.load_count = 1

    def predict_file(self, path: Path) -> dict[str, Any]:
        self.calls.append(path)
        if self.block_first and len(self.calls) == 1:
            self.first_started.set()
            assert self.release_first.wait(3)
        if self.fatal:
            raise InferenceEngineNotConfiguredError("session closed")
        if path.name in self.failures:
            raise ValueError("CSV 缺少 DATA")
        return _result(self.mode)


def _task(tmp_path: Path, count: int = 3) -> BatchPredictionTask:
    task = BatchPredictionTask("test", tmp_path / "outputs")
    task.items = [
        BatchFileItem(index, _csv(tmp_path / f"file{index}.csv")) for index in range(1, count + 1)
    ]
    task.refresh_counts()
    task.model_name = "noise-model"
    task.model_version = "1.0.0"
    task.runtime_version = "1.0.0"
    task.device = "cpu"
    return task


def _loaded(tmp_path: Path) -> LoadedModel:
    record = ModelRecord(
        "noise-model",
        "1.0.0",
        tmp_path / "model",
        {"labels": ["a"], "prediction_mode": "multilabel"},
        "2026-01-01T00:00:00Z",
        True,
        "valid",
    )
    return LoadedModel(record, "1.0.0", "cpu", "multilabel", ("a",), {})


def _wait_for(predicate: Any, timeout: float = 3.0) -> None:
    elapsed = 0.0
    while not predicate():
        sleep(0.01)
        elapsed += 0.01
        if elapsed >= timeout:
            raise AssertionError("timed out waiting for worker state")


def test_add_multiple_files(tmp_path: Path) -> None:
    service = BatchPredictionService(tmp_path / "out")
    task = service.create_task("files")
    summary = service.add_paths(
        task,
        [_csv(tmp_path / "b.csv"), _csv(tmp_path / "a.csv")],
        recursive=False,
    )
    assert summary.added_count == 2
    assert [item.file_name for item in task.items] == ["a.csv", "b.csv"]


def test_recursive_folder_scan(tmp_path: Path) -> None:
    service = BatchPredictionService(tmp_path / "out")
    task = service.create_task()
    _csv(tmp_path / "root.csv")
    _csv(tmp_path / "child" / "nested.csv")
    summary = service.add_paths(task, [tmp_path], recursive=True)
    assert summary.added_count == 2
    assert task.recursive


def test_non_recursive_folder_scan_excludes_children(tmp_path: Path) -> None:
    service = BatchPredictionService(tmp_path / "out")
    task = service.create_task()
    _csv(tmp_path / "root.csv")
    _csv(tmp_path / "child" / "nested.csv")
    service.add_paths(task, [tmp_path], recursive=False)
    assert [item.file_name for item in task.items] == ["root.csv"]


def test_duplicate_path_is_ignored(tmp_path: Path) -> None:
    service = BatchPredictionService(tmp_path / "out")
    task = service.create_task()
    source = _csv(tmp_path / "same.csv")
    summary = service.add_paths(task, [source, source], recursive=False)
    assert summary.added_count == 1
    assert summary.duplicate_count == 1


def test_invalid_extension_is_rejected(tmp_path: Path) -> None:
    service = BatchPredictionService(tmp_path / "out")
    task = service.create_task()
    invalid = tmp_path / "signal.txt"
    invalid.write_text("x", encoding="utf-8")
    summary = service.add_paths(task, [invalid], recursive=False)
    assert summary.invalid_count == 1
    assert task.total_count == 0


def test_natural_sort_orders_numeric_names(tmp_path: Path) -> None:
    service = BatchPredictionService(tmp_path / "out")
    task = service.create_task()
    paths = [_csv(tmp_path / name) for name in ("file10.csv", "file2.csv", "file1.csv")]
    service.add_paths(task, paths, recursive=False)
    assert [item.file_name for item in task.items] == [
        "file1.csv",
        "file2.csv",
        "file10.csv",
    ]


def test_remove_and_renumber_items(tmp_path: Path) -> None:
    service = BatchPredictionService(tmp_path / "out")
    task = _task(tmp_path)
    service.remove_items(task, [task.items[1].item_id])
    assert [item.sequence for item in task.items] == [1, 2]


def test_clear_queue(tmp_path: Path) -> None:
    service = BatchPredictionService(tmp_path / "out")
    task = _task(tmp_path)
    service.lock_model(task, _loaded(tmp_path))
    service.clear_items(task)
    assert task.total_count == 0
    assert task.model_name == ""
    assert task.status == BatchStatus.CREATED


def test_batch_worker_runs_in_sequence(tmp_path: Path) -> None:
    task = _task(tmp_path)
    engine = FakeEngine()
    BatchWorker(engine, task).run()  # type: ignore[arg-type]
    assert engine.calls == [item.file_path for item in task.items]
    assert task.status == BatchStatus.COMPLETED


def test_worker_never_loads_model(tmp_path: Path) -> None:
    task = _task(tmp_path)
    engine = FakeEngine()
    BatchWorker(engine, task).run()  # type: ignore[arg-type]
    assert engine.load_count == 1


def test_single_item_failure_continues(tmp_path: Path) -> None:
    task = _task(tmp_path)
    engine = FakeEngine(failures={"file2.csv"})
    BatchWorker(engine, task).run()  # type: ignore[arg-type]
    assert len(engine.calls) == 3
    assert task.success_count == 2
    assert task.failed_count == 1
    assert task.status == BatchStatus.COMPLETED_WITH_ERRORS


@pytest.mark.parametrize("count", [100, 1_000])
def test_large_fake_batch_preserves_every_prediction(tmp_path: Path, count: int) -> None:
    task = _task(tmp_path, count)
    engine = FakeEngine()
    started = perf_counter()
    BatchWorker(engine, task).run()  # type: ignore[arg-type]
    elapsed = perf_counter() - started
    assert len(engine.calls) == count
    assert task.success_count == count
    assert task.failed_count == 0
    assert all(item.predicted_combination == "101" for item in task.items)
    assert all(item.predicted_sources == ["风扇", "开关电源"] for item in task.items)
    assert elapsed < 10.0


def test_deleted_file_is_item_error(tmp_path: Path) -> None:
    task = _task(tmp_path)
    task.items[1].file_path.unlink()
    engine = FakeEngine()
    BatchWorker(engine, task).run()  # type: ignore[arg-type]
    assert task.items[1].status == BatchItemStatus.FAILED
    assert task.items[2].status == BatchItemStatus.SUCCESS


def test_pause_happens_at_file_boundary(tmp_path: Path) -> None:
    task = _task(tmp_path)
    engine = FakeEngine(block_first=True)
    worker = BatchWorker(engine, task)  # type: ignore[arg-type]
    thread = Thread(target=worker.run)
    thread.start()
    assert engine.first_started.wait(2)
    worker.request_pause()
    engine.release_first.set()
    _wait_for(lambda: task.status == BatchStatus.PAUSED)
    assert len(engine.calls) == 1
    worker.request_resume()
    thread.join(3)
    assert not thread.is_alive()


def test_resume_does_not_repeat_success(tmp_path: Path) -> None:
    task = _task(tmp_path)
    engine = FakeEngine(block_first=True)
    worker = BatchWorker(engine, task)  # type: ignore[arg-type]
    thread = Thread(target=worker.run)
    thread.start()
    assert engine.first_started.wait(2)
    worker.request_pause()
    engine.release_first.set()
    _wait_for(lambda: task.status == BatchStatus.PAUSED)
    worker.request_resume()
    thread.join(3)
    assert [path.name for path in engine.calls] == ["file1.csv", "file2.csv", "file3.csv"]


def test_stop_marks_remaining_items_stopped(tmp_path: Path) -> None:
    task = _task(tmp_path)
    engine = FakeEngine(block_first=True)
    worker = BatchWorker(engine, task)  # type: ignore[arg-type]
    thread = Thread(target=worker.run)
    thread.start()
    assert engine.first_started.wait(2)
    worker.request_stop()
    engine.release_first.set()
    thread.join(3)
    assert task.status == BatchStatus.STOPPED
    assert task.success_count == 1
    assert task.stopped_count == 2


def test_fatal_error_terminates_batch(tmp_path: Path) -> None:
    task = _task(tmp_path)
    BatchWorker(FakeEngine(fatal=True), task).run()  # type: ignore[arg-type]
    assert task.status == BatchStatus.FAILED
    assert task.failed_count == 1
    assert task.stopped_count == 2


def test_progress_and_counts_are_authoritative(tmp_path: Path) -> None:
    task = _task(tmp_path, 4)
    task.items[0].status = BatchItemStatus.SUCCESS
    task.items[1].status = BatchItemStatus.FAILED
    task.refresh_counts()
    assert task.completed_count == 2
    assert task.progress == 50.0


def test_eta_without_samples_is_safe(tmp_path: Path) -> None:
    assert _task(tmp_path).estimated_remaining_seconds is None


def test_eta_requires_two_attempts(tmp_path: Path) -> None:
    task = _task(tmp_path)
    for item, elapsed in zip(task.items[:2], (1000.0, 3000.0), strict=True):
        item.status = BatchItemStatus.SUCCESS
        item.elapsed_ms = elapsed
    task.refresh_counts()
    assert task.average_item_seconds == 2.0
    assert task.estimated_remaining_seconds == 2.0


def test_structured_mapping_uses_marginals(tmp_path: Path) -> None:
    task = _task(tmp_path, 1)
    BatchWorker(FakeEngine(mode="structured"), task).run()  # type: ignore[arg-type]
    assert task.items[0].display_probabilities == [0.8, 0.3, 0.7]
    assert task.items[0].decoded_label_vector == [1, 0, 1]


def test_multilabel_mapping_uses_sigmoid_probabilities(tmp_path: Path) -> None:
    task = _task(tmp_path, 1)
    BatchWorker(FakeEngine(mode="multilabel"), task).run()  # type: ignore[arg-type]
    assert task.items[0].display_probabilities == [0.9, 0.2, 0.8]


def test_dynamic_label_count_is_preserved(tmp_path: Path) -> None:
    payload = _result("multilabel", ["a", "b"])
    assert len(display_probabilities(payload)) == 2


def test_retry_failed_item_preserves_retry_count(tmp_path: Path) -> None:
    service = BatchPredictionService(tmp_path / "out")
    task = _task(tmp_path)
    task.items[0].status = BatchItemStatus.FAILED
    task.items[0].error_message = "bad"
    task.refresh_counts()
    assert service.retry_failed(task, [task.items[0].item_id]) == 1
    assert task.items[0].status == BatchItemStatus.PENDING
    assert task.items[0].retry_count == 1
    assert not task.items[0].error_message


def test_model_metadata_is_locked(tmp_path: Path) -> None:
    service = BatchPredictionService(tmp_path / "out")
    task = service.create_task()
    service.lock_model(task, _loaded(tmp_path))
    assert (task.model_name, task.model_version, task.device) == (
        "noise-model",
        "1.0.0",
        "cpu",
    )


def test_predictions_export_has_complete_fields(tmp_path: Path) -> None:
    service = BatchPredictionService(tmp_path / "out")
    task = _task(tmp_path, 1)
    BatchWorker(FakeEngine(), task).run()  # type: ignore[arg-type]
    exported = service.export_results(task)
    with exported.predictions_path.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    required = {"labels", "display_probabilities", "input_shape", "error_message"}
    assert required <= row.keys()


def test_csv_array_fields_are_valid_json(tmp_path: Path) -> None:
    service = BatchPredictionService(tmp_path / "out")
    task = _task(tmp_path, 1)
    BatchWorker(FakeEngine(), task).run()  # type: ignore[arg-type]
    exported = service.export_results(task)
    with exported.predictions_path.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert json.loads(row["labels"]) == ["风扇", "电机", "开关电源"]
    assert json.loads(row["input_shape"]) == [1, 1, 128, 64]


def test_chinese_path_export_round_trip(tmp_path: Path) -> None:
    service = BatchPredictionService(tmp_path / "输出")
    task = BatchPredictionTask("中文任务", tmp_path / "输出")
    task.items = [BatchFileItem(1, _csv(tmp_path / "样本" / "风扇.csv"))]
    task.refresh_counts()
    BatchWorker(FakeEngine(), task).run()  # type: ignore[arg-type]
    exported = service.export_results(task)
    with exported.predictions_path.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert "风扇.csv" in row["file_path"]


def test_errors_export_contains_only_failures(tmp_path: Path) -> None:
    service = BatchPredictionService(tmp_path / "out")
    task = _task(tmp_path, 2)
    BatchWorker(FakeEngine(failures={"file2.csv"}), task).run()  # type: ignore[arg-type]
    exported = service.export_results(task)
    with exported.errors_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["file_name"] == "file2.csv"


def test_export_failure_does_not_mutate_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = BatchPredictionService(tmp_path / "out")
    task = _task(tmp_path, 1)
    BatchWorker(FakeEngine(), task).run()  # type: ignore[arg-type]
    result_before = dict(task.items[0].result or {})
    monkeypatch.setattr(service, "_write_predictions", lambda *_: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        service.export_results(task)
    assert task.items[0].result == result_before


def test_page_table_matches_task_count(qapp: QApplication, qtbot: QtBot, tmp_path: Path) -> None:
    page = BatchPredictionPage()
    qtbot.addWidget(page)
    page.set_task(_task(tmp_path, 4))
    assert page.task_table.model().rowCount() == 4


def test_recursive_folder_scan_is_selected_by_default(
    qapp: QApplication, qtbot: QtBot
) -> None:
    page = BatchPredictionPage()
    qtbot.addWidget(page)
    assert page.recursive_checkbox.isChecked()


def test_page_button_states_follow_lifecycle(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    page = BatchPredictionPage()
    qtbot.addWidget(page)
    task = _task(tmp_path, 1)
    page.set_task(task)
    page.set_model_available(True)
    assert page.start_button.isEnabled()
    task.status = BatchStatus.RUNNING
    page.refresh_table()
    assert not page.start_button.isEnabled()
    assert page.pause_button.isEnabled()
    assert page.stop_button.isEnabled()
    task.items[0].status = BatchItemStatus.SUCCESS
    task.status = BatchStatus.COMPLETED
    page.refresh_table()
    assert not page.start_button.isEnabled()
    assert page.export_button.isEnabled()


def test_single_and_batch_share_normalizer() -> None:
    source = SimpleNamespace(**_result())
    assert BatchPredictionService.result_payload(source) == normalize_prediction_result(source)
