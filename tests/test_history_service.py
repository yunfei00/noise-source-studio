"""Unified task-history service and filesystem contract tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from noise_source_studio.domain.batch import (
    BatchExportResult,
    BatchPredictionTask,
    BatchStatus,
)
from noise_source_studio.domain.history import (
    HistoryQuery,
    HistoryStatus,
    IntegrityStatus,
    TaskType,
)
from noise_source_studio.domain.models import ModelRecord, PredictionOutcome
from noise_source_studio.domain.validation import (
    ValidationExportResult,
    ValidationStatus,
    ValidationTask,
)
from noise_source_studio.services.history_service import (
    HistoryService,
    UnsafeHistoryDeleteError,
)


@pytest.fixture
def service(tmp_path: Path) -> HistoryService:
    return HistoryService(tmp_path / "data" / "history" / "history.db", tmp_path / "outputs")


def _model(tmp_path: Path) -> ModelRecord:
    return ModelRecord(
        model_name="noise-net",
        model_version="2.1",
        package_path=tmp_path / "models" / "noise-net",
        manifest={"checkpoint_sha256": "abc123", "labels": ["fan", "pump"]},
        installed_at="2026-01-01T00:00:00+00:00",
        is_active=True,
        integrity_status="ok",
    )


def _outcome(tmp_path: Path, task_id: str = "single-task") -> PredictionOutcome:
    started = datetime(2026, 5, 6, 8, 0, tzinfo=UTC)
    return PredictionOutcome(
        task_id=task_id,
        source_path=tmp_path / "signals" / "sample.csv",
        model=_model(tmp_path),
        started_at=started,
        completed_at=started + timedelta(milliseconds=125),
        duration_seconds=0.125,
        result={
            "runtime_version": "1.4.0",
            "device": "cpu",
            "labels": ["fan", "pump"],
            "decision_mode": "structured",
            "label_marginal_probabilities": [0.8, 0.2],
            "combination_labels": ["fan", "pump"],
            "combination_probabilities": [0.8, 0.2],
            "predicted_combination": "fan",
            "predicted_sources": ["fan"],
            "decoded_label_vector": [1, 0],
            "input_shape": [1, 2048],
        },
    )


def _touch_files(root: Path, names: tuple[str, ...]) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in names:
        path = root / name
        path.write_text("{}\n" if path.suffix == ".json" else "header\n", encoding="utf-8")
        paths[name] = path
    return paths


def test_single_success_is_auto_saved_and_indexed(
    service: HistoryService,
    tmp_path: Path,
) -> None:
    outcome = _outcome(tmp_path)
    service.register_single_running(outcome.task_id, outcome.source_path, outcome.model)
    record = service.complete_single(outcome)
    assert record.status == HistoryStatus.COMPLETED
    assert record.success_count == 1
    assert record.device == "cpu"
    assert record.decision_mode == "structured"
    assert Path(record.task_file).is_file()
    assert Path(record.detail_file).is_file()
    assert Path(record.result_directory).parts[-3:] == (
        "single",
        "2026-05-06",
        "single-task",
    )


def test_single_restore_reads_files_without_engine(
    service: HistoryService,
    tmp_path: Path,
) -> None:
    outcome = _outcome(tmp_path)
    service.complete_single(outcome)
    restored = service.load_single_prediction(outcome.task_id)
    assert restored.task_id == outcome.task_id
    assert restored.result["predicted_combination"] == "fan"
    assert restored.model.identifier == "noise-net@2.1"


def test_failed_single_has_no_fake_result_files(
    service: HistoryService,
    tmp_path: Path,
) -> None:
    model = _model(tmp_path)
    service.register_single_running("failed-task", tmp_path / "bad.csv", model)
    service.fail_task("failed-task", ValueError("invalid DATA section"))
    record = service.get_task("failed-task")
    assert record is not None
    assert record.status == HistoryStatus.FAILED
    assert record.detail_file == ""
    assert not (service.output_root / "single" / "failed-task").exists()


def test_startup_marks_stale_running_task_interrupted(tmp_path: Path) -> None:
    database = tmp_path / "history.db"
    first = HistoryService(database, tmp_path / "outputs")
    first.register_single_running("running-task", tmp_path / "x.csv", _model(tmp_path))
    second = HistoryService(database, tmp_path / "outputs")
    assert second.get_task("running-task").status == HistoryStatus.INTERRUPTED  # type: ignore[union-attr]


def test_internal_paths_are_stored_relative(service: HistoryService, tmp_path: Path) -> None:
    outcome = _outcome(tmp_path)
    service.complete_single(outcome)
    stored = service.repository.get_task(outcome.task_id)
    assert stored is not None
    assert not Path(stored.result_directory).is_absolute()
    assert service.get_task(outcome.task_id).result_directory.startswith(  # type: ignore[union-attr]
        str(service.output_root)
    )


def test_external_source_path_stays_absolute(service: HistoryService, tmp_path: Path) -> None:
    outcome = _outcome(tmp_path)
    service.complete_single(outcome)
    record = service.get_task(outcome.task_id)
    assert record is not None
    assert Path(record.source_path).is_absolute()


def test_batch_registration_reuses_export(service: HistoryService, tmp_path: Path) -> None:
    task = BatchPredictionTask("production batch", tmp_path / "unused")
    task.status = BatchStatus.COMPLETED_WITH_ERRORS
    task.model_name = "noise-net"
    task.model_version = "2.1"
    task.device = "cuda:0"
    task.runtime_version = "1.4.0"
    task.total_count = 10
    task.success_count = 9
    task.failed_count = 1
    task.started_at = datetime.now(UTC) - timedelta(seconds=2)
    task.finished_at = datetime.now(UTC)
    paths = _touch_files(
        service.output_root / "batch_example",
        ("task.json", "summary.json", "predictions.csv", "errors.csv"),
    )
    exported = BatchExportResult(
        paths["task.json"].parent,
        paths["task.json"],
        paths["summary.json"],
        paths["predictions.csv"],
        paths["errors.csv"],
    )
    record = service.register_batch(task, exported)
    assert record.task_type == TaskType.BATCH
    assert record.status == HistoryStatus.COMPLETED_WITH_ERRORS
    assert record.total_count == 10
    assert record.failed_count == 1
    assert len(service.repository.artifacts(task.task_id)) == 4


def test_validation_registration_reuses_export(
    service: HistoryService,
    tmp_path: Path,
) -> None:
    task = ValidationTask(
        manifest_path=tmp_path / "manifest.csv",
        data_root=tmp_path,
        output_directory=tmp_path / "unused",
    )
    task.status = ValidationStatus.COMPLETED
    task.model_name = "noise-net"
    task.model_version = "2.1"
    task.checkpoint_sha256 = "abc123"
    task.device = "cpu"
    task.decision_mode = "structured"
    task.total_count = 8
    task.success_count = 8
    task.metrics = {"overall": {"exact_match_accuracy": 0.875}}
    names = (
        "task.json",
        "summary.json",
        "sample_results.csv",
        "label_metrics.csv",
        "combination_metrics.csv",
        "confusion_matrix.csv",
        "group_metrics.csv",
        "errors.csv",
        "report.html",
    )
    paths = _touch_files(service.output_root / "validation_example", names)
    exported = ValidationExportResult(
        output_directory=paths["task.json"].parent,
        task_path=paths["task.json"],
        summary_path=paths["summary.json"],
        sample_results_path=paths["sample_results.csv"],
        label_metrics_path=paths["label_metrics.csv"],
        combination_metrics_path=paths["combination_metrics.csv"],
        confusion_matrix_path=paths["confusion_matrix.csv"],
        group_metrics_path=paths["group_metrics.csv"],
        errors_path=paths["errors.csv"],
        report_path=paths["report.html"],
    )
    record = service.register_validation(task, exported)
    assert record.task_type == TaskType.VALIDATION
    assert "Exact Match 87.50%" in record.primary_summary
    assert record.report_file.endswith("report.html")


def test_verify_marks_missing_artifact(service: HistoryService, tmp_path: Path) -> None:
    record = service.complete_single(_outcome(tmp_path))
    Path(record.detail_file).unlink()
    assert service.verify_task(record.task_id) == IntegrityStatus.MISSING
    assert service.get_task(record.task_id).integrity_status == IntegrityStatus.MISSING  # type: ignore[union-attr]


def test_verify_marks_corrupt_json(service: HistoryService, tmp_path: Path) -> None:
    record = service.complete_single(_outcome(tmp_path))
    Path(record.detail_file).write_text("not-json", encoding="utf-8")
    assert service.verify_task(record.task_id) == IntegrityStatus.CORRUPT


def test_notes_and_tags_are_normalized(service: HistoryService, tmp_path: Path) -> None:
    record = service.complete_single(_outcome(tmp_path))
    service.update_notes(record.task_id, "  investigate  ", ("urgent", "urgent", " "))
    restored = service.get_task(record.task_id)
    assert restored is not None
    assert restored.notes == "investigate"
    assert restored.tags == ("urgent",)


def test_delete_index_keeps_result_files(service: HistoryService, tmp_path: Path) -> None:
    record = service.complete_single(_outcome(tmp_path))
    directory = Path(record.result_directory)
    assert service.delete_index(record.task_id)
    assert directory.is_dir()


def test_delete_task_files_is_restricted_to_output_root(
    service: HistoryService,
    tmp_path: Path,
) -> None:
    outcome = _outcome(tmp_path)
    service.register_single_running(outcome.task_id, outcome.source_path, outcome.model)
    service.repository.update_task(
        outcome.task_id,
        result_directory=str(tmp_path / "external"),
    )
    with pytest.raises(UnsafeHistoryDeleteError):
        service.delete_task_files(outcome.task_id)


def test_delete_task_files_removes_internal_directory(
    service: HistoryService,
    tmp_path: Path,
) -> None:
    record = service.complete_single(_outcome(tmp_path))
    directory = Path(record.result_directory)
    assert service.delete_task_files(record.task_id)
    assert not directory.exists()
    assert service.get_task(record.task_id) is None


def test_export_summary_is_utf8_bom(service: HistoryService, tmp_path: Path) -> None:
    record = service.complete_single(_outcome(tmp_path))
    service.update_notes(record.task_id, "中文验收备注", ("正式验证",))
    destination = tmp_path / "history.csv"
    service.export_summary(HistoryQuery(), destination)
    assert destination.read_bytes().startswith(b"\xef\xbb\xbf")
    assert "single-task" in destination.read_text(encoding="utf-8-sig")
    assert "中文验收备注" in destination.read_text(encoding="utf-8-sig")


def test_scan_imports_and_deduplicates_existing_result(
    service: HistoryService,
    tmp_path: Path,
) -> None:
    service.complete_single(_outcome(tmp_path))
    second = HistoryService(tmp_path / "new-history.db", service.output_root)
    first_report = second.scan_outputs()
    second_report = second.scan_outputs()
    assert first_report.imported_tasks == 1
    assert second_report.imported_tasks == 0
    assert second_report.updated_tasks == 1
    assert second.list_tasks(HistoryQuery()).total == 1


def test_legacy_scan_id_is_stable(service: HistoryService, tmp_path: Path) -> None:
    record = service.complete_single(_outcome(tmp_path))
    task_path = Path(record.task_file)
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    del payload["task"]["task_id"]
    task_path.write_text(json.dumps(payload), encoding="utf-8")
    second = HistoryService(tmp_path / "legacy.db", service.output_root)
    second.scan_outputs()
    first_id = second.list_tasks(HistoryQuery()).records[0].task_id
    second.scan_outputs()
    assert first_id.startswith("legacy-")
    assert second.list_tasks(HistoryQuery()).records[0].task_id == first_id


def test_scan_continues_past_corrupt_directory(
    service: HistoryService,
    tmp_path: Path,
) -> None:
    service.complete_single(_outcome(tmp_path))
    corrupt = service.output_root / "batch_corrupt"
    corrupt.mkdir(parents=True)
    (corrupt / "task.json").write_text("{broken", encoding="utf-8")
    second = HistoryService(tmp_path / "scan.db", service.output_root)
    report = second.scan_outputs()
    assert report.imported_tasks == 1
    assert len(report.corrupted_directories) == 1


def test_corrupt_summary_is_reported_by_verification(
    service: HistoryService,
    tmp_path: Path,
) -> None:
    task = BatchPredictionTask("corrupt summary", tmp_path / "unused")
    task.status = BatchStatus.COMPLETED
    paths = _touch_files(
        service.output_root / "batch_corrupt_summary",
        ("task.json", "summary.json", "predictions.csv", "errors.csv"),
    )
    exported = BatchExportResult(
        paths["task.json"].parent,
        paths["task.json"],
        paths["summary.json"],
        paths["predictions.csv"],
        paths["errors.csv"],
    )
    service.register_batch(task, exported)
    paths["summary.json"].write_text("broken", encoding="utf-8")
    assert service.verify_task(task.task_id) == IntegrityStatus.CORRUPT


def test_scan_can_be_cancelled(service: HistoryService) -> None:
    event = SimpleNamespace(is_set=lambda: True)
    report = service.scan_outputs(cancel_event=event)  # type: ignore[arg-type]
    assert report.cancelled


def test_rebuild_backs_up_and_atomically_replaces_index(
    service: HistoryService,
    tmp_path: Path,
) -> None:
    service.complete_single(_outcome(tmp_path))
    service.repository.update_task("single-task", notes="index-only note")
    backup, report = service.rebuild_index()
    assert backup.is_file()
    assert report.imported_tasks == 1
    restored = service.get_task("single-task")
    assert restored is not None
    assert restored.notes == ""


def test_rebuild_failure_preserves_original_database(
    service: HistoryService,
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    service.complete_single(_outcome(tmp_path))
    original = service.database_path.read_bytes()
    monkeypatch.setattr(
        HistoryService,
        "scan_outputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("scan failed")),
    )
    with pytest.raises(RuntimeError, match="scan failed"):
        service.rebuild_index()
    assert service.database_path.read_bytes() == original
    assert service.get_task("single-task") is not None


def test_manual_backup_preserves_rows(service: HistoryService, tmp_path: Path) -> None:
    service.complete_single(_outcome(tmp_path))
    backup = service.backup_database("acceptance")
    restored = HistoryService(backup, service.output_root)
    assert restored.list_tasks(HistoryQuery()).total == 1
