"""Seed Phase 5A acceptance records, restore them, and capture four GUI views."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import QTimer
from PySide6.QtTest import QTest

from noise_source_studio.application import apply_stylesheet, create_application
from noise_source_studio.common.paths import ApplicationPaths
from noise_source_studio.domain.batch import (
    BatchFileItem,
    BatchItemStatus,
    BatchStatus,
)
from noise_source_studio.domain.history import HistoryQuery
from noise_source_studio.domain.models import ModelRecord, PredictionOutcome
from noise_source_studio.domain.validation import (
    ValidationSample,
    ValidationSampleStatus,
    ValidationStatus,
    ValidationTask,
)
from noise_source_studio.infrastructure.config import SettingsManager
from noise_source_studio.presentation.main_window import MainWindow
from noise_source_studio.services import (
    BatchPredictionService,
    DeviceService,
    HistoryService,
    ValidationService,
)


def _paths(root: Path) -> ApplicationPaths:
    return ApplicationPaths(
        data_directory=root / "data",
        config_directory=root / "config",
        log_directory=root / "logs",
        model_directory=root / "data" / "models",
        output_directory=root / "data" / "outputs",
        resource_directory=Path(__file__).resolve().parents[1] / "resources",
    )


def _model(root: Path) -> ModelRecord:
    return ModelRecord(
        "acceptance-noise-net",
        "5A.1",
        root / "model-package",
        {"checkpoint_sha256": "acceptance-sha256", "prediction_mode": "structured"},
        datetime.now(UTC).isoformat(),
        True,
        "ok",
    )


def _seed_single_tasks(
    history: HistoryService,
    root: Path,
    model: ModelRecord,
) -> list[str]:
    task_ids: list[str] = []
    for index in range(3):
        source = root / "signals" / f"acceptance-single-{index + 1}.csv"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("DATA\n0.1,0.2,0.3\n", encoding="utf-8")
        started = datetime.now(UTC) - timedelta(milliseconds=120 + index * 10)
        task_id = f"acceptance-single-{index + 1}"
        outcome = PredictionOutcome(
            task_id=task_id,
            source_path=source,
            model=model,
            started_at=started,
            completed_at=datetime.now(UTC),
            duration_seconds=0.12 + index * 0.01,
            result={
                "runtime_version": "acceptance-runtime-1.0",
                "device": "cpu",
                "decision_mode": "structured",
                "labels": ["风机", "泵", "压缩机"],
                "label_marginal_probabilities": [0.82 - index * 0.04, 0.18, 0.08],
                "multilabel_probabilities": [0.81, 0.17, 0.07],
                "combination_labels": ["100", "010", "001", "110", "101", "011", "111"],
                "combination_probabilities": [0.72, 0.08, 0.04, 0.07, 0.04, 0.02, 0.03],
                "predicted_combination": "100",
                "predicted_sources": ["风机"],
                "decoded_label_vector": [1, 0, 0],
                "input_shape": [1, 1, 128, 64],
                "thresholds": [0.5, 0.5, 0.5],
                "thresholds_applicable": False,
            },
        )
        history.complete_single(outcome)
        history.update_notes(task_id, f"单文件验收记录 {index + 1}", ("正式验证", "CPU"))
        task_ids.append(task_id)
    return task_ids


def _seed_batch_task(
    history: HistoryService,
    batch_service: BatchPredictionService,
    root: Path,
    index: int,
    *,
    stopped: bool = False,
) -> str:
    task = batch_service.create_task(f"历史批量验收 {index}")
    task.model_name = "acceptance-noise-net"
    task.model_version = "5A.1"
    task.runtime_version = "acceptance-runtime-1.0"
    task.device = "cpu"
    task.started_at = datetime.now(UTC) - timedelta(seconds=2)
    for sequence in range(1, 4):
        source = root / "batch-signals" / f"batch-{index}-{sequence}.csv"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("DATA\n0.1,0.2,0.3\n", encoding="utf-8")
        status = BatchItemStatus.STOPPED if stopped and sequence == 3 else BatchItemStatus.SUCCESS
        item = BatchFileItem(
            sequence=sequence,
            file_path=source,
            status=status,
            status_message="已停止" if status == BatchItemStatus.STOPPED else "预测成功",
            elapsed_ms=32.0 + sequence,
            labels=["风机", "泵", "压缩机"],
            decision_mode="structured",
            display_probabilities=[0.76, 0.22, 0.06],
            predicted_combination="100" if status == BatchItemStatus.SUCCESS else "",
            predicted_sources=["风机"] if status == BatchItemStatus.SUCCESS else [],
            decoded_label_vector=[1, 0, 0] if status == BatchItemStatus.SUCCESS else [],
            result={
                "labels": ["风机", "泵", "压缩机"],
                "decision_mode": "structured",
                "label_marginal_probabilities": [0.76, 0.22, 0.06],
                "combination_labels": ["100", "010", "001"],
                "combination_probabilities": [0.76, 0.18, 0.06],
                "predicted_combination": "100",
                "predicted_sources": ["风机"],
                "decoded_label_vector": [1, 0, 0],
            }
            if status == BatchItemStatus.SUCCESS
            else None,
        )
        task.items.append(item)
    task.refresh_counts()
    task.status = BatchStatus.STOPPED if stopped else BatchStatus.COMPLETED
    task.finished_at = datetime.now(UTC)
    exported = batch_service.export_results(task)
    history.register_batch(task, exported)
    return task.task_id


def _seed_validation_task(
    history: HistoryService,
    validation_service: ValidationService,
    root: Path,
    index: int,
) -> str:
    manifest = root / "manifests" / f"validation-{index}.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("file_path,true_combination\n", encoding="utf-8")
    task = ValidationTask(
        manifest_path=manifest,
        data_root=root,
        output_directory=validation_service.output_directory,
        model_name="acceptance-noise-net",
        model_version="5A.1",
        runtime_version="acceptance-runtime-1.0",
        checkpoint_sha256="acceptance-sha256",
        device="cpu",
        labels=["风机", "泵", "压缩机"],
        decision_mode="structured",
        started_at=datetime.now(UTC) - timedelta(seconds=3),
    )
    for sequence, combination in enumerate(("100", "010", "110"), start=1):
        source = root / "validation-signals" / f"validation-{index}-{sequence}.csv"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("DATA\n0.1,0.2,0.3\n", encoding="utf-8")
        vector = [int(value) for value in combination]
        sample = ValidationSample(
            sequence=sequence,
            file_path=source,
            true_label_vector=vector,
            true_combination=combination,
            predicted_label_vector=vector,
            predicted_combination=combination,
            predicted_sources=[
                label for label, present in zip(task.labels, vector, strict=True) if present
            ],
            labels=list(task.labels),
            display_probabilities=[0.82 if value else 0.12 for value in vector],
            label_marginal_probabilities=[0.82 if value else 0.12 for value in vector],
            combination_labels=["001", "010", "011", "100", "101", "110", "111"],
            combination_probabilities=[0.03, 0.08, 0.04, 0.72, 0.03, 0.07, 0.03],
            exact_match=True,
            elapsed_ms=44.0 + sequence,
            status=ValidationSampleStatus.SUCCESS,
        )
        task.samples.append(sample)
    task.refresh_counts()
    task.status = ValidationStatus.COMPLETED
    task.finished_at = datetime.now(UTC)
    exported = validation_service.export_results(task)
    history.register_validation(task, exported)
    return task.task_id


def _capture(window: MainWindow, path: Path) -> None:
    window.repaint()
    QTest.qWait(180)
    if not window.grab().save(str(path), "PNG"):
        raise RuntimeError(f"Could not save screenshot: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hold-seconds", type=int, default=0)
    arguments = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(".pytest-tmp") / f"history-acceptance-{timestamp}"
    paths = _paths(root.resolve())
    paths.ensure_directories()
    history = HistoryService(paths.history_database, paths.output_directory)
    model = _model(root)
    single_ids = _seed_single_tasks(history, root, model)
    batch_service = BatchPredictionService(paths.output_directory)
    batch_ids = [
        _seed_batch_task(history, batch_service, root, 1),
        _seed_batch_task(history, batch_service, root, 2),
    ]
    validation_service = ValidationService(paths.output_directory)
    validation_ids = [
        _seed_validation_task(history, validation_service, root, 1),
        _seed_validation_task(history, validation_service, root, 2),
    ]
    history.register_single_running("acceptance-failed", root / "failed.csv", model)
    history.fail_task("acceptance-failed", ValueError("验收用输入格式错误"))
    stopped_id = _seed_batch_task(history, batch_service, root, 3, stopped=True)

    # Prove persistence by constructing a fresh service over the same database.
    restarted = HistoryService(paths.history_database, paths.output_directory)
    before_maintenance = restarted.list_tasks(HistoryQuery(page_size=20)).total
    deleted_directory = Path(restarted.get_task(single_ids[0]).result_directory)  # type: ignore[union-attr]
    restarted.delete_index(single_ids[0])
    index_only_kept_files = deleted_directory.is_dir()
    scan_report = restarted.scan_outputs()
    backup = restarted.backup_database("gui-acceptance")
    total = restarted.list_tasks(HistoryQuery(page_size=20)).total
    restored_single = restarted.load_single_prediction(single_ids[1])

    app = create_application([])
    apply_stylesheet(app)
    manager = SettingsManager(paths)
    settings = manager.load().model_copy(
        update={"device_preference": "cpu", "window_width": 1440, "window_height": 900}
    )
    manager.save(settings)
    cpu_report = DeviceService(SimpleNamespace()).cpu_report()
    window = MainWindow(
        settings,
        manager,
        paths.log_file,
        history_service=restarted,
        batch_prediction_service=batch_service,
        validation_service=validation_service,
        device_service=DeviceService(SimpleNamespace(probe_devices=lambda: cpu_report)),
    )
    window.resize(1440, 900)
    window.show()
    QTest.qWait(250)

    screenshots = root / "screenshots"
    screenshots.mkdir(parents=True, exist_ok=True)
    result = restarted.list_tasks(HistoryQuery(page_size=50))
    window._history_page.show_results(result, restarted.overview())
    window.navigation.select_page(5)
    _capture(window, screenshots / "01-task-list.png")
    window._history_page.table.selectRow(0)
    _capture(window, screenshots / "02-task-detail.png")

    batch_record = restarted.get_task(batch_ids[0])
    assert batch_record is not None
    restored_batch = batch_service.load_history(Path(batch_record.result_directory))
    window.batch_task = restored_batch
    window._batch_prediction_page.show_history(restored_batch)
    window.navigation.select_page(2)
    _capture(window, screenshots / "03-batch-history-restored.png")

    validation_record = restarted.get_task(validation_ids[0])
    assert validation_record is not None
    restored_validation = validation_service.load_history(
        Path(validation_record.result_directory)
    )
    window.validation_task = restored_validation
    window._validation_page.show_history(restored_validation)
    window.navigation.select_page(3)
    _capture(window, screenshots / "04-validation-history-restored.png")

    report = {
        "root": str(root.resolve()),
        "database": str(paths.history_database),
        "task_count_before_maintenance": before_maintenance,
        "task_count_after_restart_and_scan": total,
        "expected_task_count": 9,
        "single_task_ids": single_ids,
        "batch_task_ids": batch_ids,
        "validation_task_ids": validation_ids,
        "stopped_task_id": stopped_id,
        "index_only_delete_kept_files": index_only_kept_files,
        "single_restore_without_inference": (
            restored_single.task_id == single_ids[1]
            and restored_single.result.get("predicted_combination") == "100"
        ),
        "scan_imported": scan_report.imported_tasks,
        "backup": str(backup),
        "backup_exists": backup.is_file(),
        "screenshots": [str(path.resolve()) for path in sorted(screenshots.glob("*.png"))],
    }
    report_path = root / "acceptance.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    if arguments.hold_seconds > 0:
        QTimer.singleShot(arguments.hold_seconds * 1000, app.quit)
        return app.exec()
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
