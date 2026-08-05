"""Unified task-history registration, recovery, scanning, and maintenance."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any
from uuid import uuid4

from noise_source_studio.domain.batch import (
    BatchExportResult,
    BatchPredictionTask,
    BatchStatus,
)
from noise_source_studio.domain.history import (
    HistoryArtifact,
    HistoryOverview,
    HistoryPageResult,
    HistoryQuery,
    HistoryScanReport,
    HistoryStatus,
    IntegrityStatus,
    TaskHistoryRecord,
    TaskType,
)
from noise_source_studio.domain.models import ModelRecord, PredictionOutcome
from noise_source_studio.domain.validation import (
    ValidationExportResult,
    ValidationStatus,
    ValidationTask,
)
from noise_source_studio.infrastructure.history import HistoryDatabase, HistoryRepository
from noise_source_studio.services.result_adapter import (
    normalize_prediction_result,
    primary_probability_summary,
)
from noise_source_studio.version import __version__

LOGGER = logging.getLogger("noise_source_studio.history_service")
ProgressCallback = Callable[[int, int, str], None]


class UnsafeHistoryDeleteError(ValueError):
    """Raised when a result folder is outside the configured output root."""


class HistoryService:
    """Coordinate the SQLite index while full results remain on disk."""

    def __init__(
        self,
        database_path: Path,
        output_root: Path,
        *,
        model_root: Path | None = None,
        initialize: bool = True,
    ) -> None:
        self.database = HistoryDatabase(Path(database_path))
        self.repository = HistoryRepository(self.database)
        self.output_root = Path(output_root).resolve()
        self.model_root = (
            Path(model_root).resolve()
            if model_root is not None
            else (self.output_root.parent / "models").resolve()
        )
        if initialize:
            schema_version = self.repository.initialize()
            LOGGER.info(
                "History database ready | path=%s | schema_version=%s",
                self.database.path,
                schema_version,
            )
            interrupted = self.repository.interrupt_running(_now())
            if interrupted:
                LOGGER.warning("Marked stale history tasks interrupted | count=%s", interrupted)

    @property
    def database_path(self) -> Path:
        return self.database.path

    def register_task(
        self,
        record: TaskHistoryRecord,
        artifacts: tuple[HistoryArtifact, ...] = (),
    ) -> None:
        """Register generic metadata for import tools and future task kinds."""
        self.repository.upsert_task(record, artifacts)
        LOGGER.info(
            "History task registered | task_id=%s | type=%s",
            record.task_id,
            record.task_type,
        )

    def update_task(self, task_id: str, **changes: Any) -> bool:
        """Update whitelisted task metadata."""
        return self.repository.update_task(task_id, **changes)

    def complete_task(
        self,
        task_id: str,
        *,
        primary_summary: str = "",
        completed_with_errors: bool = False,
    ) -> bool:
        """Finalize a generic task when a specialized adapter is unnecessary."""
        return self.repository.update_task(
            task_id,
            status=(
                HistoryStatus.COMPLETED_WITH_ERRORS
                if completed_with_errors
                else HistoryStatus.COMPLETED
            ),
            finished_at=_now(),
            primary_summary=primary_summary,
        )

    def interrupt_stale_tasks(self) -> int:
        """Mark tasks left running by a previous process as interrupted."""
        return self.repository.interrupt_running(_now())

    def search_tasks(
        self,
        keyword: str,
        *,
        page: int = 1,
        page_size: int = 50,
    ) -> HistoryPageResult:
        return self.list_tasks(
            HistoryQuery(keyword=keyword, page=page, page_size=page_size)
        )

    def delete_record(self, task_id: str) -> bool:
        return self.delete_index(task_id)

    def delete_record_and_files(self, task_id: str) -> bool:
        return self.delete_task_files(task_id)

    def verify_artifacts(self, task_id: str) -> IntegrityStatus:
        return self.verify_task(task_id)

    def scan_result_directory(
        self,
        directory: Path | None = None,
        *,
        cancel_event: Event | None = None,
        progress: ProgressCallback | None = None,
    ) -> HistoryScanReport:
        root = Path(directory).resolve() if directory is not None else self.output_root
        if root != self.output_root:
            raise ValueError("扫描目录必须是当前配置的输出根目录。")
        return self.scan_outputs(cancel_event=cancel_event, progress=progress)

    def export_history_list(self, query: HistoryQuery, destination: Path) -> Path:
        return self.export_summary(query, destination)

    def register_single_running(
        self,
        task_id: str,
        source_path: Path,
        model: ModelRecord,
    ) -> TaskHistoryRecord:
        """Register a single prediction before inference starts."""
        source = Path(source_path).resolve()
        now = _now()
        record = TaskHistoryRecord(
            task_id=task_id,
            task_type=TaskType.SINGLE,
            task_name=source.name,
            status=HistoryStatus.RUNNING,
            created_at=now,
            started_at=now,
            model_name=model.model_name,
            model_version=model.model_version,
            model_identifier=model.identifier,
            model_package_sha256=str(model.manifest.get("checkpoint_sha256", "")),
            application_version=__version__,
            source_description=source.name,
            source_path=str(source),
            total_count=1,
        )
        self.repository.upsert_task(record)
        return record

    def complete_single(self, outcome: PredictionOutcome) -> TaskHistoryRecord:
        """Persist a successful single prediction and finalize its index row."""
        result = normalize_prediction_result(outcome.result)
        output = (
            self.output_root
            / "single"
            / outcome.completed_at.astimezone().strftime("%Y-%m-%d")
            / outcome.task_id
        )
        output.mkdir(parents=True, exist_ok=True)
        task_path = output / "task.json"
        prediction_path = output / "prediction.json"
        task_payload = {
            "schema_version": 1,
            "task": {
                "task_id": outcome.task_id,
                "task_type": TaskType.SINGLE.value,
                "created_at": outcome.started_at.isoformat(),
                "started_at": outcome.started_at.isoformat(),
                "finished_at": outcome.completed_at.isoformat(),
                "duration_seconds": outcome.duration_seconds,
                "source_path": str(outcome.source_path),
                "status": HistoryStatus.COMPLETED.value,
            },
            "model": outcome.model.to_dict(),
            "result_files": {"prediction": str(prediction_path)},
        }
        _write_json_atomic(task_path, task_payload)
        _write_json_atomic(prediction_path, result)
        previous = self.repository.get_task(outcome.task_id)
        predicted = str(result.get("predicted_combination", ""))
        sources = ", ".join(str(value) for value in result.get("predicted_sources", []) or [])
        summary_parts = [value for value in (predicted, sources) if value]
        probability = primary_probability_summary(result)
        if probability != "—":
            summary_parts.append(probability)
        record = TaskHistoryRecord(
            task_id=outcome.task_id,
            task_type=TaskType.SINGLE,
            task_name=outcome.source_path.name,
            status=HistoryStatus.COMPLETED,
            created_at=outcome.started_at.isoformat(),
            started_at=outcome.started_at.isoformat(),
            finished_at=outcome.completed_at.isoformat(),
            duration_ms=outcome.duration_seconds * 1000.0,
            model_name=outcome.model.model_name,
            model_version=outcome.model.model_version,
            model_identifier=outcome.model.identifier,
            model_package_sha256=str(
                outcome.model.manifest.get("checkpoint_sha256", "")
            ),
            runtime_version=str(result.get("runtime_version", "")),
            application_version=__version__,
            device=str(result.get("device", "")),
            decision_mode=str(result.get("decision_mode", "")),
            total_count=1,
            success_count=1,
            primary_summary=" · ".join(summary_parts),
            source_description=outcome.source_path.name,
            source_path=str(outcome.source_path.resolve()),
            result_directory=self._store_path(output),
            task_file=self._store_path(task_path),
            detail_file=self._store_path(prediction_path),
            integrity_status=IntegrityStatus.OK,
            tags=previous.tags if previous else (),
            notes=previous.notes if previous else "",
        )
        artifacts = self._artifacts(
            {"task_json": task_path, "prediction_json": prediction_path},
        )
        self.repository.upsert_task(record, artifacts)
        LOGGER.info("Single history completed | task_id=%s | output=%s", outcome.task_id, output)
        return self._resolve_record(record)

    def register_batch_running(self, task: BatchPredictionTask) -> TaskHistoryRecord:
        """Register or update a batch when its worker starts."""
        record = self._batch_record(task, status=HistoryStatus.RUNNING)
        self.repository.upsert_task(record)
        return record

    def register_batch(
        self,
        task: BatchPredictionTask,
        exported: BatchExportResult,
    ) -> TaskHistoryRecord:
        """Index the existing automatic batch export without exporting again."""
        record = replace(
            self._batch_record(task),
            result_directory=self._store_path(exported.output_directory),
            task_file=self._store_path(exported.task_path),
            summary_file=self._store_path(exported.summary_path),
            detail_file=self._store_path(exported.predictions_path),
            error_file=self._store_path(exported.errors_path),
            integrity_status=self._directory_integrity(
                exported.output_directory,
                ("task.json", "summary.json", "predictions.csv", "errors.csv"),
            ),
        )
        artifacts = self._artifacts(
            {
                "task_json": exported.task_path,
                "summary_json": exported.summary_path,
                "predictions_csv": exported.predictions_path,
                "errors_csv": exported.errors_path,
            }
        )
        self.repository.upsert_task(record, artifacts)
        LOGGER.info("Batch history registered | task_id=%s", task.task_id)
        return self._resolve_record(record)

    def register_validation_running(self, task: ValidationTask) -> TaskHistoryRecord:
        """Register or update a validation when its worker starts."""
        record = self._validation_record(task, status=HistoryStatus.RUNNING)
        self.repository.upsert_task(record)
        return record

    def register_validation(
        self,
        task: ValidationTask,
        exported: ValidationExportResult,
    ) -> TaskHistoryRecord:
        """Index the existing validation export without exporting again."""
        record = replace(
            self._validation_record(task),
            result_directory=self._store_path(exported.output_directory),
            task_file=self._store_path(exported.task_path),
            summary_file=self._store_path(exported.summary_path),
            detail_file=self._store_path(exported.sample_results_path),
            manifest_file=self._store_path(task.manifest_path),
            error_file=self._store_path(exported.errors_path),
            report_file=self._store_path(exported.report_path),
            integrity_status=self._directory_integrity(
                exported.output_directory,
                (
                    "task.json",
                    "summary.json",
                    "sample_results.csv",
                    "label_metrics.csv",
                    "combination_metrics.csv",
                    "confusion_matrix.csv",
                    "group_metrics.csv",
                    "errors.csv",
                    "report.html",
                ),
            ),
        )
        artifacts = self._artifacts(
            {
                "task_json": exported.task_path,
                "summary_json": exported.summary_path,
                "sample_results_csv": exported.sample_results_path,
                "label_metrics_csv": exported.label_metrics_path,
                "combination_metrics_csv": exported.combination_metrics_path,
                "confusion_matrix_csv": exported.confusion_matrix_path,
                "group_metrics_csv": exported.group_metrics_path,
                "errors_csv": exported.errors_path,
                "html_report": exported.report_path,
            }
        )
        self.repository.upsert_task(record, artifacts)
        LOGGER.info("Validation history registered | task_id=%s", task.task_id)
        return self._resolve_record(record)

    def fail_task(
        self,
        task_id: str,
        error: BaseException | str,
        *,
        status: HistoryStatus = HistoryStatus.FAILED,
    ) -> bool:
        """Finalize a task failure without inventing result artifacts."""
        error_type = type(error).__name__ if isinstance(error, BaseException) else "Error"
        message = str(error)
        updated = self.repository.update_task(
            task_id,
            status=status,
            finished_at=_now(),
            failed_count=1,
            error_type=error_type,
            error_message=message,
            primary_summary=message,
        )
        LOGGER.warning("History task failed | task_id=%s | error_type=%s", task_id, error_type)
        return updated

    def interrupt_task(self, task_id: str, message: str = "任务已中断") -> bool:
        return self.repository.update_task(
            task_id,
            status=HistoryStatus.INTERRUPTED,
            finished_at=_now(),
            primary_summary=message,
        )

    def get_task(self, task_id: str) -> TaskHistoryRecord | None:
        record = self.repository.get_task(task_id)
        return self._resolve_record(record) if record is not None else None

    def list_tasks(self, query: HistoryQuery) -> HistoryPageResult:
        result = self.repository.list_tasks(query)
        return replace(
            result,
            records=tuple(self._resolve_record(record) for record in result.records),
        )

    def overview(self) -> HistoryOverview:
        return self.repository.overview()

    def update_notes(self, task_id: str, notes: str, tags: tuple[str, ...]) -> bool:
        clean_tags = tuple(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))
        return self.repository.update_task(task_id, notes=notes.strip(), tags=clean_tags)

    def mark_opened(self, task_id: str) -> bool:
        return self.repository.update_task(task_id, last_opened_at=_now())

    def delete_index(self, task_id: str) -> bool:
        deleted = self.repository.delete_task(task_id)
        LOGGER.info("History index deleted | task_id=%s | deleted=%s", task_id, deleted)
        return deleted

    def delete_task_files(self, task_id: str) -> bool:
        """Delete an indexed result directory only when it is under output_root."""
        record = self.get_task(task_id)
        if record is None:
            return False
        result = Path(record.result_directory).resolve() if record.result_directory else None
        if (
            result is None
            or result == self.output_root
            or not result.is_relative_to(self.output_root)
            or result == self.model_root
            or result.is_relative_to(self.model_root)
        ):
            raise UnsafeHistoryDeleteError("仅允许删除当前输出目录内的任务结果文件。")
        try:
            if result.exists():
                shutil.rmtree(result)
        except OSError as exc:
            self.repository.update_task(
                task_id,
                integrity_status=IntegrityStatus.PARTIAL,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            LOGGER.exception("History files delete failed | task_id=%s", task_id)
            raise
        deleted = self.repository.delete_task(task_id)
        LOGGER.info("History files deleted | task_id=%s | directory=%s", task_id, result)
        return deleted

    def verify_task(self, task_id: str) -> IntegrityStatus:
        """Verify existence/readability and refresh artifact metadata."""
        record = self.repository.get_task(task_id)
        if record is None:
            raise KeyError(task_id)
        artifacts = self.repository.artifacts(task_id)
        if not artifacts:
            paths = self._record_artifact_paths(record)
            artifacts = self._artifacts(paths, hash_files=False)
        updated: list[HistoryArtifact] = []
        status = IntegrityStatus.OK
        for artifact in artifacts:
            path = self._resolve_path(artifact.path)
            artifact_status = IntegrityStatus.OK
            if not path.is_file():
                artifact_status = IntegrityStatus.MISSING
                status = IntegrityStatus.MISSING
            elif path.suffix.casefold() == ".json":
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    artifact_status = IntegrityStatus.CORRUPT
                    status = IntegrityStatus.CORRUPT
            if (
                artifact_status == IntegrityStatus.OK
                and artifact.sha256
                and _sha256(path) != artifact.sha256
            ):
                artifact_status = IntegrityStatus.CORRUPT
                status = IntegrityStatus.CORRUPT
            updated.append(
                replace(
                    artifact,
                    exists_status=artifact_status.value,
                    file_size=path.stat().st_size if path.is_file() else None,
                    sha256=(
                        _sha256(path)
                        if artifact_status == IntegrityStatus.OK and path.is_file()
                        else artifact.sha256
                    ),
                )
            )
        self.repository.upsert_task(replace(record, integrity_status=status), updated)
        LOGGER.info("History artifacts verified | task_id=%s | status=%s", task_id, status)
        return status

    def export_summary(self, query: HistoryQuery, destination: Path) -> Path:
        """Export all current filtered records as an Excel-friendly UTF-8-BOM CSV."""
        records = self.repository.list_all(replace(query, page=1))
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        columns = (
            "task_id",
            "task_type",
            "task_name",
            "status",
            "created_at",
            "finished_at",
            "duration_ms",
            "model_name",
            "model_version",
            "device",
            "total_count",
            "success_count",
            "failed_count",
            "primary_summary",
            "result_directory",
            "integrity_status",
            "tags",
            "notes",
        )
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for record in records:
                row = {column: getattr(record, column) for column in columns}
                row["task_type"] = record.task_type.value
                row["status"] = record.status.value
                row["integrity_status"] = record.integrity_status.value
                row["tags"] = ",".join(record.tags)
                row["result_directory"] = str(
                    self._resolve_path(record.result_directory)
                    if record.result_directory
                    else ""
                )
                writer.writerow(row)
        os.replace(temporary, path)
        return path

    def backup_database(self, reason: str = "manual") -> Path:
        return self.database.backup(reason)

    def load_single_prediction(self, task_id: str) -> PredictionOutcome:
        """Restore one saved single result without invoking model inference."""
        record = self.get_task(task_id)
        if record is None or record.task_type != TaskType.SINGLE:
            raise KeyError(task_id)
        task_payload = json.loads(Path(record.task_file).read_text(encoding="utf-8"))
        result = json.loads(Path(record.detail_file).read_text(encoding="utf-8"))
        task = task_payload["task"]
        return PredictionOutcome(
            task_id=task_id,
            source_path=Path(task["source_path"]),
            model=ModelRecord.from_dict(task_payload["model"]),
            started_at=_parse_datetime(task["started_at"]),
            completed_at=_parse_datetime(task["finished_at"]),
            duration_seconds=float(task.get("duration_seconds", 0.0)),
            result=result,
        )

    def scan_outputs(
        self,
        *,
        cancel_event: Event | None = None,
        progress: ProgressCallback | None = None,
    ) -> HistoryScanReport:
        """Scan existing output contracts, deduplicate, and continue past corruption."""
        task_files = (
            sorted(self.output_root.rglob("task.json"))
            if self.output_root.exists()
            else []
        )
        if cancel_event is not None and cancel_event.is_set():
            return HistoryScanReport(len(task_files), 0, 0, 0, (), True)
        existing = {record.task_id for record in self.repository.list_all(HistoryQuery())}
        records: list[TaskHistoryRecord] = []
        corrupted: list[str] = []
        skipped = 0
        imported = 0
        updated = 0
        for index, task_file in enumerate(task_files, start=1):
            if cancel_event is not None and cancel_event.is_set():
                return HistoryScanReport(
                    len(task_files), imported, updated, skipped, tuple(corrupted), True
                )
            if progress is not None:
                progress(index, len(task_files), str(task_file.parent))
            try:
                record = self._record_from_result_directory(task_file.parent)
            except (OSError, UnicodeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                corrupted.append(f"{task_file.parent}: {exc}")
                continue
            if record is None:
                skipped += 1
                continue
            records.append(record)
            if record.task_id in existing:
                updated += 1
            else:
                imported += 1
        self.repository.upsert_many(records)
        report = HistoryScanReport(
            len(task_files), imported, updated, skipped, tuple(corrupted), False
        )
        LOGGER.info(
            "History scan completed | scanned=%s | imported=%s | updated=%s | corrupt=%s",
            report.scanned_directories,
            report.imported_tasks,
            report.updated_tasks,
            len(report.corrupted_directories),
        )
        return report

    def rebuild_index(
        self,
        *,
        cancel_event: Event | None = None,
        progress: ProgressCallback | None = None,
    ) -> tuple[Path, HistoryScanReport]:
        """Rebuild into a temporary database and atomically replace only on success."""
        backup = self.backup_database("pre-rebuild")
        temporary = self.database.path.parent / f"history-rebuild-{uuid4().hex}.db"
        temporary_service = HistoryService(temporary, self.output_root)
        try:
            report = temporary_service.scan_outputs(
                cancel_event=cancel_event,
                progress=progress,
            )
            if report.cancelled:
                raise RuntimeError("历史索引重建已取消，原数据库保持不变。")
            with temporary_service.database.connect() as connection:
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            os.replace(temporary, self.database.path)
        except Exception:
            temporary.unlink(missing_ok=True)
            temporary.with_name(temporary.name + "-wal").unlink(missing_ok=True)
            temporary.with_name(temporary.name + "-shm").unlink(missing_ok=True)
            raise
        LOGGER.info(
            "History index rebuilt | backup=%s | imported=%s",
            backup,
            report.imported_tasks,
        )
        return backup, report

    def _record_from_result_directory(self, directory: Path) -> TaskHistoryRecord | None:
        task_payload = json.loads((directory / "task.json").read_text(encoding="utf-8"))
        task = dict(task_payload.get("task", {}))
        if not task:
            return None
        declared_type = str(task.get("task_type", task_payload.get("task_type", "")))
        if (directory / "prediction.json").is_file():
            return self._single_record_from_payload(directory, task_payload)
        if (directory / "predictions.csv").is_file():
            return self._batch_record_from_payload(directory, task_payload)
        if (directory / "sample_results.csv").is_file():
            return self._validation_record_from_payload(directory, task_payload)
        directory_name = directory.name.casefold()
        if declared_type == TaskType.SINGLE.value or "single" in {
            part.casefold() for part in directory.parts
        }:
            raise ValueError("单文件历史结果缺少 prediction.json")
        if declared_type == TaskType.BATCH.value or directory_name.startswith("batch_"):
            raise ValueError("批量历史结果缺少 predictions.csv")
        if (
            declared_type == TaskType.VALIDATION.value
            or directory_name.startswith("validation_")
        ):
            raise ValueError("验证历史结果缺少 sample_results.csv")
        return None

    def _single_record_from_payload(
        self,
        directory: Path,
        payload: dict[str, Any],
    ) -> TaskHistoryRecord:
        task = payload["task"]
        model = payload.get("model", {})
        result = json.loads((directory / "prediction.json").read_text(encoding="utf-8"))
        task_id, legacy = self._task_identifier(task, directory)
        source = str(task.get("source_path", ""))
        return TaskHistoryRecord(
            task_id=task_id,
            task_type=TaskType.SINGLE,
            task_name=Path(source).name or f"单文件任务 {task_id[:8]}",
            status=self._history_status(str(task.get("status", "completed"))),
            created_at=str(task.get("created_at", task.get("started_at", _now()))),
            started_at=str(task.get("started_at", "")),
            finished_at=str(task.get("finished_at", "")),
            duration_ms=float(task.get("duration_seconds", 0.0) or 0.0) * 1000.0,
            model_name=str(model.get("model_name", "")),
            model_version=str(model.get("model_version", "")),
            model_identifier=self._model_identifier(model),
            model_package_sha256=str(model.get("manifest", {}).get("checkpoint_sha256", "")),
            runtime_version=str(result.get("runtime_version", "")),
            application_version=__version__,
            device=str(result.get("device", "")),
            decision_mode=str(result.get("decision_mode", "")),
            total_count=1,
            success_count=1,
            primary_summary=primary_probability_summary(result),
            source_description=Path(source).name,
            source_path=source,
            result_directory=self._store_path(directory),
            task_file=self._store_path(directory / "task.json"),
            detail_file=self._store_path(directory / "prediction.json"),
            integrity_status=self._directory_integrity(
                directory,
                ("task.json", "prediction.json"),
            ),
            created_by_application=not legacy,
            imported_from_legacy=legacy,
        )

    def _batch_record_from_payload(
        self,
        directory: Path,
        payload: dict[str, Any],
    ) -> TaskHistoryRecord:
        task = payload["task"]
        model = payload.get("model", {})
        task_id, legacy = self._task_identifier(task, directory)
        items = list(payload.get("items", []))
        statuses = [str(item.get("status", "")) for item in items]
        success = statuses.count("success")
        failed = statuses.count("failed")
        skipped = statuses.count("skipped")
        stopped = statuses.count("stopped")
        return TaskHistoryRecord(
            task_id=task_id,
            task_type=TaskType.BATCH,
            task_name=str(task.get("name", f"批量任务 {task_id[:8]}")),
            status=self._history_status(str(task.get("status", "completed"))),
            created_at=str(task.get("created_at", _now())),
            started_at=str(task.get("started_at", "")),
            finished_at=str(task.get("finished_at", "")),
            duration_ms=_duration_between(
                str(task.get("started_at", "")),
                str(task.get("finished_at", "")),
            ) * 1000.0,
            model_name=str(model.get("model_name", "")),
            model_version=str(model.get("model_version", "")),
            model_identifier=self._model_identifier(model),
            runtime_version=str(model.get("runtime_version", "")),
            application_version=__version__,
            device=str(model.get("device", "")),
            total_count=len(items),
            success_count=success,
            failed_count=failed,
            skipped_count=skipped,
            stopped_count=stopped,
            primary_summary=self._batch_payload_summary(items, success, failed),
            source_description="; ".join(task.get("source_directories", []) or []),
            result_directory=self._store_path(directory),
            task_file=self._store_path(directory / "task.json"),
            summary_file=self._store_path(directory / "summary.json"),
            detail_file=self._store_path(directory / "predictions.csv"),
            error_file=self._store_path(directory / "errors.csv"),
            integrity_status=self._directory_integrity(
                directory,
                ("task.json", "summary.json", "predictions.csv", "errors.csv"),
            ),
            created_by_application=not legacy,
            imported_from_legacy=legacy,
        )

    def _validation_record_from_payload(
        self,
        directory: Path,
        payload: dict[str, Any],
    ) -> TaskHistoryRecord:
        task = payload["task"]
        model = payload.get("model", {})
        metrics = payload.get("metrics", {})
        samples = list(payload.get("samples", []))
        statuses = [str(sample.get("status", "")) for sample in samples]
        task_id, legacy = self._task_identifier(task, directory)
        success = statuses.count("success")
        failed = statuses.count("inference_failed")
        return TaskHistoryRecord(
            task_id=task_id,
            task_type=TaskType.VALIDATION,
            task_name=f"模型验证 {Path(str(task.get('manifest_path', ''))).name}",
            status=self._history_status(str(task.get("status", "completed"))),
            created_at=str(task.get("created_at", _now())),
            started_at=str(task.get("started_at", "")),
            finished_at=str(task.get("finished_at", "")),
            duration_ms=_duration_between(
                str(task.get("started_at", "")),
                str(task.get("finished_at", "")),
            ) * 1000.0,
            model_name=str(model.get("model_name", "")),
            model_version=str(model.get("model_version", "")),
            model_identifier=self._model_identifier(model),
            model_package_sha256=str(model.get("checkpoint_sha256", "")),
            runtime_version=str(model.get("runtime_version", "")),
            application_version=__version__,
            device=str(model.get("device", "")),
            decision_mode=str(model.get("decision_mode", "")),
            total_count=len(samples),
            success_count=success,
            failed_count=failed,
            skipped_count=statuses.count("skipped"),
            stopped_count=statuses.count("stopped"),
            primary_summary=self._validation_summary(metrics, success, failed),
            source_description=str(task.get("manifest_path", "")),
            source_path=str(task.get("manifest_path", "")),
            result_directory=self._store_path(directory),
            task_file=self._store_path(directory / "task.json"),
            summary_file=self._store_path(directory / "summary.json"),
            detail_file=self._store_path(directory / "sample_results.csv"),
            error_file=self._store_path(directory / "errors.csv"),
            report_file=self._store_path(directory / "report.html"),
            manifest_file=self._store_path(Path(str(task.get("manifest_path", "")))),
            integrity_status=self._directory_integrity(
                directory,
                (
                    "task.json",
                    "summary.json",
                    "sample_results.csv",
                    "label_metrics.csv",
                    "combination_metrics.csv",
                    "confusion_matrix.csv",
                    "group_metrics.csv",
                    "errors.csv",
                    "report.html",
                ),
            ),
            created_by_application=not legacy,
            imported_from_legacy=legacy,
        )

    def _batch_record(
        self,
        task: BatchPredictionTask,
        *,
        status: HistoryStatus | None = None,
    ) -> TaskHistoryRecord:
        previous = self.repository.get_task(task.task_id)
        return TaskHistoryRecord(
            task_id=task.task_id,
            task_type=TaskType.BATCH,
            task_name=task.name,
            status=status or self._history_status(task.status.value),
            created_at=task.created_at.isoformat(),
            started_at=_iso(task.started_at),
            finished_at=_iso(task.finished_at),
            duration_ms=task.elapsed_seconds * 1000.0,
            model_name=task.model_name,
            model_version=task.model_version,
            model_identifier=(
                f"{task.model_name}@{task.model_version}" if task.model_name else ""
            ),
            runtime_version=task.runtime_version,
            application_version=__version__,
            device=task.device,
            decision_mode=self._task_decision_mode(task),
            total_count=task.total_count,
            success_count=task.success_count,
            failed_count=task.failed_count,
            skipped_count=task.skipped_count,
            stopped_count=task.stopped_count,
            primary_summary=self._batch_task_summary(task),
            source_description="; ".join(task.source_directories),
            tags=previous.tags if previous else (),
            notes=previous.notes if previous else "",
        )

    def _validation_record(
        self,
        task: ValidationTask,
        *,
        status: HistoryStatus | None = None,
    ) -> TaskHistoryRecord:
        previous = self.repository.get_task(task.task_id)
        return TaskHistoryRecord(
            task_id=task.task_id,
            task_type=TaskType.VALIDATION,
            task_name=f"模型验证 {task.manifest_path.name}",
            status=status or self._history_status(task.status.value),
            created_at=task.created_at.isoformat(),
            started_at=_iso(task.started_at),
            finished_at=_iso(task.finished_at),
            duration_ms=task.elapsed_seconds * 1000.0,
            model_name=task.model_name,
            model_version=task.model_version,
            model_identifier=(
                f"{task.model_name}@{task.model_version}" if task.model_name else ""
            ),
            model_package_sha256=task.checkpoint_sha256,
            runtime_version=task.runtime_version,
            application_version=__version__,
            device=task.device,
            decision_mode=task.decision_mode,
            total_count=task.total_count,
            success_count=task.success_count,
            failed_count=task.inference_failed_count,
            skipped_count=task.skipped_count,
            stopped_count=task.stopped_count,
            primary_summary=self._validation_summary(
                task.metrics,
                task.success_count,
                task.inference_failed_count,
            ),
            source_description=str(task.manifest_path),
            source_path=str(task.manifest_path.resolve()),
            manifest_file=self._store_path(task.manifest_path),
            tags=previous.tags if previous else (),
            notes=previous.notes if previous else "",
        )

    @staticmethod
    def _history_status(status: str) -> HistoryStatus:
        mapping = {
            BatchStatus.CREATED.value: HistoryStatus.CREATED,
            BatchStatus.RUNNING.value: HistoryStatus.RUNNING,
            BatchStatus.PAUSED.value: HistoryStatus.RUNNING,
            BatchStatus.STOPPING.value: HistoryStatus.RUNNING,
            BatchStatus.STOPPED.value: HistoryStatus.STOPPED,
            BatchStatus.COMPLETED.value: HistoryStatus.COMPLETED,
            BatchStatus.COMPLETED_WITH_ERRORS.value: HistoryStatus.COMPLETED_WITH_ERRORS,
            BatchStatus.FAILED.value: HistoryStatus.FAILED,
            ValidationStatus.CHECKED.value: HistoryStatus.CREATED,
        }
        return mapping.get(status, HistoryStatus.FAILED)

    @staticmethod
    def _validation_summary(metrics: dict[str, Any], success: int, failed: int) -> str:
        overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
        parts = [f"成功 {success}", f"失败 {failed}"]
        for key, label in (
            ("exact_match_accuracy", "Exact Match"),
            ("micro_f1", "Micro F1"),
            ("macro_f1", "Macro F1"),
        ):
            value = overall.get(key)
            if value is not None:
                parts.append(f"{label} {float(value) * 100:.2f}%")
        label_rows = metrics.get("labels", []) if isinstance(metrics, dict) else []
        comparable_labels = [row for row in label_rows if row.get("f1") is not None]
        if comparable_labels:
            worst = min(comparable_labels, key=lambda row: float(row["f1"]))
            parts.append(f"最差标签 {worst.get('label', '—')}")
        combination_rows = metrics.get("combinations", []) if isinstance(metrics, dict) else []
        comparable_combinations = [
            row for row in combination_rows if row.get("exact_accuracy") is not None
        ]
        if comparable_combinations:
            worst = min(
                comparable_combinations,
                key=lambda row: float(row["exact_accuracy"]),
            )
            parts.append(f"最差组合 {worst.get('true_combination', '—')}")
        return " · ".join(parts)

    @staticmethod
    def _batch_task_summary(task: BatchPredictionTask) -> str:
        combinations = Counter(
            item.predicted_combination for item in task.items if item.predicted_combination
        )
        distribution = ", ".join(
            f"{name} {count}" for name, count in combinations.most_common(3)
        )
        average = task.average_item_seconds
        parts = [
            f"成功 {task.success_count}",
            f"失败 {task.failed_count}",
            f"共 {task.total_count}",
        ]
        if distribution:
            parts.append(f"组合 {distribution}")
        if average is not None:
            parts.append(f"平均 {average * 1000.0:.1f} ms")
        return " · ".join(parts)

    @staticmethod
    def _batch_payload_summary(
        items: list[dict[str, Any]],
        success: int,
        failed: int,
    ) -> str:
        combinations = Counter(
            str(item.get("predicted_combination", ""))
            for item in items
            if item.get("predicted_combination")
        )
        elapsed = [
            float(item["elapsed_ms"])
            for item in items
            if item.get("elapsed_ms") is not None
        ]
        parts = [f"成功 {success}", f"失败 {failed}", f"共 {len(items)}"]
        if combinations:
            parts.append(
                "组合 "
                + ", ".join(f"{name} {count}" for name, count in combinations.most_common(3))
            )
        if elapsed:
            parts.append(f"平均 {sum(elapsed) / len(elapsed):.1f} ms")
        return " · ".join(parts)

    @staticmethod
    def _task_decision_mode(task: BatchPredictionTask) -> str:
        return next((item.decision_mode for item in task.items if item.decision_mode), "")

    @staticmethod
    def _model_identifier(model: dict[str, Any]) -> str:
        name = str(model.get("model_name", ""))
        version = str(model.get("model_version", ""))
        return f"{name}@{version}" if name else ""

    @staticmethod
    def _directory_integrity(
        directory: Path,
        required_files: tuple[str, ...],
    ) -> IntegrityStatus:
        return (
            IntegrityStatus.OK
            if all((directory / name).is_file() for name in required_files)
            else IntegrityStatus.MISSING
        )

    @staticmethod
    def _task_identifier(task: dict[str, Any], directory: Path) -> tuple[str, bool]:
        task_id = str(task.get("task_id", "")).strip()
        if task_id:
            return task_id, False
        digest = hashlib.sha256(str(directory.resolve()).casefold().encode()).hexdigest()
        return f"legacy-{digest[:32]}", True

    def _store_path(self, path: Path) -> str:
        resolved = Path(path).resolve()
        try:
            return str(resolved.relative_to(self.output_root))
        except ValueError:
            return str(resolved)

    def _resolve_path(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.output_root / candidate).resolve()

    def _resolve_record(self, record: TaskHistoryRecord) -> TaskHistoryRecord:
        path_fields = (
            "result_directory",
            "task_file",
            "summary_file",
            "detail_file",
            "error_file",
            "report_file",
        )
        values = {
            name: str(self._resolve_path(getattr(record, name))) if getattr(record, name) else ""
            for name in path_fields
        }
        return replace(record, **values)

    def _artifacts(
        self,
        paths: dict[str, Path],
        *,
        hash_files: bool = False,
    ) -> tuple[HistoryArtifact, ...]:
        artifacts: list[HistoryArtifact] = []
        for artifact_type, value in paths.items():
            path = Path(value)
            exists = path.is_file()
            artifacts.append(
                HistoryArtifact(
                    artifact_type=artifact_type,
                    path=self._store_path(path),
                    file_size=path.stat().st_size if exists else None,
                    sha256=_sha256(path) if exists and hash_files else "",
                    created_at=_now(),
                    exists_status=(
                        IntegrityStatus.OK.value if exists else IntegrityStatus.MISSING.value
                    ),
                )
            )
        return tuple(artifacts)

    def _record_artifact_paths(self, record: TaskHistoryRecord) -> dict[str, Path]:
        return {
            name: self._resolve_path(value)
            for name, value in (
                ("task", record.task_file),
                ("summary", record.summary_file),
                ("result", record.detail_file),
                ("manifest", record.manifest_file),
                ("errors", record.error_file),
                ("report", record.report_file),
            )
            if value
        }


def _iso(value: datetime | None) -> str:
    return value.isoformat() if value is not None else ""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _duration_between(started_at: str, finished_at: str) -> float:
    if not started_at or not finished_at:
        return 0.0
    try:
        elapsed = _parse_datetime(finished_at) - _parse_datetime(started_at)
        return max(0.0, elapsed.total_seconds())
    except ValueError:
        return 0.0


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["HistoryService", "UnsafeHistoryDeleteError"]
