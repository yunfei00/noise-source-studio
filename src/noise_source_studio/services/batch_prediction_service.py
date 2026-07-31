"""Batch queue construction, retry state and durable result export."""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from noise_source_studio.domain.batch import (
    RUNNING_BATCH_STATUSES,
    BatchExportResult,
    BatchFileItem,
    BatchItemStatus,
    BatchPredictionTask,
    BatchScanSummary,
    BatchStatus,
)
from noise_source_studio.domain.models import LoadedModel
from noise_source_studio.services.result_adapter import (
    normalize_prediction_result,
)

SUPPORTED_EXTENSIONS = {".csv"}
LOGGER = logging.getLogger("noise_source_studio.batch_prediction_service")
PREDICTION_COLUMNS = (
    "task_id",
    "sequence",
    "file_name",
    "file_path",
    "status",
    "model_name",
    "model_version",
    "runtime_version",
    "device",
    "decision_mode",
    "labels",
    "display_probabilities",
    "multilabel_probabilities",
    "label_marginal_probabilities",
    "combination_labels",
    "combination_probabilities",
    "decoded_label_vector",
    "predicted_combination",
    "predicted_sources",
    "thresholds",
    "thresholds_applicable",
    "input_shape",
    "elapsed_ms",
    "started_at",
    "finished_at",
    "retry_count",
    "error_type",
    "error_message",
)
ERROR_COLUMNS = (
    "task_id",
    "sequence",
    "file_name",
    "file_path",
    "status",
    "error_type",
    "error_message",
    "retry_count",
    "started_at",
    "finished_at",
)
JSON_CSV_FIELDS = {
    "labels",
    "display_probabilities",
    "multilabel_probabilities",
    "label_marginal_probabilities",
    "combination_labels",
    "combination_probabilities",
    "decoded_label_vector",
    "predicted_sources",
    "thresholds",
    "input_shape",
}


class BatchPredictionService:
    """Own the in-memory queue; the worker owns sequential execution."""

    def __init__(self, output_directory: Path) -> None:
        self.output_directory = Path(output_directory)

    def create_task(self, name: str | None = None) -> BatchPredictionTask:
        """Create an empty queue without loading or selecting a model."""
        created = datetime.now(UTC)
        return BatchPredictionTask(
            name=name or f"批量任务 {created.astimezone().strftime('%Y-%m-%d %H:%M:%S')}",
            output_directory=self.output_directory,
            created_at=created,
        )

    def add_paths(
        self,
        task: BatchPredictionTask,
        paths: Iterable[Path],
        *,
        recursive: bool,
    ) -> BatchScanSummary:
        """Perform lightweight path checks and add naturally sorted CSV files."""
        self._ensure_editable(task)
        candidates: list[Path] = []
        source_directories: set[str] = set()
        invalid_count = 0
        scanned_count = 0
        for supplied in paths:
            path = Path(supplied).expanduser()
            if path.is_dir():
                resolved_directory = path.resolve()
                source_directories.add(str(resolved_directory))
                iterator = (
                    resolved_directory.rglob("*") if recursive else resolved_directory.glob("*")
                )
                for child in iterator:
                    if child.is_file():
                        scanned_count += 1
                        if child.suffix.lower() in SUPPORTED_EXTENSIONS:
                            candidates.append(child)
                        else:
                            invalid_count += 1
                continue
            scanned_count += 1
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                candidates.append(path)
            else:
                invalid_count += 1

        known = {self.normalized_path(item.file_path) for item in task.items}
        duplicate_count = 0
        added: list[Path] = []
        for candidate in sorted(candidates, key=self._natural_path_key):
            try:
                resolved = candidate.resolve(strict=True)
                with resolved.open("rb") as handle:
                    handle.read(1)
            except OSError:
                invalid_count += 1
                continue
            normalized = self.normalized_path(resolved)
            if normalized in known:
                duplicate_count += 1
                continue
            known.add(normalized)
            added.append(resolved)

        next_sequence = len(task.items) + 1
        task.items.extend(
            BatchFileItem(sequence=next_sequence + offset, file_path=path)
            for offset, path in enumerate(added)
        )
        task.recursive = recursive
        for directory in sorted(source_directories, key=str.casefold):
            if directory not in task.source_directories:
                task.source_directories.append(directory)
        task.refresh_counts()
        LOGGER.info(
            "Batch files scanned | task_id=%s | scanned=%d | added=%d | "
            "duplicates=%d | invalid=%d | recursive=%s",
            task.task_id,
            scanned_count,
            len(added),
            duplicate_count,
            invalid_count,
            recursive,
        )
        return BatchScanSummary(
            added_count=len(added),
            duplicate_count=duplicate_count,
            invalid_count=invalid_count,
            scanned_count=scanned_count,
            source_directories=tuple(sorted(source_directories)),
        )

    def remove_items(self, task: BatchPredictionTask, item_ids: Iterable[str]) -> None:
        """Remove selected items while the batch is editable."""
        self._ensure_editable(task)
        selected = set(item_ids)
        task.items = [item for item in task.items if item.item_id not in selected]
        self._renumber(task)

    def clear_items(self, task: BatchPredictionTask) -> None:
        """Clear an idle queue."""
        self._ensure_editable(task)
        task.items.clear()
        task.status = BatchStatus.CREATED
        task.started_at = None
        task.finished_at = None
        task.model_name = ""
        task.model_version = ""
        task.package_path = ""
        task.runtime_version = ""
        task.device = ""
        task.exported_files.clear()
        task.refresh_counts()

    def deduplicate(self, task: BatchPredictionTask) -> int:
        """Remove duplicates using Windows-safe normalized absolute paths."""
        self._ensure_editable(task)
        seen: set[str] = set()
        unique: list[BatchFileItem] = []
        removed = 0
        for item in task.items:
            normalized = self.normalized_path(item.file_path)
            if normalized in seen:
                removed += 1
            else:
                seen.add(normalized)
                unique.append(item)
        task.items = unique
        self._renumber(task)
        return removed

    def lock_model(self, task: BatchPredictionTask, model: LoadedModel) -> None:
        """Lock the exact active model/session metadata before execution."""
        task.model_name = model.record.model_name
        task.model_version = model.record.model_version
        task.package_path = str(model.record.package_path)
        task.runtime_version = model.runtime_version
        task.device = model.device

    def prepare_remaining(self, task: BatchPredictionTask) -> None:
        """Make uniformly stopped items pending for a deliberate rerun."""
        if task.status in RUNNING_BATCH_STATUSES:
            raise RuntimeError("批量任务运行时不能重新准备队列。")
        for item in task.items:
            if item.status == BatchItemStatus.STOPPED:
                item.status = BatchItemStatus.PENDING
                item.status_message = "等待重新执行"
        task.status = BatchStatus.CREATED
        task.started_at = None
        task.finished_at = None
        task.refresh_counts()

    def retry_failed(
        self,
        task: BatchPredictionTask,
        item_ids: Iterable[str] | None = None,
    ) -> int:
        """Reset selected or all failed items while preserving retry counts."""
        if task.status in RUNNING_BATCH_STATUSES:
            raise RuntimeError("批量任务运行时不能重试。")
        selected = set(item_ids) if item_ids is not None else None
        retried = 0
        for item in task.items:
            if item.status != BatchItemStatus.FAILED:
                continue
            if selected is not None and item.item_id not in selected:
                continue
            item.reset_for_retry()
            retried += 1
        if retried:
            task.status = BatchStatus.CREATED
            task.finished_at = None
            task.refresh_counts()
        return retried

    def export_results(
        self,
        task: BatchPredictionTask,
        *,
        destination: Path | None = None,
    ) -> BatchExportResult:
        """Write summary and UTF-8-BOM CSV files without mutating results."""
        root = Path(destination) if destination is not None else self.output_directory
        root.mkdir(parents=True, exist_ok=True)
        batch_name = (
            f"batch_{task.created_at.astimezone().strftime('%Y%m%d-%H%M%S')}_{task.task_id[:8]}"
        )
        output_directory = self._unique_directory(root / batch_name)
        output_directory.mkdir(parents=True)
        summary_path = output_directory / "summary.json"
        predictions_path = output_directory / "predictions.csv"
        errors_path = output_directory / "errors.csv"

        self._write_predictions(task, predictions_path)
        self._write_errors(task, errors_path)
        summary = self._summary_payload(
            task,
            summary_path,
            predictions_path,
            errors_path,
        )
        self._write_json_atomic(summary_path, summary)

        exported = BatchExportResult(
            output_directory=output_directory,
            summary_path=summary_path,
            predictions_path=predictions_path,
            errors_path=errors_path,
        )
        if destination is None:
            task.output_directory = output_directory
            task.exported_files = {
                "summary": str(summary_path),
                "predictions": str(predictions_path),
                "errors": str(errors_path),
            }
        LOGGER.info(
            "Batch results exported | task_id=%s | output=%s | status=%s",
            task.task_id,
            output_directory,
            task.status.value,
        )
        return exported

    @staticmethod
    def normalized_path(path: Path) -> str:
        """Normalize absolute paths with Windows case-insensitive semantics."""
        return os.path.normcase(os.path.normpath(str(Path(path).resolve())))

    @staticmethod
    def result_payload(result: Any) -> dict[str, Any]:
        """Expose the same adapter used by the single-file page."""
        return normalize_prediction_result(result)

    def _write_predictions(self, task: BatchPredictionTask, path: Path) -> None:
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
            writer.writeheader()
            for item in task.items:
                writer.writerow(self._prediction_row(task, item))
        os.replace(temporary, path)

    def _write_errors(self, task: BatchPredictionTask, path: Path) -> None:
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ERROR_COLUMNS)
            writer.writeheader()
            for item in task.items:
                if item.status != BatchItemStatus.FAILED:
                    continue
                prediction = self._prediction_row(task, item)
                writer.writerow({key: prediction[key] for key in ERROR_COLUMNS})
        os.replace(temporary, path)

    def _prediction_row(
        self,
        task: BatchPredictionTask,
        item: BatchFileItem,
    ) -> dict[str, Any]:
        payload = item.result or {}
        row: dict[str, Any] = {
            "task_id": task.task_id,
            "sequence": item.sequence,
            "file_name": item.file_name,
            "file_path": str(item.file_path),
            "status": item.status.value,
            "model_name": task.model_name,
            "model_version": task.model_version,
            "runtime_version": task.runtime_version,
            "device": task.device,
            "decision_mode": item.decision_mode,
            "labels": item.labels,
            "display_probabilities": item.display_probabilities,
            "multilabel_probabilities": payload.get("multilabel_probabilities", []),
            "label_marginal_probabilities": payload.get(
                "label_marginal_probabilities",
                [],
            ),
            "combination_labels": payload.get("combination_labels", []),
            "combination_probabilities": payload.get("combination_probabilities"),
            "decoded_label_vector": item.decoded_label_vector,
            "predicted_combination": item.predicted_combination,
            "predicted_sources": item.predicted_sources,
            "thresholds": payload.get("thresholds", []),
            "thresholds_applicable": payload.get("thresholds_applicable"),
            "input_shape": payload.get("input_shape", []),
            "elapsed_ms": item.elapsed_ms,
            "started_at": self._datetime_text(item.started_at),
            "finished_at": self._datetime_text(item.finished_at),
            "retry_count": item.retry_count,
            "error_type": item.error_type,
            "error_message": item.error_message,
        }
        for key in JSON_CSV_FIELDS:
            row[key] = json.dumps(row[key], ensure_ascii=False, separators=(",", ":"))
        return row

    def _summary_payload(
        self,
        task: BatchPredictionTask,
        summary_path: Path,
        predictions_path: Path,
        errors_path: Path,
    ) -> dict[str, Any]:
        combination_distribution = Counter(
            item.predicted_combination
            for item in task.items
            if item.status == BatchItemStatus.SUCCESS and item.predicted_combination
        )
        label_counts = Counter(
            label
            for item in task.items
            if item.status == BatchItemStatus.SUCCESS
            for label in item.predicted_sources
        )
        return {
            "task": {
                "task_id": task.task_id,
                "name": task.name,
                "created_at": self._datetime_text(task.created_at),
                "started_at": self._datetime_text(task.started_at),
                "finished_at": self._datetime_text(task.finished_at),
                "status": task.status.value,
                "recursive": task.recursive,
                "duplicate_policy": task.duplicate_policy,
                "source_directories": task.source_directories,
            },
            "model": {
                "model_name": task.model_name,
                "model_version": task.model_version,
                "package_path": task.package_path,
                "runtime_version": task.runtime_version,
                "device": task.device,
            },
            "statistics": {
                "total_count": task.total_count,
                "pending_count": task.pending_count,
                "success_count": task.success_count,
                "failed_count": task.failed_count,
                "skipped_count": task.skipped_count,
                "stopped_count": task.stopped_count,
                "progress": task.progress,
                "elapsed_seconds": task.elapsed_seconds,
                "average_item_seconds": task.average_item_seconds,
                "status_distribution": task.status_distribution(),
                "prediction_combination_distribution": dict(combination_distribution),
                "predicted_label_counts": dict(label_counts),
            },
            "output_files": {
                "summary": str(summary_path),
                "predictions": str(predictions_path),
                "errors": str(errors_path),
            },
        }

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    def _unique_directory(candidate: Path) -> Path:
        if not candidate.exists():
            return candidate
        suffix = 2
        while True:
            alternative = candidate.with_name(f"{candidate.name}_{suffix}")
            if not alternative.exists():
                return alternative
            suffix += 1

    @staticmethod
    def _natural_path_key(path: Path) -> tuple[Any, ...]:
        parts = re.split(r"(\d+)", str(path.resolve()).casefold())
        return tuple((1, int(part)) if part.isdigit() else (0, part) for part in parts)

    @staticmethod
    def _datetime_text(value: datetime | None) -> str:
        return value.isoformat() if value is not None else ""

    @staticmethod
    def _ensure_editable(task: BatchPredictionTask) -> None:
        if task.status in RUNNING_BATCH_STATUSES:
            raise RuntimeError("批量任务运行期间不能修改文件队列。")

    @staticmethod
    def _renumber(task: BatchPredictionTask) -> None:
        for sequence, item in enumerate(task.items, start=1):
            item.sequence = sequence
        task.refresh_counts()


__all__ = [
    "BatchPredictionService",
    "ERROR_COLUMNS",
    "PREDICTION_COLUMNS",
    "SUPPORTED_EXTENSIONS",
]
