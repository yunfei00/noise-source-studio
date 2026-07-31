"""Batch prediction task models and explicit lifecycle states."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4


class BatchItemStatus(StrEnum):
    """Lifecycle states for one file in a batch."""

    PENDING = "pending"
    VALIDATING = "validating"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    STOPPED = "stopped"


class BatchStatus(StrEnum):
    """Lifecycle states for the complete batch."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"


TERMINAL_ITEM_STATUSES = {
    BatchItemStatus.SUCCESS,
    BatchItemStatus.FAILED,
    BatchItemStatus.SKIPPED,
    BatchItemStatus.STOPPED,
}
RUNNING_BATCH_STATUSES = {
    BatchStatus.RUNNING,
    BatchStatus.PAUSED,
    BatchStatus.STOPPING,
}


@dataclass(slots=True)
class BatchFileItem:
    """One lightweight file task and its normalized prediction result."""

    sequence: int
    file_path: Path
    item_id: str = field(default_factory=lambda: uuid4().hex)
    file_name: str = ""
    file_size: int = 0
    modified_at: str = ""
    status: BatchItemStatus = BatchItemStatus.PENDING
    status_message: str = "等待执行"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    elapsed_ms: float | None = None
    labels: list[str] = field(default_factory=list)
    decision_mode: str = ""
    display_probabilities: list[float] = field(default_factory=list)
    predicted_combination: str = ""
    predicted_sources: list[str] = field(default_factory=list)
    decoded_label_vector: list[int] = field(default_factory=list)
    error_type: str = ""
    error_message: str = ""
    retry_count: int = 0
    result: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.file_path = Path(self.file_path).resolve()
        if not self.file_name:
            self.file_name = self.file_path.name
        if not self.file_size or not self.modified_at:
            try:
                stat = self.file_path.stat()
            except OSError:
                return
            self.file_size = stat.st_size
            self.modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()

    def reset_for_retry(self) -> None:
        """Clear stale outcome data while preserving retry history."""
        self.retry_count += 1
        self.status = BatchItemStatus.PENDING
        self.status_message = "等待重试"
        self.started_at = None
        self.finished_at = None
        self.elapsed_ms = None
        self.labels.clear()
        self.decision_mode = ""
        self.display_probabilities.clear()
        self.predicted_combination = ""
        self.predicted_sources.clear()
        self.decoded_label_vector.clear()
        self.error_type = ""
        self.error_message = ""
        self.result = None


@dataclass(slots=True)
class BatchPredictionTask:
    """One ordered, single-session batch prediction run."""

    name: str
    output_directory: Path
    task_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    model_name: str = ""
    model_version: str = ""
    package_path: str = ""
    runtime_version: str = ""
    device: str = ""
    status: BatchStatus = BatchStatus.CREATED
    items: list[BatchFileItem] = field(default_factory=list)
    total_count: int = 0
    pending_count: int = 0
    running_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    stopped_count: int = 0
    progress: float = 0.0
    source_directories: list[str] = field(default_factory=list)
    recursive: bool = False
    duplicate_policy: str = "normalized_absolute_path"
    exported_files: dict[str, str] = field(default_factory=dict)

    def refresh_counts(self) -> None:
        """Recalculate all counters from authoritative item states."""
        counts = Counter(item.status for item in self.items)
        self.total_count = len(self.items)
        self.pending_count = counts[BatchItemStatus.PENDING]
        self.running_count = counts[BatchItemStatus.VALIDATING] + counts[BatchItemStatus.RUNNING]
        self.success_count = counts[BatchItemStatus.SUCCESS]
        self.failed_count = counts[BatchItemStatus.FAILED]
        self.skipped_count = counts[BatchItemStatus.SKIPPED]
        self.stopped_count = counts[BatchItemStatus.STOPPED]
        finished = sum(counts[status] for status in TERMINAL_ITEM_STATUSES)
        self.progress = finished / self.total_count * 100.0 if self.total_count else 0.0

    @property
    def completed_count(self) -> int:
        return self.success_count + self.failed_count + self.skipped_count + self.stopped_count

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or datetime.now(UTC)
        return max(0.0, (end - self.started_at).total_seconds())

    @property
    def average_item_seconds(self) -> float | None:
        values = [
            item.elapsed_ms / 1000.0
            for item in self.items
            if item.elapsed_ms is not None
            and item.status in {BatchItemStatus.SUCCESS, BatchItemStatus.FAILED}
        ]
        return sum(values) / len(values) if values else None

    @property
    def estimated_remaining_seconds(self) -> float | None:
        attempts = [
            item
            for item in self.items
            if item.elapsed_ms is not None
            and item.status in {BatchItemStatus.SUCCESS, BatchItemStatus.FAILED}
        ]
        average = self.average_item_seconds
        if len(attempts) < 2 or average is None:
            return None
        return max(0.0, average * (self.pending_count + self.running_count))

    def status_distribution(self) -> dict[str, int]:
        counts = Counter(item.status.value for item in self.items)
        return {status.value: counts[status.value] for status in BatchItemStatus}


@dataclass(frozen=True, slots=True)
class BatchScanSummary:
    """Outcome of a lightweight file/folder scan."""

    added_count: int
    duplicate_count: int
    invalid_count: int
    scanned_count: int
    source_directories: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BatchExportResult:
    """Paths written by one automatic or manual batch export."""

    output_directory: Path
    summary_path: Path
    predictions_path: Path
    errors_path: Path


__all__ = [
    "BatchExportResult",
    "BatchFileItem",
    "BatchItemStatus",
    "BatchPredictionTask",
    "BatchScanSummary",
    "BatchStatus",
    "RUNNING_BATCH_STATUSES",
    "TERMINAL_ITEM_STATUSES",
]
