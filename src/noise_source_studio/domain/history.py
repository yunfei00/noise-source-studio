"""Unified task-history domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TaskType(StrEnum):
    """Task kinds stored in the shared history index."""

    SINGLE = "single_prediction"
    BATCH = "batch_prediction"
    VALIDATION = "model_validation"


class HistoryStatus(StrEnum):
    """Normalized lifecycle states shared by all task kinds."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"
    STOPPED = "stopped"
    INTERRUPTED = "interrupted"


class IntegrityStatus(StrEnum):
    """Availability state of a task's filesystem artifacts."""

    UNKNOWN = "unknown"
    OK = "ok"
    MISSING = "missing"
    PARTIAL = "partial"
    CORRUPT = "corrupt"
    EXTERNAL = "external"


@dataclass(frozen=True, slots=True)
class HistoryArtifact:
    """One file belonging to a history task."""

    artifact_type: str
    path: str
    artifact_id: str = ""
    file_size: int | None = None
    sha256: str = ""
    created_at: str = ""
    exists_status: str = IntegrityStatus.UNKNOWN.value


@dataclass(frozen=True, slots=True)
class TaskHistoryRecord:
    """Complete searchable metadata for a historical task."""

    task_id: str
    task_type: TaskType
    task_name: str
    status: HistoryStatus
    created_at: str
    started_at: str = ""
    finished_at: str = ""
    duration_ms: float = 0.0
    model_name: str = ""
    model_version: str = ""
    model_identifier: str = ""
    model_package_sha256: str = ""
    runtime_version: str = ""
    application_version: str = ""
    device: str = ""
    decision_mode: str = ""
    total_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    stopped_count: int = 0
    primary_summary: str = ""
    source_description: str = ""
    source_path: str = ""
    result_directory: str = ""
    task_file: str = ""
    summary_file: str = ""
    detail_file: str = ""
    error_file: str = ""
    report_file: str = ""
    manifest_file: str = ""
    integrity_status: IntegrityStatus = IntegrityStatus.UNKNOWN
    error_type: str = ""
    error_message: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""
    created_by_application: bool = True
    imported_from_legacy: bool = False
    last_opened_at: str = ""


@dataclass(frozen=True, slots=True)
class HistoryQuery:
    """Database-side filters, ordering, and pagination."""

    task_type: TaskType | None = None
    status: HistoryStatus | None = None
    date_from: str = ""
    date_to: str = ""
    model_name: str = ""
    model_version: str = ""
    device: str = ""
    keyword: str = ""
    failed_only: bool = False
    missing_only: bool = False
    page: int = 1
    page_size: int = 50
    sort_by: str = "created_at"
    descending: bool = True


@dataclass(frozen=True, slots=True)
class HistoryPageResult:
    """One page returned from a history query."""

    records: tuple[TaskHistoryRecord, ...]
    total: int
    page: int
    page_size: int

    @property
    def page_count(self) -> int:
        return max(1, (self.total + self.page_size - 1) // self.page_size)


@dataclass(frozen=True, slots=True)
class HistoryOverview:
    """Small dashboard summary computed directly in SQLite."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    interrupted: int = 0
    last_seven_days: int = 0
    missing_artifacts: int = 0


@dataclass(frozen=True, slots=True)
class HistoryScanReport:
    """Outcome of scanning result folders without performing inference."""

    scanned_directories: int
    imported_tasks: int
    updated_tasks: int
    skipped_tasks: int
    corrupted_directories: tuple[str, ...] = field(default_factory=tuple)
    cancelled: bool = False


__all__ = [
    "HistoryArtifact",
    "HistoryOverview",
    "HistoryPageResult",
    "HistoryQuery",
    "HistoryScanReport",
    "HistoryStatus",
    "IntegrityStatus",
    "TaskHistoryRecord",
    "TaskType",
]
