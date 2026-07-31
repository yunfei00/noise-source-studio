"""Domain models for manifest-driven model validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4


class ValidationStatus(StrEnum):
    """Lifecycle states for a validation task."""

    CREATED = "created"
    CHECKED = "checked"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"


class ValidationSampleStatus(StrEnum):
    """Lifecycle states for one validation sample."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    INFERENCE_FAILED = "inference_failed"
    SKIPPED = "skipped"
    STOPPED = "stopped"


RUNNING_VALIDATION_STATUSES = {
    ValidationStatus.RUNNING,
    ValidationStatus.PAUSED,
    ValidationStatus.STOPPING,
}


@dataclass(slots=True)
class ValidationSample:
    """One lightweight labelled sample; raw signal data is never retained."""

    sequence: int
    file_path: Path
    true_label_vector: list[int]
    true_combination: str
    metadata: dict[str, str] = field(default_factory=dict)
    predicted_label_vector: list[int] = field(default_factory=list)
    predicted_combination: str = ""
    predicted_sources: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    display_probabilities: list[float] = field(default_factory=list)
    multilabel_probabilities: list[float] = field(default_factory=list)
    label_marginal_probabilities: list[float] = field(default_factory=list)
    combination_labels: list[str] = field(default_factory=list)
    combination_probabilities: list[float] = field(default_factory=list)
    thresholds: list[float] = field(default_factory=list)
    thresholds_applicable: bool = False
    exact_match: bool | None = None
    false_positive_labels: list[str] = field(default_factory=list)
    false_negative_labels: list[str] = field(default_factory=list)
    true_source_count: int = 0
    predicted_source_count: int = 0
    confidence: float | None = None
    confidence_margin: float | None = None
    elapsed_ms: float | None = None
    status: ValidationSampleStatus = ValidationSampleStatus.PENDING
    error_type: str = ""
    error_message: str = ""
    input_shape: list[int] = field(default_factory=list)
    result: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.file_path = Path(self.file_path).resolve()
        self.true_source_count = sum(self.true_label_vector)


@dataclass(frozen=True, slots=True)
class ManifestIssue:
    """One actionable manifest validation issue."""

    severity: str
    code: str
    message: str
    row: int | None = None
    file_path: str = ""


@dataclass(slots=True)
class ManifestValidationReport:
    """Parsed manifest records and its complete validation diagnostics."""

    manifest_path: Path
    data_root: Path
    labels: list[str]
    samples: list[ValidationSample] = field(default_factory=list)
    issues: list[ManifestIssue] = field(default_factory=list)
    metadata_fields: list[str] = field(default_factory=list)
    combination_counts: dict[str, int] = field(default_factory=dict)
    label_positive_counts: dict[str, int] = field(default_factory=dict)
    missing_count: int = 0
    duplicate_count: int = 0
    invalid_label_count: int = 0
    source_row_count: int = 0
    missing_policy: str = "stop"

    @property
    def valid_count(self) -> int:
        return sum(
            sample.status in {ValidationSampleStatus.PENDING, ValidationSampleStatus.SKIPPED}
            for sample in self.samples
        )

    @property
    def runnable_count(self) -> int:
        return sum(sample.status == ValidationSampleStatus.PENDING for sample in self.samples)

    @property
    def fatal_issues(self) -> list[ManifestIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def can_start(self) -> bool:
        return not self.fatal_issues and self.runnable_count > 0


@dataclass(slots=True)
class ValidationTask:
    """Complete in-memory validation task and aggregate result cache."""

    manifest_path: Path
    data_root: Path
    output_directory: Path
    task_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    model_name: str = ""
    model_version: str = ""
    runtime_version: str = ""
    checkpoint_sha256: str = ""
    device: str = ""
    labels: list[str] = field(default_factory=list)
    decision_mode: str = ""
    status: ValidationStatus = ValidationStatus.CREATED
    total_count: int = 0
    valid_count: int = 0
    success_count: int = 0
    inference_failed_count: int = 0
    skipped_count: int = 0
    stopped_count: int = 0
    progress: float = 0.0
    metadata_fields: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    samples: list[ValidationSample] = field(default_factory=list)
    manifest_report: ManifestValidationReport | None = None
    exported_files: dict[str, str] = field(default_factory=dict)

    def refresh_counts(self) -> None:
        counts = Counter(sample.status for sample in self.samples)
        self.total_count = len(self.samples)
        self.valid_count = self.total_count - counts[ValidationSampleStatus.SKIPPED]
        self.success_count = counts[ValidationSampleStatus.SUCCESS]
        self.inference_failed_count = counts[ValidationSampleStatus.INFERENCE_FAILED]
        self.skipped_count = counts[ValidationSampleStatus.SKIPPED]
        self.stopped_count = counts[ValidationSampleStatus.STOPPED]
        completed = (
            self.success_count
            + self.inference_failed_count
            + self.skipped_count
            + self.stopped_count
        )
        self.progress = completed / self.total_count * 100.0 if self.total_count else 0.0

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or datetime.now(UTC)
        return max(0.0, (end - self.started_at).total_seconds())


@dataclass(frozen=True, slots=True)
class ValidationExportResult:
    """All files generated for one validation result directory."""

    output_directory: Path
    task_path: Path
    summary_path: Path
    sample_results_path: Path
    label_metrics_path: Path
    combination_metrics_path: Path
    confusion_matrix_path: Path
    group_metrics_path: Path
    errors_path: Path
    report_path: Path


__all__ = [
    "ManifestIssue",
    "ManifestValidationReport",
    "RUNNING_VALIDATION_STATUSES",
    "ValidationExportResult",
    "ValidationSample",
    "ValidationSampleStatus",
    "ValidationStatus",
    "ValidationTask",
]
