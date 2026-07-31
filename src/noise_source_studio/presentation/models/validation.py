"""Virtual table models for validation manifests, metrics and samples."""

from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt

from noise_source_studio.domain.confidence import is_low_confidence
from noise_source_studio.domain.validation import (
    ManifestValidationReport,
    ValidationSample,
    ValidationSampleStatus,
    ValidationTask,
)


class ValidationSampleTableModel(QAbstractTableModel):
    """Lightweight view over authoritative validation samples."""

    SampleRole = Qt.ItemDataRole.UserRole + 1
    SortRole = Qt.ItemDataRole.UserRole + 2
    HEADERS = (
        "序号",
        "文件名",
        "真实组合",
        "预测组合",
        "是否正确",
        "错误类型",
        "置信度",
        "第一候选",
        "第二候选",
        "概率差",
        "推理耗时",
        "元数据摘要",
    )

    def __init__(self) -> None:
        super().__init__()
        self.task: ValidationTask | None = None
        self.samples: list[ValidationSample] = []
        self._row_by_sequence: dict[int, int] = {}

    def set_task(self, task: ValidationTask | None) -> None:
        self.beginResetModel()
        self.task = task
        self.samples = list(task.samples) if task else []
        self._row_by_sequence = {sample.sequence: row for row, sample in enumerate(self.samples)}
        self.endResetModel()

    def refresh(self) -> None:
        self.set_task(self.task)

    def refresh_sample(self, sample: ValidationSample) -> None:
        row = self._row_by_sequence.get(sample.sequence)
        if row is None:
            return
        self.dataChanged.emit(
            self.index(row, 0),
            self.index(row, self.columnCount() - 1),
        )

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(self.samples)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not 0 <= index.row() < len(self.samples):
            return None
        sample = self.samples[index.row()]
        if role == self.SampleRole:
            return sample
        if role == self.SortRole:
            return self._sort_values(sample)[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return self._display_values(sample)[index.column()]
        if role == Qt.ItemDataRole.ToolTipRole:
            return str(sample.file_path) if index.column() == 1 else None
        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() != 1:
            return int(Qt.AlignmentFlag.AlignCenter)
        if (
            role == Qt.ItemDataRole.ForegroundRole
            and sample.status == ValidationSampleStatus.INFERENCE_FAILED
        ):
            return Qt.GlobalColor.darkRed
        if role == Qt.ItemDataRole.BackgroundRole and is_low_confidence(sample.result or {}):
            return Qt.GlobalColor.lightGray
        return None

    def sample_at(self, row: int) -> ValidationSample | None:
        return self.samples[row] if 0 <= row < len(self.samples) else None

    @staticmethod
    def _candidates(sample: ValidationSample) -> list[tuple[str, float]]:
        return sorted(
            zip(
                sample.combination_labels,
                sample.combination_probabilities,
                strict=False,
            ),
            key=lambda value: value[1],
            reverse=True,
        )

    @classmethod
    def _display_values(cls, sample: ValidationSample) -> tuple[Any, ...]:
        candidates = cls._candidates(sample)
        first = candidates[0][0] if candidates else "—"
        second = candidates[1][0] if len(candidates) > 1 else "—"
        return (
            sample.sequence,
            sample.file_path.name,
            sample.true_combination,
            sample.predicted_combination or "—",
            "正确" if sample.exact_match else ("错误" if sample.exact_match is False else "—"),
            sample.error_type or "—",
            f"{sample.confidence * 100:.2f}%" if sample.confidence is not None else "—",
            first,
            second,
            (
                f"{sample.confidence_margin * 100:.2f}%"
                if sample.confidence_margin is not None
                else "—"
            ),
            f"{sample.elapsed_ms:.1f} ms" if sample.elapsed_ms is not None else "—",
            ", ".join(f"{key}={value}" for key, value in sample.metadata.items()) or "—",
        )

    @classmethod
    def _sort_values(cls, sample: ValidationSample) -> tuple[Any, ...]:
        candidates = cls._candidates(sample)
        return (
            sample.sequence,
            sample.file_path.name.casefold(),
            sample.true_combination,
            sample.predicted_combination,
            -1 if sample.exact_match is None else int(sample.exact_match),
            sample.error_type.casefold(),
            sample.confidence if sample.confidence is not None else -1.0,
            candidates[0][1] if candidates else -1.0,
            candidates[1][1] if len(candidates) > 1 else -1.0,
            sample.confidence_margin if sample.confidence_margin is not None else -1.0,
            sample.elapsed_ms if sample.elapsed_ms is not None else -1.0,
            json.dumps(sample.metadata, ensure_ascii=False, sort_keys=True).casefold(),
        )


class ValidationSampleFilterProxyModel(QSortFilterProxyModel):
    """Composable filters for error-analysis workflows."""

    def __init__(self) -> None:
        super().__init__()
        self.keyword = ""
        self.outcome = "all"
        self.true_combination = ""
        self.predicted_combination = ""
        self.label = ""
        self.label_relation = ""
        self.metadata_field = ""
        self.metadata_value = ""
        self.special = ""
        self.confidence_min = 0.0
        self.confidence_max = 1.0
        self.error_type = ""
        self.setSortRole(ValidationSampleTableModel.SortRole)
        self.setDynamicSortFilter(True)

    def set_filters(
        self,
        *,
        keyword: str | None = None,
        outcome: str | None = None,
        true_combination: str | None = None,
        predicted_combination: str | None = None,
        label: str | None = None,
        label_relation: str | None = None,
        metadata_field: str | None = None,
        metadata_value: str | None = None,
        confidence_min: float | None = None,
        confidence_max: float | None = None,
        error_type: str | None = None,
    ) -> None:
        self.beginFilterChange()
        if keyword is not None:
            self.keyword = keyword.strip().casefold()
        if outcome is not None:
            self.outcome = outcome
        if true_combination is not None:
            self.true_combination = true_combination
        if predicted_combination is not None:
            self.predicted_combination = predicted_combination
        if label is not None:
            self.label = label
        if label_relation is not None:
            self.label_relation = label_relation
        if metadata_field is not None:
            self.metadata_field = metadata_field
        if metadata_value is not None:
            self.metadata_value = metadata_value
        if confidence_min is not None:
            self.confidence_min = confidence_min
        if confidence_max is not None:
            self.confidence_max = confidence_max
        if error_type is not None:
            self.error_type = error_type
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    def filterAcceptsRow(  # noqa: N802
        self, source_row: int, source_parent: QModelIndex
    ) -> bool:
        model = self.sourceModel()
        if not isinstance(model, ValidationSampleTableModel):
            return True
        sample = model.sample_at(source_row)
        if sample is None:
            return False
        if self.keyword and self.keyword not in f"{sample.file_path}".casefold():
            return False
        if not self._outcome_matches(sample):
            return False
        if self.true_combination and sample.true_combination != self.true_combination:
            return False
        if (
            self.predicted_combination
            and sample.predicted_combination != self.predicted_combination
        ):
            return False
        if self.label and not self._label_matches(sample):
            return False
        if (
            sample.confidence is not None
            and not self.confidence_min <= sample.confidence <= self.confidence_max
        ):
            return False
        if self.error_type and sample.error_type != self.error_type:
            return False
        if self.metadata_field:
            value = sample.metadata.get(self.metadata_field, "")
            if self.metadata_value and value != self.metadata_value:
                return False
            if not self.metadata_value and not value:
                return False
        if self.special == "double_to_triple":
            return sample.true_source_count == 2 and sample.predicted_source_count == 3
        if self.special == "triple_to_double":
            return sample.true_source_count == 3 and sample.predicted_source_count == 2
        if self.special == "single_to_multi":
            return sample.true_source_count == 1 and sample.predicted_source_count > 1
        return True

    def _outcome_matches(self, sample: ValidationSample) -> bool:
        checks = {
            "all": True,
            "correct": sample.exact_match is True,
            "wrong": sample.exact_match is False,
            "failed": sample.status == ValidationSampleStatus.INFERENCE_FAILED,
            "low_confidence": is_low_confidence(sample.result or {}),
            "overprediction": sample.predicted_source_count > sample.true_source_count,
            "underprediction": sample.predicted_source_count < sample.true_source_count,
            "source_count_error": sample.predicted_source_count != sample.true_source_count,
        }
        return checks.get(self.outcome, True)

    def _label_matches(self, sample: ValidationSample) -> bool:
        try:
            index = sample.labels.index(self.label)
        except ValueError:
            return False
        true = bool(sample.true_label_vector[index])
        predicted = bool(sample.predicted_label_vector[index])
        checks = {
            "all": true or predicted,
            "false_positive": not true and predicted,
            "false_negative": true and not predicted,
            "correct": true == predicted,
        }
        return checks.get(self.label_relation, True)

    def filtered_samples(self) -> list[ValidationSample]:
        model = self.sourceModel()
        if not isinstance(model, ValidationSampleTableModel):
            return []
        return [
            sample
            for row in range(self.rowCount())
            if (sample := model.sample_at(self.mapToSource(self.index(row, 0)).row())) is not None
        ]


class DictTableModel(QAbstractTableModel):
    """Sortable generic table for already aggregated dictionaries."""

    SortRole = Qt.ItemDataRole.UserRole + 2

    def __init__(self, columns: list[tuple[str, str]]) -> None:
        super().__init__()
        self.columns = columns
        self.rows: list[dict[str, Any]] = []

    def set_rows(self, rows: list[dict[str, Any]]) -> None:
        self.beginResetModel()
        self.rows = list(rows)
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(self.columns)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.columns[section][1]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not 0 <= index.row() < len(self.rows):
            return None
        value = self.rows[index.row()].get(self.columns[index.column()][0])
        if role == self.SortRole:
            return -1.0 if value is None else value
        if role == Qt.ItemDataRole.DisplayRole:
            if value is None:
                return "—"
            field_name = self.columns[index.column()][0]
            if field_name.endswith("_ms") and isinstance(value, (int, float)):
                return f"{value:.1f} ms"
            if isinstance(value, float):
                return f"{value * 100:.2f}%"
            return str(value)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignCenter)
        return None

    def row_at(self, row: int) -> dict[str, Any] | None:
        return self.rows[row] if 0 <= row < len(self.rows) else None


class ManifestPreviewTableModel(QAbstractTableModel):
    """Small preview of parsed manifest records."""

    HEADERS = ("序号", "文件", "真实组合", "状态", "元数据")

    def __init__(self) -> None:
        super().__init__()
        self.report: ManifestValidationReport | None = None

    def set_report(self, report: ManifestValidationReport | None) -> None:
        self.beginResetModel()
        self.report = report
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802, B008
        if parent.isValid() or self.report is None:
            return 0
        return min(100, len(self.report.samples))

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or self.report is None:
            return None
        sample = self.report.samples[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return (
                sample.sequence,
                sample.file_path.name,
                sample.true_combination,
                sample.status.value,
                ", ".join(f"{key}={value}" for key, value in sample.metadata.items()),
            )[index.column()]
        if role == Qt.ItemDataRole.ToolTipRole and index.column() == 1:
            return str(sample.file_path)
        return None


class ConfusionMatrixTableModel(QAbstractTableModel):
    """Dynamic combination-level confusion matrix model."""

    CellRole = Qt.ItemDataRole.UserRole + 1

    def __init__(self) -> None:
        super().__init__()
        self.labels: list[str] = []
        self.counts: list[list[int]] = []
        self.percentages = False

    def set_matrix(self, matrix: dict[str, Any]) -> None:
        self.beginResetModel()
        self.labels = [str(value) for value in matrix.get("labels", [])]
        self.counts = [[int(value) for value in row] for row in matrix.get("counts", [])]
        self.endResetModel()

    def set_percentages(self, enabled: bool) -> None:
        self.percentages = enabled
        if self.rowCount() and self.columnCount():
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(self.rowCount() - 1, self.columnCount() - 1),
            )

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(self.labels)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802, B008
        return 0 if parent.isValid() else len(self.labels)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            return self.labels[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        count = self.counts[index.row()][index.column()]
        if role == self.CellRole:
            return self.labels[index.row()], self.labels[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            if not self.percentages:
                return count
            total = sum(self.counts[index.row()])
            return f"{count / total * 100:.1f}%" if total else "—"
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignCenter)
        if role == Qt.ItemDataRole.BackgroundRole and count:
            return Qt.GlobalColor.lightGray
        return None


__all__ = [
    "ConfusionMatrixTableModel",
    "DictTableModel",
    "ManifestPreviewTableModel",
    "ValidationSampleFilterProxyModel",
    "ValidationSampleTableModel",
]
