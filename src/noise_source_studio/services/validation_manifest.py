"""Validation-manifest parsing and data-contract checks."""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path

from noise_source_studio.domain.validation import (
    ManifestIssue,
    ManifestValidationReport,
    ValidationSample,
    ValidationSampleStatus,
)


class ValidationManifestService:
    """Parse an explicit manifest against the active model label order."""

    RESERVED_FIELDS = {"file_path", "true_combination"}

    def inspect(
        self,
        manifest_path: Path,
        labels: list[str] | tuple[str, ...],
        *,
        data_root: Path | None = None,
        missing_policy: str = "stop",
    ) -> ManifestValidationReport:
        path = Path(manifest_path).expanduser().resolve()
        label_order = [str(label) for label in labels]
        root = Path(data_root).expanduser().resolve() if data_root else path.parent
        report = ManifestValidationReport(
            manifest_path=path,
            data_root=root,
            labels=label_order,
            missing_policy=missing_policy,
        )
        if missing_policy not in {"stop", "skip"}:
            report.issues.append(
                ManifestIssue(
                    "error",
                    "invalid_missing_policy",
                    "缺失文件策略必须是 stop 或 skip。",
                )
            )
            return report
        if not label_order or len(set(label_order)) != len(label_order):
            report.issues.append(
                ManifestIssue("error", "invalid_model_labels", "当前模型标签为空或存在重复。")
            )
            return report
        if not path.is_file():
            report.issues.append(
                ManifestIssue("error", "manifest_unreadable", f"验证清单不存在：{path}")
            )
            return report

        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = [str(value).strip() for value in (reader.fieldnames or [])]
                rows = list(reader)
        except (OSError, UnicodeError, csv.Error) as exc:
            report.issues.append(
                ManifestIssue("error", "manifest_unreadable", f"无法读取验证清单：{exc}")
            )
            return report

        if "file_path" not in fieldnames:
            report.issues.append(
                ManifestIssue("error", "missing_file_path", "验证清单缺少 file_path 列。")
            )
            return report
        has_combination = "true_combination" in fieldnames
        present_label_columns = [field for field in fieldnames if field in label_order]
        if not has_combination and set(present_label_columns) != set(label_order):
            report.issues.append(
                ManifestIssue(
                    "error",
                    "missing_truth",
                    "清单必须包含 true_combination，或包含全部模型标签列并保持模型标签顺序。",
                )
            )
            return report
        if present_label_columns and present_label_columns != label_order:
            report.issues.append(
                ManifestIssue(
                    "error",
                    "label_order_unknown",
                    "逐标签列顺序与当前激活模型 labels 不一致。",
                )
            )
            return report

        report.metadata_fields = [
            field
            for field in fieldnames
            if field not in self.RESERVED_FIELDS and field not in label_order
        ]
        report.source_row_count = len(rows)
        seen: dict[str, tuple[int, ...]] = {}
        combinations: Counter[str] = Counter()
        positives: Counter[str] = Counter()

        for sequence, row in enumerate(rows, start=1):
            sample = self._parse_row(
                sequence,
                row,
                path,
                root,
                label_order,
                has_combination,
                present_label_columns == label_order,
                report,
            )
            if sample is None:
                continue
            normalized = self._normalized_path(sample.file_path)
            vector = tuple(sample.true_label_vector)
            if normalized in seen:
                report.duplicate_count += 1
                if seen[normalized] != vector:
                    report.issues.append(
                        ManifestIssue(
                            "error",
                            "conflicting_duplicate",
                            "同一文件出现了相互矛盾的真实标签。",
                            sequence,
                            str(sample.file_path),
                        )
                    )
                else:
                    report.issues.append(
                        ManifestIssue(
                            "warning",
                            "duplicate_file",
                            "清单包含重复文件记录。",
                            sequence,
                            str(sample.file_path),
                        )
                    )
            else:
                seen[normalized] = vector
            report.samples.append(sample)
            combinations[sample.true_combination] += 1
            for label, value in zip(label_order, sample.true_label_vector, strict=True):
                positives[label] += value

        report.combination_counts = dict(combinations)
        report.label_positive_counts = {label: positives[label] for label in label_order}
        if report.source_row_count and not report.samples:
            report.issues.append(
                ManifestIssue("error", "all_invalid", "验证清单中的所有文件均无效。")
            )
        elif not report.source_row_count:
            report.issues.append(ManifestIssue("error", "empty_manifest", "验证清单没有数据行。"))
        if report.runnable_count == 0:
            report.issues.append(
                ManifestIssue("error", "no_runnable_samples", "没有可执行验证的有效样本。")
            )
        return report

    def _parse_row(
        self,
        sequence: int,
        row: dict[str, str | None],
        manifest_path: Path,
        data_root: Path,
        labels: list[str],
        has_combination: bool,
        has_label_columns: bool,
        report: ManifestValidationReport,
    ) -> ValidationSample | None:
        raw_path = (row.get("file_path") or "").strip()
        if not raw_path:
            report.issues.append(
                ManifestIssue("error", "empty_file_path", "file_path 不能为空。", sequence)
            )
            return None
        supplied_path = Path(raw_path).expanduser()
        file_path = supplied_path if supplied_path.is_absolute() else data_root / supplied_path
        file_path = file_path.resolve()

        combination = (row.get("true_combination") or "").strip() if has_combination else ""
        vector_from_combination = self._parse_combination(combination, len(labels))
        vector_from_columns = self._parse_label_columns(row, labels) if has_label_columns else None
        if has_combination and vector_from_combination is None:
            report.invalid_label_count += 1
            report.issues.append(
                ManifestIssue(
                    "error",
                    "invalid_combination",
                    f"true_combination 必须是长度为 {len(labels)} 的非全零二进制字符串。",
                    sequence,
                    str(file_path),
                )
            )
            return None
        if has_label_columns and vector_from_columns is None:
            report.invalid_label_count += 1
            report.issues.append(
                ManifestIssue(
                    "error",
                    "invalid_label_columns",
                    "逐标签列只能包含 0 或 1。",
                    sequence,
                    str(file_path),
                )
            )
            return None
        if (
            vector_from_combination is not None
            and vector_from_columns is not None
            and vector_from_combination != vector_from_columns
        ):
            report.invalid_label_count += 1
            report.issues.append(
                ManifestIssue(
                    "error",
                    "truth_conflict",
                    "true_combination 与逐标签列内容不一致。",
                    sequence,
                    str(file_path),
                )
            )
            return None
        vector = vector_from_combination or vector_from_columns
        if vector is None:
            report.invalid_label_count += 1
            report.issues.append(
                ManifestIssue("error", "missing_truth", "该行缺少真实标签。", sequence)
            )
            return None
        true_combination = "".join(str(value) for value in vector)
        metadata = {
            field: (row.get(field) or "").strip()
            for field in report.metadata_fields
            if (row.get(field) or "").strip()
        }
        status = ValidationSampleStatus.PENDING
        if not file_path.is_file():
            report.missing_count += 1
            severity = "error" if report.missing_policy == "stop" else "warning"
            report.issues.append(
                ManifestIssue(
                    severity,
                    "missing_file",
                    "样本文件不存在。",
                    sequence,
                    str(file_path),
                )
            )
            status = ValidationSampleStatus.SKIPPED
        return ValidationSample(
            sequence=sequence,
            file_path=file_path,
            true_label_vector=vector,
            true_combination=true_combination,
            metadata=metadata,
            status=status,
            error_type="MissingFile" if status == ValidationSampleStatus.SKIPPED else "",
            error_message="样本文件不存在。" if status == ValidationSampleStatus.SKIPPED else "",
        )

    @staticmethod
    def _parse_combination(value: str, label_count: int) -> list[int] | None:
        if (
            len(value) != label_count
            or not value
            or any(character not in {"0", "1"} for character in value)
            or "1" not in value
        ):
            return None
        return [int(character) for character in value]

    @staticmethod
    def _parse_label_columns(
        row: dict[str, str | None],
        labels: list[str],
    ) -> list[int] | None:
        values = [(row.get(label) or "").strip() for label in labels]
        if any(value not in {"0", "1"} for value in values):
            return None
        if "1" not in values:
            return None
        return [int(value) for value in values]

    @staticmethod
    def _normalized_path(path: Path) -> str:
        return os.path.normcase(os.path.normpath(str(path.resolve())))


__all__ = ["ValidationManifestService"]
