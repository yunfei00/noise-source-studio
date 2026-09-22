"""Validation task creation, persistence, recovery and batch-result reuse."""

from __future__ import annotations

import csv
import html
import json
import os
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from noise_source_studio.domain.batch import BatchItemStatus
from noise_source_studio.domain.models import LoadedModel
from noise_source_studio.domain.validation import (
    ManifestValidationReport,
    ValidationExportResult,
    ValidationSample,
    ValidationSampleStatus,
    ValidationStatus,
    ValidationTask,
)
from noise_source_studio.services.batch_prediction_service import BatchPredictionService
from noise_source_studio.services.validation_manifest import ValidationManifestService
from noise_source_studio.services.validation_metrics import calculate_validation_metrics
from noise_source_studio.services.validation_result import apply_validation_result
from noise_source_studio.version import __version__

SAMPLE_COLUMNS = (
    "sequence",
    "file_path",
    "status",
    "true_combination",
    "predicted_combination",
    "true_label_vector",
    "predicted_label_vector",
    "predicted_sources",
    "display_probabilities",
    "multilabel_probabilities",
    "label_marginal_probabilities",
    "combination_labels",
    "combination_probabilities",
    "thresholds",
    "thresholds_applicable",
    "exact_match",
    "false_positive_labels",
    "false_negative_labels",
    "true_source_count",
    "predicted_source_count",
    "confidence",
    "confidence_margin",
    "elapsed_ms",
    "error_type",
    "error_message",
    "input_shape",
    "metadata",
)

LABEL_COLUMNS = (
    "label",
    "tp",
    "fp",
    "tn",
    "fn",
    "precision",
    "recall",
    "f1",
    "specificity",
    "false_positive_rate",
    "false_negative_rate",
    "support",
    "predicted_positive_count",
)

COMBINATION_COLUMNS = (
    "true_combination",
    "support",
    "exact_count",
    "exact_accuracy",
    "most_common_mistake",
    "most_common_mistake_count",
    "precision",
    "recall",
    "f1",
    "average_confidence",
    "average_confidence_margin",
)

GROUP_COLUMNS = (
    "field",
    "value",
    "sample_count",
    "exact_match",
    "micro_f1",
    "macro_f1",
    "overprediction_rate",
    "underprediction_rate",
    "inference_failed_count",
    "average_confidence",
    "average_elapsed_ms",
)

ERROR_COLUMNS = (
    "sequence",
    "file_path",
    "status",
    "true_combination",
    "error_type",
    "error_message",
    "metadata",
)


class ValidationService:
    """Own validation contracts while workers own long-running execution."""

    def __init__(
        self,
        output_directory: Path,
        *,
        manifest_service: ValidationManifestService | None = None,
    ) -> None:
        self.output_directory = Path(output_directory)
        self.manifest_service = manifest_service or ValidationManifestService()

    def inspect_manifest(
        self,
        manifest_path: Path,
        model: LoadedModel,
        *,
        data_root: Path | None = None,
        missing_policy: str = "stop",
    ) -> ManifestValidationReport:
        """Validate one manifest against the exact active model label order."""
        return self.manifest_service.inspect(
            manifest_path,
            model.labels,
            data_root=data_root,
            missing_policy=missing_policy,
        )

    def create_task(
        self,
        report: ManifestValidationReport,
        model: LoadedModel,
    ) -> ValidationTask:
        """Create an immutable-model validation task from a successful inspection."""
        if not report.can_start:
            raise ValueError("验证清单检查未通过，不能创建验证任务。")
        manifest = model.record.manifest
        task = ValidationTask(
            manifest_path=report.manifest_path,
            data_root=report.data_root,
            output_directory=self.output_directory,
            model_name=model.record.model_name,
            model_version=model.record.model_version,
            runtime_version=model.runtime_version,
            checkpoint_sha256=str(manifest.get("checkpoint_sha256", "")),
            device=model.device,
            labels=list(model.labels),
            decision_mode=model.prediction_mode,
            status=ValidationStatus.CHECKED,
            metadata_fields=list(report.metadata_fields),
            samples=deepcopy(report.samples),
            manifest_report=report,
        )
        task.refresh_counts()
        return task

    def export_results(
        self,
        task: ValidationTask,
        *,
        destination: Path | None = None,
    ) -> ValidationExportResult:
        """Write the complete result contract and a self-contained UTF-8 report."""
        root = Path(destination) if destination is not None else self.output_directory
        root.mkdir(parents=True, exist_ok=True)
        name = (
            f"validation_{task.created_at.astimezone().strftime('%Y%m%d-%H%M%S')}_"
            f"{task.task_id[:8]}"
        )
        output = self._unique_directory(root / name)
        output.mkdir(parents=True)
        paths = ValidationExportResult(
            output_directory=output,
            task_path=output / "task.json",
            summary_path=output / "summary.json",
            sample_results_path=output / "sample_results.csv",
            label_metrics_path=output / "label_metrics.csv",
            combination_metrics_path=output / "combination_metrics.csv",
            confusion_matrix_path=output / "confusion_matrix.csv",
            group_metrics_path=output / "group_metrics.csv",
            errors_path=output / "errors.csv",
            report_path=output / "report.html",
        )
        if destination is None:
            task.output_directory = output
        metrics = task.metrics or calculate_validation_metrics(
            task.samples,
            task.labels,
            decision_mode=task.decision_mode,
            group_fields=task.metadata_fields,
        )
        task.metrics = metrics
        self._write_csv(paths.sample_results_path, SAMPLE_COLUMNS, self._sample_rows(task))
        self._write_csv(paths.label_metrics_path, LABEL_COLUMNS, metrics.get("labels", []))
        self._write_csv(
            paths.combination_metrics_path,
            COMBINATION_COLUMNS,
            metrics.get("combinations", []),
        )
        self._write_confusion(paths.confusion_matrix_path, metrics.get("confusion_matrix", {}))
        group_rows = [
            row for field_rows in metrics.get("groups", {}).values() for row in field_rows
        ]
        self._write_csv(paths.group_metrics_path, GROUP_COLUMNS, group_rows)
        self._write_csv(paths.errors_path, ERROR_COLUMNS, self._error_rows(task))
        self._write_json(paths.task_path, self._task_payload(task))
        self._write_json(paths.summary_path, self._summary_payload(task, paths))
        self._write_html(paths.report_path, task)
        if destination is None:
            task.exported_files = {
                name: str(value)
                for name, value in asdict(paths).items()
                if name != "output_directory"
            }
        return paths

    def load_history(self, directory: Path) -> ValidationTask:
        """Restore all validation views without opening a model session."""
        root = Path(directory)
        required = {
            "task": root / "task.json",
            "summary": root / "summary.json",
            "samples": root / "sample_results.csv",
            "labels": root / "label_metrics.csv",
            "combinations": root / "combination_metrics.csv",
            "confusion": root / "confusion_matrix.csv",
            "groups": root / "group_metrics.csv",
            "errors": root / "errors.csv",
            "report": root / "report.html",
        }
        missing = [path.name for path in required.values() if not path.is_file()]
        if missing:
            raise ValueError(f"历史验证结果缺少文件：{', '.join(missing)}")
        payload = json.loads(required["task"].read_text(encoding="utf-8"))
        summary = json.loads(required["summary"].read_text(encoding="utf-8"))
        task_info = payload["task"]
        model_info = payload["model"]
        task = ValidationTask(
            manifest_path=Path(task_info["manifest_path"]),
            data_root=Path(task_info["data_root"]),
            output_directory=root,
            task_id=str(task_info["task_id"]),
            created_at=self._datetime(task_info.get("created_at")) or datetime.now(UTC),
            started_at=self._datetime(task_info.get("started_at")),
            finished_at=self._datetime(task_info.get("finished_at")),
            model_name=str(model_info.get("model_name", "")),
            model_version=str(model_info.get("model_version", "")),
            runtime_version=str(model_info.get("runtime_version", "")),
            checkpoint_sha256=str(model_info.get("checkpoint_sha256", "")),
            device=str(model_info.get("device", "")),
            labels=[str(value) for value in model_info.get("labels", [])],
            decision_mode=str(model_info.get("decision_mode", "")),
            status=ValidationStatus(task_info["status"]),
            metadata_fields=[str(value) for value in task_info.get("metadata_fields", [])],
            metrics=dict(payload.get("metrics", summary.get("metrics", {}))),
            samples=[self._sample_from_payload(row) for row in payload.get("samples", [])],
        )
        with required["samples"].open(encoding="utf-8-sig", newline="") as handle:
            csv_count = sum(1 for _ in csv.DictReader(handle))
        if csv_count != len(task.samples):
            raise ValueError("task.json 与 sample_results.csv 的样本数量不一致。")
        csv_rows = {}
        for name in ("labels", "combinations", "confusion", "groups", "errors"):
            with required[name].open(encoding="utf-8-sig", newline="") as handle:
                csv_rows[name] = list(csv.DictReader(handle))
        expected_group_count = sum(len(rows) for rows in task.metrics.get("groups", {}).values())
        expected_error_count = sum(
            sample.status != ValidationSampleStatus.SUCCESS for sample in task.samples
        )
        expected_counts = {
            "labels": len(task.metrics.get("labels", [])),
            "combinations": len(task.metrics.get("combinations", [])),
            "confusion": len(task.metrics.get("confusion_matrix", {}).get("labels", [])),
            "groups": expected_group_count,
            "errors": expected_error_count,
        }
        for name, expected in expected_counts.items():
            if len(csv_rows[name]) != expected:
                raise ValueError(f"{required[name].name} 与 task.json 的聚合结果不一致。")
        task.refresh_counts()
        task.exported_files = {name: str(path) for name, path in required.items()}
        return task

    def validate_existing_batch(
        self,
        batch_directory: Path,
        report: ManifestValidationReport,
        model: LoadedModel,
    ) -> ValidationTask:
        """Join existing batch predictions to manifest truth without inference."""
        if not report.can_start:
            raise ValueError("验证清单检查未通过。")
        batch = BatchPredictionService(self.output_directory).load_history(batch_directory)
        if (batch.model_name, batch.model_version) != (
            model.record.model_name,
            model.record.model_version,
        ):
            raise ValueError("批量结果记录的模型名称或版本与当前激活模型不一致。")
        successful = [item for item in batch.items if item.status == BatchItemStatus.SUCCESS]
        normalized_paths = [
            BatchPredictionService.normalized_path(item.file_path) for item in batch.items
        ]
        if len(normalized_paths) != len(set(normalized_paths)):
            raise ValueError("已有批量结果包含重复 file_path，无法安全关联真实标签。")
        result_labels = next(
            (
                [str(value) for value in (item.result or {}).get("labels", [])]
                for item in successful
                if item.result
            ),
            [],
        )
        if result_labels != list(model.labels):
            raise ValueError("批量结果标签顺序与当前激活模型不一致。")
        by_path = {
            BatchPredictionService.normalized_path(item.file_path): item for item in batch.items
        }
        task = self.create_task(report, model)
        for sample in task.samples:
            if sample.status == ValidationSampleStatus.SKIPPED:
                continue
            item = by_path.get(BatchPredictionService.normalized_path(sample.file_path))
            if item is None:
                sample.status = ValidationSampleStatus.SKIPPED
                sample.error_type = "UnmatchedBatchResult"
                sample.error_message = "批量结果中没有匹配到该 file_path。"
                continue
            if item.status != BatchItemStatus.SUCCESS or item.result is None:
                sample.status = ValidationSampleStatus.INFERENCE_FAILED
                sample.error_type = item.error_type or "BatchPredictionFailed"
                sample.error_message = item.error_message or "已有批量预测结果失败。"
                sample.elapsed_ms = item.elapsed_ms
                continue
            apply_validation_result(
                sample,
                dict(item.result),
                task.labels,
                task.decision_mode,
            )
            sample.elapsed_ms = item.elapsed_ms
            sample.status = ValidationSampleStatus.SUCCESS
        task.started_at = batch.started_at
        task.finished_at = datetime.now(UTC)
        task.status = ValidationStatus.COMPLETED_WITH_ERRORS
        task.refresh_counts()
        task.metrics = calculate_validation_metrics(
            task.samples,
            task.labels,
            decision_mode=task.decision_mode,
            group_fields=task.metadata_fields,
        )
        if not task.inference_failed_count and not task.skipped_count:
            task.status = ValidationStatus.COMPLETED
        return task

    @staticmethod
    def _sample_rows(task: ValidationTask) -> list[dict[str, Any]]:
        return [
            {
                "sequence": sample.sequence,
                "file_path": str(sample.file_path),
                "status": sample.status.value,
                "true_combination": sample.true_combination,
                "predicted_combination": sample.predicted_combination,
                "true_label_vector": _json(sample.true_label_vector),
                "predicted_label_vector": _json(sample.predicted_label_vector),
                "predicted_sources": _json(sample.predicted_sources),
                "display_probabilities": _json(sample.display_probabilities),
                "multilabel_probabilities": _json(sample.multilabel_probabilities),
                "label_marginal_probabilities": _json(sample.label_marginal_probabilities),
                "combination_labels": _json(sample.combination_labels),
                "combination_probabilities": _json(sample.combination_probabilities),
                "thresholds": _json(sample.thresholds),
                "thresholds_applicable": sample.thresholds_applicable,
                "exact_match": sample.exact_match,
                "false_positive_labels": _json(sample.false_positive_labels),
                "false_negative_labels": _json(sample.false_negative_labels),
                "true_source_count": sample.true_source_count,
                "predicted_source_count": sample.predicted_source_count,
                "confidence": sample.confidence,
                "confidence_margin": sample.confidence_margin,
                "elapsed_ms": sample.elapsed_ms,
                "error_type": sample.error_type,
                "error_message": sample.error_message,
                "input_shape": _json(sample.input_shape),
                "metadata": _json(sample.metadata),
            }
            for sample in task.samples
        ]

    @staticmethod
    def _error_rows(task: ValidationTask) -> list[dict[str, Any]]:
        return [
            {
                "sequence": sample.sequence,
                "file_path": str(sample.file_path),
                "status": sample.status.value,
                "true_combination": sample.true_combination,
                "error_type": sample.error_type,
                "error_message": sample.error_message,
                "metadata": _json(sample.metadata),
            }
            for sample in task.samples
            if sample.status
            in {
                ValidationSampleStatus.INFERENCE_FAILED,
                ValidationSampleStatus.SKIPPED,
                ValidationSampleStatus.STOPPED,
            }
        ]

    @staticmethod
    def _task_payload(task: ValidationTask) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task": {
                "task_id": task.task_id,
                "created_at": _iso(task.created_at),
                "started_at": _iso(task.started_at),
                "finished_at": _iso(task.finished_at),
                "manifest_path": str(task.manifest_path),
                "data_root": str(task.data_root),
                "status": task.status.value,
                "metadata_fields": task.metadata_fields,
                "output_directory": str(task.output_directory),
            },
            "model": {
                "model_name": task.model_name,
                "model_version": task.model_version,
                "runtime_version": task.runtime_version,
                "checkpoint_sha256": task.checkpoint_sha256,
                "device": task.device,
                "labels": task.labels,
                "decision_mode": task.decision_mode,
            },
            "metrics": task.metrics,
            "samples": [ValidationService._sample_payload(sample) for sample in task.samples],
        }

    @staticmethod
    def _sample_payload(sample: ValidationSample) -> dict[str, Any]:
        payload = asdict(sample)
        payload["file_path"] = str(sample.file_path)
        payload["status"] = sample.status.value
        return payload

    @staticmethod
    def _sample_from_payload(payload: dict[str, Any]) -> ValidationSample:
        values = dict(payload)
        values["file_path"] = Path(values["file_path"])
        values["status"] = ValidationSampleStatus(values["status"])
        return ValidationSample(**values)

    @staticmethod
    def _summary_payload(task: ValidationTask, paths: ValidationExportResult) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "software_version": __version__,
            "task_id": task.task_id,
            "status": task.status.value,
            "counts": {
                "total": task.total_count,
                "valid": task.valid_count,
                "success": task.success_count,
                "inference_failed": task.inference_failed_count,
                "skipped": task.skipped_count,
            },
            "metrics": task.metrics,
            "output_files": {
                key: str(value) for key, value in asdict(paths).items() if key != "output_directory"
            },
        }

    @staticmethod
    def _write_csv(
        path: Path,
        fields: tuple[str, ...],
        rows: list[dict[str, Any]],
    ) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)

    @staticmethod
    def _write_confusion(path: Path, matrix: dict[str, Any]) -> None:
        labels = [str(value) for value in matrix.get("labels", [])]
        rows = matrix.get("counts", [])
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true\\predicted", *labels])
            for label, counts in zip(labels, rows, strict=False):
                writer.writerow([label, *counts])
        os.replace(temporary, path)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    def _write_html(path: Path, task: ValidationTask) -> None:
        metrics = task.metrics
        overall = metrics.get("overall", {})
        labels = metrics.get("labels", [])
        combinations = metrics.get("combinations", [])
        groups = [row for rows in metrics.get("groups", {}).values() for row in rows]
        errors = [
            sample
            for sample in task.samples
            if sample.status != ValidationSampleStatus.SUCCESS or sample.exact_match is False
        ]
        error_rows = [
            {
                "file": sample.file_path.name,
                "true": sample.true_combination,
                "predicted": sample.predicted_combination,
                "error": sample.error_message,
            }
            for sample in errors[:100]
        ]
        page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Noise Source Studio 验证报告</title>
<style>
body{{font-family:"Microsoft YaHei",sans-serif;margin:32px;color:#223}}
h1,h2{{color:#1f4f82}} table{{border-collapse:collapse;width:100%;margin:12px 0}}
th,td{{border:1px solid #ccd6e0;padding:7px;text-align:left}} th{{background:#edf3f8}}
.meta{{display:grid;grid-template-columns:repeat(2,minmax(220px,1fr));gap:6px 24px}}
</style></head><body>
<h1>Noise Source Studio 模型验证报告</h1>
<div class="meta">
<div>软件版本：{html.escape(__version__)}</div>
<div>模型：{html.escape(task.model_name)} {html.escape(task.model_version)}</div>
<div>Runtime：{html.escape(task.runtime_version)}</div>
<div>设备：{html.escape(task.device)}</div>
<div>Checkpoint SHA256：{html.escape(task.checkpoint_sha256 or "—")}</div>
<div>验证时间：{html.escape(_iso(task.finished_at) or "—")}</div>
<div>数据清单：{html.escape(str(task.manifest_path))}</div>
<div>结果目录：{html.escape(str(task.output_directory))}</div>
</div>
<h2>样本统计与总体指标</h2>{_html_mapping(overall)}
<p>指标分母：推理成功且真实标签有效的样本；推理失败率单独显示。</p>
<h2>逐标签指标</h2>{_html_table(labels)}
<h2>组合指标</h2>{_html_table(combinations)}
<h2>Top-K 组合指标</h2>{_html_mapping(metrics.get("top_k") or {})}
<h2>组合混淆矩阵</h2>{_html_confusion(metrics.get("confusion_matrix", {}))}
<h2>分组指标</h2>{_html_table(groups)}
<h2>典型错误与失败样本摘要</h2>{_html_table(error_rows)}
<h2>结果目录说明</h2>
<p>完整机器可读结果位于本报告同目录的 JSON/CSV 文件中。本报告不嵌入原始信号。</p>
</body></html>"""
        temporary = path.with_suffix(".html.tmp")
        temporary.write_text(page, encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _datetime(value: Any) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    @staticmethod
    def _unique_directory(path: Path) -> Path:
        if not path.exists():
            return path
        return path.with_name(f"{path.name}_{uuid4().hex[:6]}")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _iso(value: datetime | None) -> str:
    return value.isoformat().replace("+00:00", "Z") if value else ""


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _html_mapping(mapping: dict[str, Any]) -> str:
    return _html_table([{"指标": key, "值": _display(value)} for key, value in mapping.items()])


def _html_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>无可用数据。</p>"
    fields = list(rows[0])
    header = "".join(f"<th>{html.escape(str(field))}</th>" for field in fields)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(_display(row.get(field)))}</td>" for field in fields)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _html_confusion(matrix: dict[str, Any]) -> str:
    labels = [str(value) for value in matrix.get("labels", [])]
    counts = matrix.get("counts", [])
    rows = [
        {"真实\\预测": label, **dict(zip(labels, row, strict=False))}
        for label, row in zip(labels, counts, strict=False)
    ]
    return _html_table(rows)


__all__ = ["ValidationService"]
