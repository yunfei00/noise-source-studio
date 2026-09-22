"""Centralized, dependency-light validation metric calculations."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from statistics import mean
from typing import Any

from noise_source_studio.domain.confidence import is_low_confidence
from noise_source_studio.domain.validation import ValidationSample, ValidationSampleStatus


def legal_combinations(label_count: int) -> list[str]:
    """Return every legal non-empty binary combination in stable order."""
    return [format(value, f"0{label_count}b") for value in range(1, 2**label_count)]


def calculate_validation_metrics(
    samples: Sequence[ValidationSample],
    labels: Sequence[str],
    *,
    decision_mode: str,
    group_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Calculate all aggregate metrics using successful, valid samples only."""
    label_order = list(labels)
    successful = [
        sample
        for sample in samples
        if sample.status == ValidationSampleStatus.SUCCESS
        and len(sample.true_label_vector) == len(label_order)
        and len(sample.predicted_label_vector) == len(label_order)
    ]
    failed = [
        sample for sample in samples if sample.status == ValidationSampleStatus.INFERENCE_FAILED
    ]
    label_rows = _label_metrics(successful, label_order)
    overall = _overall_metrics(successful, failed, label_rows, len(label_order))
    combinations, confusion = _combination_metrics(successful, len(label_order))
    top_k = _top_k_metrics(successful) if decision_mode == "structured" else None
    selected_fields = list(group_fields or _available_group_fields(samples))
    groups = {
        field: _group_metrics(successful, failed, label_order, field)
        for field in selected_fields
    }
    return {
        "denominator": {
            "definition": "推理成功且真实标签有效的样本",
            "sample_count": len(successful),
            "inference_failed_count": len(failed),
        },
        "overall": overall,
        "labels": label_rows,
        "combinations": combinations,
        "confusion_matrix": {
            "labels": legal_combinations(len(label_order)),
            "counts": confusion,
        },
        "top_k": top_k,
        "groups": groups,
        "zero_division": "分母为零时表格显示 —；宏平均按 sklearn zero_division=0 计入 0。",
    }


def _label_metrics(
    samples: Sequence[ValidationSample], labels: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        tp = sum(
            sample.true_label_vector[index] == 1 and sample.predicted_label_vector[index] == 1
            for sample in samples
        )
        fp = sum(
            sample.true_label_vector[index] == 0 and sample.predicted_label_vector[index] == 1
            for sample in samples
        )
        tn = sum(
            sample.true_label_vector[index] == 0 and sample.predicted_label_vector[index] == 0
            for sample in samples
        )
        fn = sum(
            sample.true_label_vector[index] == 1 and sample.predicted_label_vector[index] == 0
            for sample in samples
        )
        precision = _ratio_or_none(tp, tp + fp)
        recall = _ratio_or_none(tp, tp + fn)
        rows.append(
            {
                "label": label,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
                "specificity": _ratio_or_none(tn, tn + fp),
                "false_positive_rate": _ratio_or_none(fp, fp + tn),
                "false_negative_rate": _ratio_or_none(fn, fn + tp),
                "support": tp + fn,
                "predicted_positive_count": tp + fp,
            }
        )
    return rows


def _overall_metrics(
    samples: Sequence[ValidationSample],
    failed: Sequence[ValidationSample],
    label_rows: Sequence[dict[str, Any]],
    label_count: int,
) -> dict[str, Any]:
    count = len(samples)
    tp = sum(row["tp"] for row in label_rows)
    fp = sum(row["fp"] for row in label_rows)
    fn = sum(row["fn"] for row in label_rows)
    micro_precision = _ratio(tp, tp + fp)
    micro_recall = _ratio(tp, tp + fn)
    f1_values = [float(row["f1"] or 0.0) for row in label_rows]
    precision_values = [float(row["precision"] or 0.0) for row in label_rows]
    recall_values = [float(row["recall"] or 0.0) for row in label_rows]
    support_total = sum(int(row["support"]) for row in label_rows)
    mismatches = sum(
        sum(true != predicted for true, predicted in zip(
            sample.true_label_vector, sample.predicted_label_vector, strict=True
        ))
        for sample in samples
    )
    latencies = sorted(
        float(sample.elapsed_ms) for sample in samples if sample.elapsed_ms is not None
    )
    return {
        "valid_sample_count": count + len(failed),
        "inference_success_count": count,
        "inference_failed_count": len(failed),
        "inference_failure_rate": _ratio(len(failed), count + len(failed)),
        "exact_match_accuracy": _ratio(
            sum(sample.true_label_vector == sample.predicted_label_vector for sample in samples),
            count,
        ),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": _f1(micro_precision, micro_recall) or 0.0,
        "macro_precision": mean(precision_values) if precision_values else 0.0,
        "macro_recall": mean(recall_values) if recall_values else 0.0,
        "macro_f1": mean(f1_values) if f1_values else 0.0,
        "weighted_f1": (
            sum(float(row["f1"] or 0.0) * int(row["support"]) for row in label_rows)
            / support_total
            if support_total
            else 0.0
        ),
        "hamming_loss": _ratio(mismatches, count * label_count),
        "average_wrong_labels_per_sample": _ratio(mismatches, count),
        "source_count_accuracy": _ratio(
            sum(sample.true_source_count == sample.predicted_source_count for sample in samples),
            count,
        ),
        "overprediction_rate": _ratio(
            sum(sample.predicted_source_count > sample.true_source_count for sample in samples),
            count,
        ),
        "underprediction_rate": _ratio(
            sum(sample.predicted_source_count < sample.true_source_count for sample in samples),
            count,
        ),
        "low_confidence_rate": _ratio(
            sum(is_low_confidence(sample.result or {}) for sample in samples),
            count,
        ),
        "average_inference_ms": mean(latencies) if latencies else 0.0,
        "p95_inference_ms": _percentile(latencies, 0.95),
    }


def _combination_metrics(
    samples: Sequence[ValidationSample], label_count: int
) -> tuple[list[dict[str, Any]], list[list[int]]]:
    combinations = legal_combinations(label_count)
    indexes = {value: index for index, value in enumerate(combinations)}
    matrix = [[0 for _ in combinations] for _ in combinations]
    true_counts = Counter(sample.true_combination for sample in samples)
    predicted_counts = Counter(sample.predicted_combination for sample in samples)
    correct_counts: Counter[str] = Counter()
    mistakes: dict[str, Counter[str]] = defaultdict(Counter)
    confidences: dict[str, list[float]] = defaultdict(list)
    margins: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        if sample.true_combination in indexes and sample.predicted_combination in indexes:
            matrix[indexes[sample.true_combination]][indexes[sample.predicted_combination]] += 1
        if sample.true_combination == sample.predicted_combination:
            correct_counts[sample.true_combination] += 1
        else:
            mistakes[sample.true_combination][sample.predicted_combination] += 1
        if sample.confidence is not None:
            confidences[sample.true_combination].append(sample.confidence)
        if sample.confidence_margin is not None:
            margins[sample.true_combination].append(sample.confidence_margin)

    rows: list[dict[str, Any]] = []
    for combination in combinations:
        support = true_counts[combination]
        predicted = predicted_counts[combination]
        correct = correct_counts[combination]
        precision = _ratio_or_none(correct, predicted)
        recall = _ratio_or_none(correct, support)
        common = mistakes[combination].most_common(1)
        rows.append(
            {
                "true_combination": combination,
                "support": support,
                "exact_count": correct,
                "exact_accuracy": _ratio_or_none(correct, support),
                "most_common_mistake": common[0][0] if common else "",
                "most_common_mistake_count": common[0][1] if common else 0,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
                "average_confidence": (
                    mean(confidences[combination]) if confidences[combination] else None
                ),
                "average_confidence_margin": (
                    mean(margins[combination]) if margins[combination] else None
                ),
            }
        )
    return rows, matrix


def _top_k_metrics(samples: Sequence[ValidationSample]) -> dict[str, Any] | None:
    eligible = [
        sample
        for sample in samples
        if sample.combination_labels
        and len(sample.combination_labels) == len(sample.combination_probabilities)
        and sample.true_combination in sample.combination_labels
    ]
    if not eligible:
        return None
    ranks: list[int] = []
    probabilities: list[float] = []
    for sample in eligible:
        ordered = sorted(
            zip(sample.combination_labels, sample.combination_probabilities, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        rank = next(
            index
            for index, (combination, _probability) in enumerate(ordered, start=1)
            if combination == sample.true_combination
        )
        ranks.append(rank)
        probabilities.append(
            float(
                sample.combination_probabilities[
                    sample.combination_labels.index(sample.true_combination)
                ]
            )
        )
    return {
        "sample_count": len(eligible),
        "top_1_accuracy": _ratio(sum(rank <= 1 for rank in ranks), len(ranks)),
        "top_2_accuracy": _ratio(sum(rank <= 2 for rank in ranks), len(ranks)),
        "average_true_combination_rank": mean(ranks),
        "average_true_combination_probability": mean(probabilities),
        "negative_log_likelihood": mean(-math.log(max(value, 1e-15)) for value in probabilities),
    }


def _available_group_fields(samples: Sequence[ValidationSample]) -> list[str]:
    counts: Counter[str] = Counter(
        key
        for sample in samples
        for key, value in sample.metadata.items()
        if str(value).strip()
    )
    return sorted(key for key, count in counts.items() if count > 0)


def _group_metrics(
    successful: Sequence[ValidationSample],
    failed: Sequence[ValidationSample],
    labels: Sequence[str],
    field: str,
) -> list[dict[str, Any]]:
    values = sorted(
        {
            sample.metadata.get(field, "")
            for sample in [*successful, *failed]
            if sample.metadata.get(field, "")
        },
        key=str.casefold,
    )
    rows: list[dict[str, Any]] = []
    for value in values:
        group_success = [
            sample for sample in successful if sample.metadata.get(field, "") == value
        ]
        group_failed = [sample for sample in failed if sample.metadata.get(field, "") == value]
        label_rows = _label_metrics(group_success, labels)
        overall = _overall_metrics(group_success, group_failed, label_rows, len(labels))
        confidences = [
            sample.confidence for sample in group_success if sample.confidence is not None
        ]
        rows.append(
            {
                "field": field,
                "value": value,
                "sample_count": len(group_success) + len(group_failed),
                "exact_match": overall["exact_match_accuracy"],
                "micro_f1": overall["micro_f1"],
                "macro_f1": overall["macro_f1"],
                "overprediction_rate": overall["overprediction_rate"],
                "underprediction_rate": overall["underprediction_rate"],
                "inference_failed_count": len(group_failed),
                "average_confidence": mean(confidences) if confidences else None,
                "average_elapsed_ms": overall["average_inference_ms"],
            }
        )
    return rows


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _ratio_or_none(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None if precision is None or recall is None else 0.0
    return 2.0 * precision * recall / (precision + recall)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


__all__ = ["calculate_validation_metrics", "legal_combinations"]
