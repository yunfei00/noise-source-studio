"""Map normalized runtime results into authoritative validation decisions."""

from __future__ import annotations

from typing import Any

from noise_source_studio.domain.confidence import (
    candidate_probability_margin,
    result_confidence,
)
from noise_source_studio.domain.validation import ValidationSample
from noise_source_studio.services.result_adapter import display_probabilities


def apply_validation_result(
    sample: ValidationSample,
    payload: dict[str, Any],
    task_labels: list[str],
    decision_mode: str,
) -> None:
    """Apply mode-specific final decisions without recomputing model outputs."""
    labels = [str(value) for value in payload.get("labels", []) or []]
    if labels != task_labels:
        raise ValueError("推理结果标签顺序与验证任务锁定的模型 labels 不一致。")
    result_mode = str(payload.get("decision_mode", ""))
    if result_mode != decision_mode:
        raise ValueError("推理结果 decision_mode 与验证任务锁定模式不一致。")
    if result_mode == "structured":
        predicted = [int(value) for value in payload.get("decoded_label_vector", []) or []]
    elif result_mode == "multilabel":
        probabilities = [
            float(value) for value in payload.get("multilabel_probabilities", []) or []
        ]
        thresholds = [float(value) for value in payload.get("thresholds", []) or []]
        if len(probabilities) != len(task_labels) or len(thresholds) != len(task_labels):
            raise ValueError("multilabel probabilities/thresholds 长度与模型 labels 不一致。")
        predicted = [
            int(probability >= threshold)
            for probability, threshold in zip(probabilities, thresholds, strict=True)
        ]
    else:
        raise ValueError(f"不支持的 decision_mode：{result_mode}")
    if len(predicted) != len(task_labels):
        raise ValueError("最终预测标签向量长度与模型 labels 不一致。")

    sample.result = payload
    sample.labels = labels
    sample.predicted_label_vector = predicted
    sample.predicted_combination = "".join(str(value) for value in predicted)
    sample.predicted_sources = [
        label for label, value in zip(labels, predicted, strict=True) if value
    ]
    sample.display_probabilities = display_probabilities(payload)
    sample.multilabel_probabilities = [
        float(value) for value in payload.get("multilabel_probabilities", []) or []
    ]
    sample.label_marginal_probabilities = [
        float(value) for value in payload.get("label_marginal_probabilities", []) or []
    ]
    sample.combination_labels = [
        str(value) for value in payload.get("combination_labels", []) or []
    ]
    sample.combination_probabilities = [
        float(value) for value in payload.get("combination_probabilities", []) or []
    ]
    sample.thresholds = [float(value) for value in payload.get("thresholds", []) or []]
    sample.thresholds_applicable = bool(payload.get("thresholds_applicable", False))
    sample.exact_match = predicted == sample.true_label_vector
    sample.false_positive_labels = [
        label
        for label, true, predicted_value in zip(
            labels, sample.true_label_vector, predicted, strict=True
        )
        if not true and predicted_value
    ]
    sample.false_negative_labels = [
        label
        for label, true, predicted_value in zip(
            labels, sample.true_label_vector, predicted, strict=True
        )
        if true and not predicted_value
    ]
    sample.true_source_count = sum(sample.true_label_vector)
    sample.predicted_source_count = sum(predicted)
    sample.confidence = result_confidence(payload)
    sample.confidence_margin = candidate_probability_margin(payload)
    sample.input_shape = [int(value) for value in payload.get("input_shape", []) or []]
    sample.error_type = ""
    sample.error_message = ""


__all__ = ["apply_validation_result"]
