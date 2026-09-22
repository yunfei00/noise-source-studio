"""Shared runtime-result normalization for single and batch prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

RESULT_FIELDS = (
    "runtime_version",
    "csv_path",
    "checkpoint_path",
    "model_class",
    "device",
    "labels",
    "decision_mode",
    "thresholds",
    "thresholds_applicable",
    "threshold_source",
    "multilabel_logits",
    "multilabel_probabilities",
    "combination_labels",
    "combination_probabilities",
    "label_marginal_probabilities",
    "decoded_label_vector",
    "predicted_combination",
    "predicted_sources",
    "input_shape",
    "sample_tensor_shape",
    "auxiliary_logits",
    "preprocessing_statistics",
    "csv_contract",
)


def result_value(result: Any, key: str, default: Any = None) -> Any:
    """Read one runtime result field from either an object or mapping."""
    if isinstance(result, Mapping):
        return result.get(key, default)
    return getattr(result, key, default)


def normalize_prediction_result(result: Any) -> dict[str, Any]:
    """Convert a runtime result to plain Python values without retaining tensors."""
    if hasattr(result, "to_dict"):
        try:
            payload = result.to_dict(include_compatibility_aliases=False)
        except TypeError:
            payload = result.to_dict()
    elif is_dataclass(result):
        payload = asdict(result)
    elif isinstance(result, Mapping):
        payload = dict(result)
    else:
        payload = {
            field: result_value(result, field) for field in RESULT_FIELDS if hasattr(result, field)
        }
    normalized = {key: _plain_value(value) for key, value in payload.items()}
    for field in RESULT_FIELDS:
        if field not in normalized:
            value = result_value(result, field)
            if value is not None:
                normalized[field] = _plain_value(value)
    return normalized


def display_probabilities(payload: Mapping[str, Any]) -> list[float]:
    """Return the authoritative per-label probability vector."""
    key = (
        "label_marginal_probabilities"
        if payload.get("decision_mode") == "structured"
        else "multilabel_probabilities"
    )
    return [float(value) for value in payload.get(key, []) or []]


def primary_probability_summary(payload: Mapping[str, Any]) -> str:
    """Return a compact table summary without recomputing decisions."""
    labels = list(payload.get("combination_labels", []) or [])
    probabilities = payload.get("combination_probabilities")
    if labels and probabilities:
        values = list(probabilities)
        index = max(range(len(values)), key=values.__getitem__)
        return f"{labels[index]} {float(values[index]) * 100:.2f}%"
    labels = list(payload.get("labels", []) or [])
    values = display_probabilities(payload)
    if not labels or not values:
        return "—"
    index = max(range(min(len(labels), len(values))), key=values.__getitem__)
    return f"{labels[index]} {values[index] * 100:.2f}%"


def _plain_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_value(item) for item in value]
    if is_dataclass(value):
        return _plain_value(asdict(value))
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return _plain_value(value.tolist())
    if hasattr(value, "item"):
        return _plain_value(value.item())
    return str(value)


__all__ = [
    "display_probabilities",
    "normalize_prediction_result",
    "primary_probability_summary",
    "result_value",
]
