"""Central confidence policy shared by result tables, statistics and exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ConfidencePolicy:
    """Configurable defaults for low-confidence classification."""

    structured_top_probability: float = 0.60
    structured_candidate_margin: float = 0.15
    multilabel_threshold_distance: float = 0.10


DEFAULT_CONFIDENCE_POLICY = ConfidencePolicy()


def combination_candidates(payload: dict[str, Any]) -> list[tuple[str, float]]:
    """Return runtime combinations ordered by descending probability."""
    labels = list(payload.get("combination_labels", []) or [])
    probabilities = list(payload.get("combination_probabilities", []) or [])
    pairs = zip(labels, probabilities, strict=False)
    return sorted(
        ((str(label), float(probability)) for label, probability in pairs),
        key=lambda pair: pair[1],
        reverse=True,
    )


def highest_combination_probability(payload: dict[str, Any]) -> float | None:
    candidates = combination_candidates(payload)
    return candidates[0][1] if candidates else None


def candidate_probability_margin(payload: dict[str, Any]) -> float | None:
    candidates = combination_candidates(payload)
    if len(candidates) < 2:
        return None
    return candidates[0][1] - candidates[1][1]


def result_confidence(payload: dict[str, Any]) -> float | None:
    """Return one sortable confidence score without changing result semantics."""
    if payload.get("decision_mode") == "structured":
        return highest_combination_probability(payload)
    probabilities = list(payload.get("multilabel_probabilities", []) or [])
    return max((float(value) for value in probabilities), default=None)


def is_low_confidence(
    payload: dict[str, Any],
    policy: ConfidencePolicy = DEFAULT_CONFIDENCE_POLICY,
) -> bool:
    """Apply the centralized structured or multilabel low-confidence rule."""
    if payload.get("decision_mode") == "structured":
        top = highest_combination_probability(payload)
        margin = candidate_probability_margin(payload)
        return bool(
            (top is not None and top < policy.structured_top_probability)
            or (margin is not None and margin < policy.structured_candidate_margin)
        )
    probabilities = [float(value) for value in payload.get("multilabel_probabilities", []) or []]
    thresholds = [float(value) for value in payload.get("thresholds", []) or []]
    return any(
        abs(probability - threshold) < policy.multilabel_threshold_distance
        for probability, threshold in zip(probabilities, thresholds, strict=False)
    )


def confidence_bucket(payload: dict[str, Any]) -> str:
    score = result_confidence(payload)
    if score is None:
        return "无置信度"
    if score < 0.40:
        return "0–40%"
    if score < 0.60:
        return "40–60%"
    if score < 0.80:
        return "60–80%"
    return "80–100%"


__all__ = [
    "DEFAULT_CONFIDENCE_POLICY",
    "ConfidencePolicy",
    "candidate_probability_margin",
    "combination_candidates",
    "confidence_bucket",
    "highest_combination_probability",
    "is_low_confidence",
    "result_confidence",
]
