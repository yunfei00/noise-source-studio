"""Optional end-to-end comparison against the delivered golden model."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from noise_source_studio.infrastructure.inference import RuntimeAdapter

ENVIRONMENT_VARIABLES = (
    "NOISE_STUDIO_MODEL_PACKAGE",
    "NOISE_STUDIO_GOLDEN_CSV",
    "NOISE_STUDIO_GOLDEN_RESULT",
)


def _required_paths() -> tuple[Path, Path, Path]:
    values = [os.environ.get(name) for name in ENVIRONMENT_VARIABLES]
    if not all(values):
        pytest.skip("未配置真实模型黄金样本环境变量")
    return tuple(Path(value) for value in values if value is not None)  # type: ignore[return-value]


def _value(result: Any, key: str) -> Any:
    return result[key] if isinstance(result, dict) else getattr(result, key)


def test_golden_model_matches_runtime_contract() -> None:
    package, csv_path, golden_path = _required_paths()
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    engine = RuntimeAdapter()

    try:
        verification = engine.verify_model_package(package)
        inspection = engine.load_model(package, device="auto")
        result = engine.predict_file(csv_path)
    finally:
        engine.close()

    assert verification["checkpoint_sha256"] == golden["checkpoint_sha256"]
    assert inspection["labels"] == golden["labels"]
    assert _value(result, "labels") == golden["labels"]
    assert _value(result, "decision_mode") == golden["decision_mode"]
    assert _value(result, "multilabel_probabilities") == pytest.approx(
        golden["multilabel_probabilities"],
        rel=1e-6,
        abs=1e-7,
    )
    assert _value(result, "combination_probabilities") == golden["combination_probabilities"]
    assert _value(result, "label_marginal_probabilities") == pytest.approx(
        golden["label_marginal_probabilities"],
        rel=1e-6,
        abs=1e-7,
    )
    assert _value(result, "decoded_label_vector") == golden["decoded_label_vector"]
    assert _value(result, "predicted_combination") == golden["predicted_combination"]
    assert _value(result, "input_shape") == golden["input_shape"]
