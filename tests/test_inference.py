"""Safe placeholder inference tests."""

from pathlib import Path

import pytest

from noise_source_studio.common.exceptions import InferenceEngineNotConfiguredError
from noise_source_studio.infrastructure.inference import UnconfiguredInferenceEngine


@pytest.mark.parametrize("operation", ["load", "predict"])
def test_unconfigured_inference_engine_raises_expected_error(operation: str) -> None:
    engine = UnconfiguredInferenceEngine()

    with pytest.raises(
        InferenceEngineNotConfiguredError,
        match="推理引擎尚未配置",
    ):
        if operation == "load":
            engine.load_model(Path("model.pt"))
        else:
            engine.predict_file(Path("sample.csv"))
