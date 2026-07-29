"""Safe inference placeholder used until a model adapter is configured."""

from pathlib import Path
from typing import NoReturn

from noise_source_studio.common.exceptions import InferenceEngineNotConfiguredError

ERROR_MESSAGE = "推理引擎尚未配置，请先导入兼容模型。"


class UnconfiguredInferenceEngine:
    """Inference engine that fails explicitly instead of fabricating output."""

    def load_model(self, model_path: Path) -> NoReturn:
        """Reject model loading because no concrete adapter is available."""
        del model_path
        raise InferenceEngineNotConfiguredError(ERROR_MESSAGE)

    def predict_file(self, file_path: Path) -> NoReturn:
        """Reject prediction because no concrete adapter is available."""
        del file_path
        raise InferenceEngineNotConfiguredError(ERROR_MESSAGE)

    def close(self) -> None:
        """Release resources; the unconfigured engine owns none."""
