"""Safe inference placeholder used until a model adapter is configured."""

from pathlib import Path
from typing import Any, NoReturn

from noise_source_studio.common.exceptions import InferenceEngineNotConfiguredError
from noise_source_studio.domain.models import SignalPreview

ERROR_MESSAGE = "推理引擎尚未配置，请先导入兼容模型。"


class UnconfiguredInferenceEngine:
    """Inference engine that fails explicitly instead of fabricating output."""

    @property
    def runtime_version(self) -> str:
        """Return an explicit unconfigured marker."""
        return "unconfigured"

    def verify_model_package(self, package_path: Path) -> NoReturn:
        """Reject package verification while no adapter exists."""
        del package_path
        raise InferenceEngineNotConfiguredError(ERROR_MESSAGE)

    def load_model(self, model_path: Path, *, device: str = "auto") -> NoReturn:
        """Reject model loading because no concrete adapter is available."""
        del model_path, device
        raise InferenceEngineNotConfiguredError(ERROR_MESSAGE)

    def inspect_model(self) -> NoReturn:
        """Reject session inspection while no adapter exists."""
        raise InferenceEngineNotConfiguredError(ERROR_MESSAGE)

    def preview_file(self, file_path: Path) -> SignalPreview:
        """Reject parsing while no adapter exists."""
        del file_path
        raise InferenceEngineNotConfiguredError(ERROR_MESSAGE)

    def predict_file(self, file_path: Path) -> Any:
        """Reject prediction because no concrete adapter is available."""
        del file_path
        raise InferenceEngineNotConfiguredError(ERROR_MESSAGE)

    def export_result(
        self,
        result: Any,
        json_path: Path,
        *,
        contract_path: Path | None = None,
    ) -> NoReturn:
        """Reject exports while no adapter exists."""
        del result, json_path, contract_path
        raise InferenceEngineNotConfiguredError(ERROR_MESSAGE)

    def close(self) -> None:
        """Release resources; the unconfigured engine owns none."""
