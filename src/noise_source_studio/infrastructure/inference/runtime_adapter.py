"""Adapter around the separately delivered ``noise_source_runtime`` wheel."""

from __future__ import annotations

import importlib
import logging
import math
import os
import platform
from pathlib import Path
from threading import Lock, RLock
from types import ModuleType
from typing import Any

from noise_source_studio.common.exceptions import (
    InferenceEngineNotConfiguredError,
    ModelActivationError,
    ModelPackageValidationError,
    PredictionBusyError,
    PredictionExecutionError,
    PredictionExportError,
    RuntimeUnavailableError,
)
from noise_source_studio.domain.device import DeviceInfo, DeviceProbeReport
from noise_source_studio.domain.models import SignalPreview

MAX_PREVIEW_POINTS = 5000
LOGGER = logging.getLogger("noise_source_studio.runtime_adapter")


class RuntimeAdapter:
    """Keep one reusable runtime session behind a GUI-safe boundary."""

    def __init__(
        self,
        runtime_module: ModuleType | Any | None = None,
        *,
        torch_module: ModuleType | Any | None = None,
    ) -> None:
        self._runtime = runtime_module
        self._torch = torch_module
        self._session: Any | None = None
        self._session_lock = RLock()
        self._operation_lock = Lock()
        self._model_load_count = 0

    @property
    def runtime_version(self) -> str:
        """Return the installed runtime version without exposing its module."""
        runtime = self._get_runtime()
        return str(getattr(runtime, "RUNTIME_VERSION", getattr(runtime, "__version__", "unknown")))

    @property
    def has_session(self) -> bool:
        """Return whether a loaded session is currently retained."""
        with self._session_lock:
            return self._session is not None

    @property
    def model_load_count(self) -> int:
        """Return successful session loads for diagnostics and stability checks."""
        return self._model_load_count

    def verify_model_package(self, package_path: Path) -> dict[str, Any]:
        """Call the runtime package verifier and translate expected failures."""
        try:
            result = self._get_runtime().verify_model_package(package_path)
        except RuntimeUnavailableError:
            raise
        except Exception as exc:
            raise ModelPackageValidationError(
                f"模型包校验失败：{self._concise_runtime_message(exc)}"
            ) from exc
        return dict(result)

    def load_model(self, package_path: Path, *, device: str = "auto") -> dict[str, Any]:
        """Load a candidate session, atomically swap it, then close the old one."""
        package = Path(package_path)
        with self._operation_lock:
            candidate: Any | None = None
            try:
                runtime = self._get_runtime()
                candidate = runtime.InferenceSession.load_model(
                    package / "model.pt",
                    device=device,
                )
                inspection = dict(candidate.inspect_model())
            except RuntimeUnavailableError:
                raise
            except Exception as exc:
                if candidate is not None:
                    try:
                        candidate.close()
                    except Exception:
                        LOGGER.exception("Failed to close rejected candidate session")
                raise ModelActivationError(self._model_load_message(exc)) from exc
            with self._session_lock:
                previous = self._session
                self._session = candidate
                self._model_load_count += 1
            LOGGER.info(
                "Inference session replaced | requested_device=%s | resolved_device=%s",
                device,
                inspection.get("device", "unknown"),
            )
            if previous is not None:
                try:
                    previous.close()
                    LOGGER.info("Previous inference session closed")
                except Exception:
                    LOGGER.exception("Previous inference session close failed after swap")
            return inspection

    def probe_devices(self) -> DeviceProbeReport:
        """Perform a real CUDA tensor probe without exposing torch to GUI pages."""
        processor = (
            platform.processor().strip()
            or os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
        )
        devices = [
            DeviceInfo(
                device_id="cpu",
                display_name=f"CPU · {processor}" if processor else "CPU",
                type="cpu",
                available=True,
                tested=True,
            )
        ]
        try:
            torch = self._get_torch()
        except Exception as exc:
            LOGGER.exception("PyTorch import failed during device probe")
            return DeviceProbeReport.cpu_only(
                devices[0].display_name,
                error_message=f"PyTorch 无法加载：{self._concise_runtime_message(exc)}",
            )

        torch_version = str(getattr(torch, "__version__", "unknown"))
        version_namespace = getattr(torch, "version", None)
        cuda_runtime_version = str(getattr(version_namespace, "cuda", "") or "")
        cuda = getattr(torch, "cuda", None)
        if cuda is None:
            return DeviceProbeReport(
                devices=tuple(devices),
                torch_cuda_available=False,
                cuda_runtime_version=cuda_runtime_version,
                torch_version=torch_version,
                cuda_device_count=0,
                error_message="当前 PyTorch 未提供 CUDA 接口。",
            )

        try:
            cuda_available = bool(cuda.is_available())
        except Exception as exc:
            LOGGER.exception("torch.cuda.is_available failed")
            return DeviceProbeReport(
                devices=tuple(devices),
                torch_cuda_available=False,
                cuda_runtime_version=cuda_runtime_version,
                torch_version=torch_version,
                cuda_device_count=0,
                error_message=(
                    "CUDA 可用性检查失败："
                    f"{self._concise_runtime_message(exc)}"
                ),
            )
        try:
            device_count = int(cuda.device_count())
        except Exception as exc:
            LOGGER.exception("torch.cuda.device_count failed")
            device_count = 0
            count_error = self._concise_runtime_message(exc)
        else:
            count_error = ""

        LOGGER.info(
            "CUDA environment | torch=%s | runtime=%s | is_available=%s | count=%d",
            torch_version,
            cuda_runtime_version,
            cuda_available,
            device_count,
        )
        for index in range(device_count):
            devices.append(
                self._probe_cuda_device(
                    torch,
                    index,
                    cuda_available=cuda_available,
                )
            )
        error_message = count_error
        if not cuda_available and not error_message:
            error_message = "torch.cuda.is_available() 返回 false。"
        if cuda_available and device_count == 0 and not error_message:
            error_message = "CUDA 可用，但没有检测到设备索引。"
        return DeviceProbeReport(
            devices=tuple(devices),
            torch_cuda_available=cuda_available,
            cuda_runtime_version=cuda_runtime_version,
            torch_version=torch_version,
            cuda_device_count=device_count,
            error_message=error_message,
        )

    def inspect_model(self) -> dict[str, Any]:
        """Return details from the retained runtime session."""
        with self._session_lock:
            if self._session is None:
                raise InferenceEngineNotConfiguredError("当前没有已加载的模型。")
            return dict(self._session.inspect_model())

    def preview_file(self, file_path: Path) -> SignalPreview:
        """Parse a CSV using the runtime parser and sample only display values."""
        try:
            parser = importlib.import_module("noise_source_runtime.csv_parser")
            parsed = parser.load_csv_signal(
                file_path,
                require_data_marker=True,
                invalid_row_policy="error",
            )
        except Exception as exc:
            raise PredictionExecutionError(self._csv_error_message(exc)) from exc

        raw_signal = parsed.raw_signal
        point_count = int(len(raw_signal))
        sample_step = max(1, math.ceil(point_count / MAX_PREVIEW_POINTS))
        display_values = tuple(float(value) for value in raw_signal[::sample_step])
        return SignalPreview(
            source_path=Path(file_path),
            display_values=display_values,
            original_point_count=point_count,
            raw_minimum=float(raw_signal.min()),
            raw_maximum=float(raw_signal.max()),
            raw_mean=float(raw_signal.mean()),
            raw_standard_deviation=float(raw_signal.std()),
            parser_mode=str(parsed.parser_mode),
            data_start_line=int(parsed.data_start_line),
            selected_columns=tuple(int(value) for value in parsed.selected_columns),
            encoding=str(parsed.encoding),
            delimiter=str(parsed.delimiter),
        )

    def predict_file(self, file_path: Path) -> Any:
        """Run one inference at a time against the retained session."""
        if not self._operation_lock.acquire(blocking=False):
            raise PredictionBusyError("当前已有推理或模型加载任务正在运行，请稍候。")
        try:
            with self._session_lock:
                session = self._session
            if session is None:
                raise InferenceEngineNotConfiguredError("请先在模型管理中激活一个模型。")
            try:
                return session.predict_file(file_path)
            except Exception as exc:
                raise PredictionExecutionError(self._prediction_error_message(exc)) from exc
        finally:
            self._operation_lock.release()

    def export_result(
        self,
        result: Any,
        json_path: Path,
        *,
        contract_path: Path | None = None,
    ) -> tuple[Path, Path | None]:
        """Use runtime reporting functions; contracts remain explicitly optional."""
        try:
            runtime = self._get_runtime()
            exported_json = Path(runtime.write_prediction_json(result, json_path))
            exported_contract: Path | None = None
            if contract_path is not None:
                with self._session_lock:
                    session = self._session
                if session is None:
                    raise InferenceEngineNotConfiguredError("模型会话已经关闭，无法导出推理契约。")
                exported_contract = Path(
                    runtime.write_inference_contract(result, session, contract_path)
                )
            return exported_json, exported_contract
        except InferenceEngineNotConfiguredError:
            raise
        except Exception as exc:
            raise PredictionExportError(
                f"结果导出失败：{self._concise_runtime_message(exc)}"
            ) from exc

    def close(self) -> None:
        """Close and forget the retained session; repeated calls are safe."""
        with self._session_lock:
            session = self._session
            self._session = None
        if session is not None:
            session.close()

    def _get_runtime(self) -> ModuleType | Any:
        if self._runtime is not None:
            return self._runtime
        try:
            self._runtime = importlib.import_module("noise_source_runtime")
        except (ImportError, OSError) as exc:
            raise RuntimeUnavailableError(
                "未安装 Noise Source Runtime。请先按 README 安装正式 runtime wheel。"
            ) from exc
        return self._runtime

    def _get_torch(self) -> ModuleType | Any:
        if self._torch is None:
            self._torch = importlib.import_module("torch")
        return self._torch

    def _probe_cuda_device(
        self,
        torch: ModuleType | Any,
        index: int,
        *,
        cuda_available: bool,
    ) -> DeviceInfo:
        device_id = f"cuda:{index}"
        name = device_id.upper()
        total_memory: int | None = None
        try:
            name = str(torch.cuda.get_device_name(index))
            properties = torch.cuda.get_device_properties(index)
            memory_value = getattr(properties, "total_memory", None)
            if memory_value is not None:
                total_memory = int(memory_value)
            if not cuda_available:
                return DeviceInfo(
                    device_id=device_id,
                    display_name=f"{device_id.upper()} · {name}",
                    type="cuda",
                    available=False,
                    tested=True,
                    error_message="torch.cuda.is_available() 返回 false。",
                    total_memory=total_memory,
                    index=index,
                )
            tensor = torch.tensor([1.0, 2.0], device=device_id)
            result = tensor * 2.0 + 1.0
            if hasattr(result, "sum"):
                summed = result.sum()
                if hasattr(summed, "item"):
                    summed.item()
            torch.cuda.synchronize(index)
        except Exception as exc:
            LOGGER.exception("CUDA tensor probe failed | device=%s", device_id)
            return DeviceInfo(
                device_id=device_id,
                display_name=f"{device_id.upper()} · {name}",
                type="cuda",
                available=False,
                tested=True,
                error_message=self._concise_runtime_message(exc),
                total_memory=total_memory,
                index=index,
            )
        LOGGER.info(
            "CUDA tensor probe passed | device=%s | name=%s | total_memory=%s",
            device_id,
            name,
            total_memory,
        )
        return DeviceInfo(
            device_id=device_id,
            display_name=f"{device_id.upper()} · {name}",
            type="cuda",
            available=True,
            tested=True,
            total_memory=total_memory,
            index=index,
        )

    @staticmethod
    def _concise_runtime_message(exc: Exception) -> str:
        message = str(exc).strip().splitlines()[0]
        return message or type(exc).__name__

    @classmethod
    def _model_load_message(cls, exc: Exception) -> str:
        message = str(exc).lower()
        if "cuda" in message:
            return "CUDA 初始化或设备选择失败，请检查 PyTorch 与显卡环境。"
        if "state_dict" in message or "size mismatch" in message:
            return "模型结构与 checkpoint 不匹配。"
        if "checkpoint" in message or "model.pt" in message:
            return "checkpoint 加载失败，请重新校验模型包。"
        return f"模型加载失败：{cls._concise_runtime_message(exc)}"

    @classmethod
    def _csv_error_message(cls, exc: Exception) -> str:
        message = str(exc)
        lowered = message.lower()
        if "data" in lowered and ("未找到" in message or "missing" in lowered):
            return "CSV 中缺少独立的 DATA 数据段。"
        if "nan" in lowered or "inf" in lowered or "nonfinite" in lowered:
            return "CSV 包含 NaN 或 Inf，无法执行推理。"
        if "column" in lowered or "列" in message or "row" in lowered:
            return "CSV 数据列格式不符合模型输入要求。"
        return f"CSV 解析失败：{cls._concise_runtime_message(exc)}"

    @classmethod
    def _prediction_error_message(cls, exc: Exception) -> str:
        message = str(exc)
        lowered = message.lower()
        if "data" in lowered and ("未找到" in message or "missing" in lowered):
            return "CSV 中缺少独立的 DATA 数据段。"
        if "nan" in lowered or "inf" in lowered or "nonfinite" in lowered:
            return "CSV 包含 NaN 或 Inf，无法执行推理。"
        if "cuda" in lowered:
            return "CUDA 推理失败，请检查设备和 PyTorch 环境。"
        if "column" in lowered or "列" in message:
            return "CSV 数据列格式不符合模型输入要求。"
        return f"推理失败：{cls._concise_runtime_message(exc)}"


__all__ = ["RuntimeAdapter"]
