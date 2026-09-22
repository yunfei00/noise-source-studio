"""Compute-device policy, probing, and user-facing resolution rules."""

from __future__ import annotations

import logging
import os
import platform
import re
from typing import Any

from noise_source_studio.common.exceptions import DeviceSelectionError
from noise_source_studio.domain.device import (
    DeviceProbeReport,
    DeviceResolution,
)

LOGGER = logging.getLogger("noise_source_studio.device_service")
CUDA_DEVICE_PATTERN = re.compile(r"^cuda:(0|[1-9]\d*)$")


class DeviceService:
    """Resolve persisted device policy without exposing PyTorch to pages."""

    def __init__(self, probe_provider: Any) -> None:
        self.probe_provider = probe_provider
        self.last_report: DeviceProbeReport | None = None

    def cpu_report(self) -> DeviceProbeReport:
        """Describe CPU without touching any CUDA API."""
        processor = (
            platform.processor().strip()
            or os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
        )
        display_name = f"CPU · {processor}" if processor else "CPU"
        return DeviceProbeReport.cpu_only(display_name)

    def probe_devices(self) -> DeviceProbeReport:
        """Run the provider's full probe; intended for a background thread."""
        LOGGER.info("Device probe started")
        probe = getattr(self.probe_provider, "probe_devices", None)
        if probe is None:
            report = self.cpu_report()
            self.last_report = report
            LOGGER.warning("Device probe provider has no probe_devices; CPU only")
            return report
        report = probe()
        if not isinstance(report, DeviceProbeReport):
            raise DeviceSelectionError("设备探测返回了不受支持的数据结构。")
        self.last_report = report
        LOGGER.info(
            "Device probe completed | torch_cuda_available=%s | cuda_count=%d | "
            "devices=%s",
            report.torch_cuda_available,
            report.cuda_device_count,
            [
                {
                    "device_id": item.device_id,
                    "available": item.available,
                    "tested": item.tested,
                    "name": item.display_name,
                    "error": item.error_message,
                }
                for item in report.devices
            ],
        )
        return report

    def resolve(
        self,
        device_preference: str,
        report: DeviceProbeReport | None = None,
    ) -> DeviceResolution:
        """Resolve a policy to a concrete device using full probe results."""
        preference = device_preference.strip().lower()
        LOGGER.info("Resolving device preference | preference=%s", preference)
        if preference == "cpu":
            cpu_report = report or self.cpu_report()
            return DeviceResolution("cpu", "cpu", cpu_report)

        active_report = report or self.probe_devices()
        if preference == "auto":
            available = active_report.available_cuda_devices
            if available:
                selected = available[0]
                LOGGER.info("Auto device selected | resolved=%s", selected.device_id)
                return DeviceResolution("auto", selected.device_id, active_report)
            reason = self._fallback_reason(active_report)
            LOGGER.warning("Auto device fallback to CPU | reason=%s", reason)
            return DeviceResolution("auto", "cpu", active_report, reason)

        if not CUDA_DEVICE_PATTERN.fullmatch(preference):
            raise DeviceSelectionError("设备策略必须是自动选择、CPU 或 CUDA:N。")
        selected = active_report.find(preference)
        if selected is None:
            raise DeviceSelectionError(
                f"{preference.upper()} 不存在或当前无法检测，请重新检测设备。"
            )
        if not selected.available or not selected.tested:
            detail = selected.error_message or "设备未通过 tensor 计算探测"
            raise DeviceSelectionError(f"{preference.upper()} 不可用：{detail}")
        LOGGER.info("Explicit CUDA device selected | resolved=%s", selected.device_id)
        return DeviceResolution(preference, selected.device_id, active_report)

    @staticmethod
    def display_name(
        resolved_device: str,
        report: DeviceProbeReport | None,
    ) -> str:
        """Format the actual session device for status areas."""
        normalized = resolved_device.strip().lower()
        if normalized == "cpu":
            return "CPU"
        if normalized.startswith("cuda:"):
            device = report.find(normalized) if report is not None else None
            if device is not None:
                name = device.display_name
                prefix = f"{normalized.upper()} · "
                if name.upper().startswith(prefix.upper()):
                    return name
                return f"{normalized.upper()} · {name}"
            return normalized.upper()
        return resolved_device or "待检测"

    @staticmethod
    def cuda_status(report: DeviceProbeReport | None) -> str:
        """Return a concise settings-page CUDA status."""
        if report is None:
            return "尚未检测"
        available = report.available_cuda_devices
        if available:
            return f"{len(available)} 个 CUDA 设备通过完整探测"
        failed = [
            item for item in report.devices if item.type == "cuda" and item.error_message
        ]
        if failed:
            return f"初始化失败：{failed[0].error_message}"
        return report.error_message or "不可用"

    @staticmethod
    def _fallback_reason(report: DeviceProbeReport) -> str:
        failed = [
            device.error_message
            for device in report.devices
            if device.type == "cuda" and device.error_message
        ]
        if failed:
            return f"CUDA 初始化失败：{failed[0]}"
        if not report.torch_cuda_available:
            return report.error_message or "CUDA 不可用"
        return "没有 CUDA 设备通过完整探测"


__all__ = ["DeviceService"]
