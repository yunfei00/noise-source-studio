"""Device discovery and resolution models shared outside the GUI layer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """One CPU or CUDA device discovered by a safe runtime probe."""

    device_id: str
    display_name: str
    type: str
    available: bool
    tested: bool
    error_message: str = ""
    total_memory: int | None = None
    index: int | None = None


@dataclass(frozen=True, slots=True)
class DeviceProbeReport:
    """Environment and per-device results from one background probe."""

    devices: tuple[DeviceInfo, ...]
    torch_cuda_available: bool
    cuda_runtime_version: str
    torch_version: str
    cuda_device_count: int
    error_message: str = ""
    probed_at: str = ""

    @classmethod
    def cpu_only(
        cls,
        display_name: str = "CPU",
        *,
        error_message: str = "",
    ) -> DeviceProbeReport:
        """Return a report without importing or initializing CUDA."""
        return cls(
            devices=(
                DeviceInfo(
                    device_id="cpu",
                    display_name=display_name,
                    type="cpu",
                    available=True,
                    tested=True,
                ),
            ),
            torch_cuda_available=False,
            cuda_runtime_version="",
            torch_version="",
            cuda_device_count=0,
            error_message=error_message,
            probed_at=datetime.now(UTC).isoformat(),
        )

    def find(self, device_id: str) -> DeviceInfo | None:
        """Return a matching discovered device."""
        return next(
            (device for device in self.devices if device.device_id == device_id),
            None,
        )

    @property
    def available_cuda_devices(self) -> tuple[DeviceInfo, ...]:
        """Return fully tested CUDA devices in index order."""
        return tuple(
            device
            for device in self.devices
            if device.type == "cuda" and device.available and device.tested
        )


@dataclass(frozen=True, slots=True)
class DeviceResolution:
    """Configured device policy resolved to a concrete runtime device."""

    device_preference: str
    resolved_device: str
    report: DeviceProbeReport
    fallback_reason: str = ""


__all__ = ["DeviceInfo", "DeviceProbeReport", "DeviceResolution"]
