"""Compute-device probing and policy tests with no real CUDA requirement."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from noise_source_studio.common.exceptions import DeviceSelectionError
from noise_source_studio.domain.device import DeviceInfo, DeviceProbeReport
from noise_source_studio.infrastructure.inference import RuntimeAdapter
from noise_source_studio.services import DeviceService


class FakeTensor:
    """Support the adapter's lightweight arithmetic probe."""

    def __mul__(self, value: float) -> FakeTensor:
        del value
        return self

    def __add__(self, value: float) -> FakeTensor:
        del value
        return self

    def sum(self) -> FakeTensor:
        return self

    def item(self) -> float:
        return 8.0


class FakeCuda:
    def __init__(self, *, available: bool, names: list[str]) -> None:
        self.available = available
        self.names = names
        self.synchronized: list[int] = []

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return len(self.names)

    def get_device_name(self, index: int) -> str:
        return self.names[index]

    def get_device_properties(self, index: int) -> SimpleNamespace:
        del index
        return SimpleNamespace(total_memory=24 * 1024**3)

    def synchronize(self, index: int) -> None:
        self.synchronized.append(index)


class FakeTorch:
    __version__ = "2.9.0-test"
    version = SimpleNamespace(cuda="13.0")

    def __init__(
        self,
        *,
        available: bool,
        names: list[str],
        failing_devices: set[str] | None = None,
    ) -> None:
        self.cuda = FakeCuda(available=available, names=names)
        self.failing_devices = failing_devices or set()
        self.tensor_devices: list[str] = []

    def tensor(self, values: list[float], *, device: str) -> FakeTensor:
        del values
        self.tensor_devices.append(device)
        if device in self.failing_devices:
            raise RuntimeError("CUDA initialization error from fake tensor")
        return FakeTensor()


class StaticProbeProvider:
    def __init__(self, report: DeviceProbeReport) -> None:
        self.report = report
        self.calls = 0

    def probe_devices(self) -> DeviceProbeReport:
        self.calls += 1
        return self.report


def _report(*cuda_devices: DeviceInfo, cuda_available: bool = True) -> DeviceProbeReport:
    return DeviceProbeReport(
        devices=(
            DeviceInfo("cpu", "CPU", "cpu", True, True),
            *cuda_devices,
        ),
        torch_cuda_available=cuda_available,
        cuda_runtime_version="13.0",
        torch_version="2.9.0-test",
        cuda_device_count=len(cuda_devices),
    )


def _cuda(index: int, *, available: bool = True, error: str = "") -> DeviceInfo:
    return DeviceInfo(
        device_id=f"cuda:{index}",
        display_name=f"CUDA:{index} · Fake GPU {index}",
        type="cuda",
        available=available,
        tested=True,
        error_message=error,
        total_memory=24 * 1024**3,
        index=index,
    )


def test_cpu_is_always_selectable_without_cuda_probe() -> None:
    provider = StaticProbeProvider(_report(_cuda(0)))
    service = DeviceService(provider)

    resolution = service.resolve("cpu")

    assert resolution.resolved_device == "cpu"
    assert resolution.report.find("cpu") is not None
    assert provider.calls == 0


def test_auto_selects_cpu_when_cuda_does_not_exist() -> None:
    service = DeviceService(StaticProbeProvider(_report(cuda_available=False)))

    resolution = service.resolve("auto")

    assert resolution.resolved_device == "cpu"
    assert "CUDA" in resolution.fallback_reason


def test_torch_cuda_false_does_not_run_tensor_probe() -> None:
    torch = FakeTorch(available=False, names=["Installed but unavailable"])
    report = RuntimeAdapter(SimpleNamespace(), torch_module=torch).probe_devices()

    assert not report.torch_cuda_available
    assert report.find("cuda:0") is not None
    assert not report.find("cuda:0").available  # type: ignore[union-attr]
    assert torch.tensor_devices == []


def test_tensor_creation_failure_forces_auto_cpu_fallback() -> None:
    torch = FakeTorch(
        available=True,
        names=["Broken GPU"],
        failing_devices={"cuda:0"},
    )
    report = RuntimeAdapter(SimpleNamespace(), torch_module=torch).probe_devices()
    service = DeviceService(StaticProbeProvider(report))

    resolution = service.resolve("auto")

    assert resolution.resolved_device == "cpu"
    assert "CUDA initialization error" in resolution.fallback_reason


def test_multiple_cuda_devices_are_probed_and_first_success_is_selected() -> None:
    torch = FakeTorch(available=True, names=["GPU A", "GPU B"])
    report = RuntimeAdapter(SimpleNamespace(), torch_module=torch).probe_devices()
    resolution = DeviceService(StaticProbeProvider(report)).resolve("auto")

    assert [item.device_id for item in report.devices] == ["cpu", "cuda:0", "cuda:1"]
    assert torch.tensor_devices == ["cuda:0", "cuda:1"]
    assert torch.cuda.synchronized == [0, 1]
    assert resolution.resolved_device == "cuda:0"


@pytest.mark.parametrize(
    ("preference", "expected"),
    [("cpu", "cpu"), ("cuda:0", "cuda:0")],
)
def test_explicit_device_preference_is_respected(
    preference: str,
    expected: str,
) -> None:
    service = DeviceService(StaticProbeProvider(_report(_cuda(0))))

    assert service.resolve(preference).resolved_device == expected


def test_explicit_cuda_failure_never_silently_falls_back() -> None:
    service = DeviceService(
        StaticProbeProvider(
            _report(
                _cuda(0, available=False, error="tensor probe failed"),
            )
        )
    )

    with pytest.raises(DeviceSelectionError, match="不可用"):
        service.resolve("cuda:0")


def test_resolved_device_display_uses_probed_gpu_name() -> None:
    report = _report(_cuda(0))

    assert DeviceService.display_name("cpu", report) == "CPU"
    assert DeviceService.display_name("cuda:0", report) == "CUDA:0 · Fake GPU 0"


def test_presentation_pages_do_not_import_torch() -> None:
    from noise_source_studio.presentation import pages

    pages_directory = next(iter(pages.__path__))
    for page_path in Path(pages_directory).glob("*.py"):
        assert "import torch" not in page_path.read_text(encoding="utf-8")
