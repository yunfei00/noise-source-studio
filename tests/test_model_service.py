"""Model package registry and activation tests with a small fake runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from noise_source_studio.common.exceptions import (
    DuplicateModelError,
    ModelPackageValidationError,
)
from noise_source_studio.domain.models import SignalPreview
from noise_source_studio.services import ModelService


class FakePackageEngine:
    """Checksum-aware fake that avoids loading torch or a real checkpoint."""

    runtime_version = "1.0.0"

    def __init__(self) -> None:
        self.closed = 0
        self.loaded_packages: list[Path] = []

    def verify_model_package(self, package_path: Path) -> dict[str, Any]:
        checksums: dict[str, str] = {}
        for line in (package_path / "sha256.txt").read_text(encoding="utf-8").splitlines():
            expected, filename = line.split(None, 1)
            actual = hashlib.sha256((package_path / filename).read_bytes()).hexdigest()
            if actual != expected:
                raise ModelPackageValidationError(f"SHA256 mismatch for {filename}")
            checksums[filename] = actual
        return {
            "valid": True,
            "checkpoint_sha256": checksums["model.pt"],
        }

    def load_model(self, package_path: Path, *, device: str = "auto") -> dict[str, Any]:
        self.loaded_packages.append(package_path)
        return {
            "runtime_version": "1.0.0",
            "device": "cpu" if device == "auto" else device,
            "prediction_mode": "multilabel",
            "labels": ["fan", "motor", "switch_power"],
        }

    def inspect_model(self) -> dict[str, Any]:
        return {}

    def preview_file(self, file_path: Path) -> SignalPreview:
        raise NotImplementedError

    def predict_file(self, file_path: Path) -> Any:
        raise NotImplementedError

    def export_result(
        self,
        result: Any,
        json_path: Path,
        *,
        contract_path: Path | None = None,
    ) -> tuple[Path, Path | None]:
        raise NotImplementedError

    def close(self) -> None:
        self.closed += 1


def _create_model_package(
    root: Path,
    *,
    name: str = "noise-source-current",
    version: str = "0.1.0",
) -> Path:
    package = root / f"{name}-{version}"
    package.mkdir(parents=True)
    model_bytes = b"fake-checkpoint"
    manifest = {
        "package_schema_version": "1.0",
        "model_name": name,
        "model_version": version,
        "runtime_version": "1.0.0",
        "prediction_mode": "multilabel",
        "labels": ["fan", "motor", "switch_power"],
        "checkpoint_sha256": hashlib.sha256(model_bytes).hexdigest(),
    }
    payloads = {
        "model.pt": model_bytes,
        "manifest.json": json.dumps(manifest).encode(),
        "metrics.json": b"{}",
        "preprocess.json": b"{}",
        "labels.json": b'{"labels":["fan","motor","switch_power"]}',
        "README.md": b"# Fake",
    }
    for filename, payload in payloads.items():
        (package / filename).write_bytes(payload)
    checksums = [
        f"{hashlib.sha256(payload).hexdigest()}  {filename}"
        for filename, payload in payloads.items()
    ]
    (package / "sha256.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return package


def test_model_package_import_is_atomic_and_registered(tmp_path: Path) -> None:
    engine = FakePackageEngine()
    service = ModelService(tmp_path / "installed", engine)
    source = _create_model_package(tmp_path / "handoff")

    record = service.import_package(source)

    assert record.package_path == tmp_path / "installed" / record.model_name / record.model_version
    assert record.package_path.is_dir()
    assert record.integrity_status == "valid"
    assert service.list_models() == (record,)
    persisted = json.loads(service.registry_path.read_text(encoding="utf-8"))
    assert persisted["models"][0]["model_name"] == "noise-source-current"
    assert not list((tmp_path / "installed").rglob("*.importing"))


def test_corrupted_sha256_is_rejected_without_partial_install(tmp_path: Path) -> None:
    engine = FakePackageEngine()
    service = ModelService(tmp_path / "installed", engine)
    source = _create_model_package(tmp_path / "handoff")
    (source / "model.pt").write_bytes(b"corrupted")

    with pytest.raises(ModelPackageValidationError, match="SHA256"):
        service.import_package(source)

    assert service.list_models() == ()
    assert not list((tmp_path / "installed").rglob("*.importing"))


def test_duplicate_model_version_is_rejected(tmp_path: Path) -> None:
    service = ModelService(tmp_path / "installed", FakePackageEngine())
    source = _create_model_package(tmp_path / "handoff")
    service.import_package(source)

    with pytest.raises(DuplicateModelError, match="已经导入"):
        service.import_package(source)


def test_model_activation_persists_one_active_record(tmp_path: Path) -> None:
    engine = FakePackageEngine()
    service = ModelService(tmp_path / "installed", engine)
    first = service.import_package(_create_model_package(tmp_path / "first", version="0.1.0"))
    second = service.import_package(_create_model_package(tmp_path / "second", version="0.2.0"))

    loaded_first = service.activate_model(first.identifier)
    loaded_second = service.activate_model(second.identifier)

    assert loaded_first.device == "cpu"
    assert loaded_second.record.identifier == second.identifier
    assert [record.is_active for record in service.list_models()] == [False, True]
    assert engine.loaded_packages == [first.package_path, second.package_path]
