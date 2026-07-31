"""Atomic model-package import and lightweight JSON registry management."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from packaging.version import InvalidVersion, Version

from noise_source_studio.common.exceptions import (
    ActiveModelDeletionError,
    DuplicateModelError,
    ModelActivationError,
    ModelPackageValidationError,
    RuntimeCompatibilityError,
)
from noise_source_studio.domain.interfaces import InferenceEngine
from noise_source_studio.domain.models import LoadedModel, ModelRecord, PackageInspection

LOGGER = logging.getLogger("noise_source_studio.model_service")
SUPPORTED_PACKAGE_SCHEMAS = {"1.0"}
SUPPORTED_PREDICTION_MODES = {"multilabel", "structured"}
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


class ModelService:
    """Manage verified model packages outside the source tree."""

    def __init__(self, model_directory: Path, engine: InferenceEngine) -> None:
        self.model_directory = Path(model_directory)
        self.registry_path = self.model_directory / "registry.json"
        self.engine = engine
        self.model_directory.mkdir(parents=True, exist_ok=True)
        self._records = self._load_registry()

    def list_models(self) -> tuple[ModelRecord, ...]:
        """Return installed models in stable name/version order."""
        return tuple(sorted(self._records, key=lambda item: (item.model_name, item.model_version)))

    def active_model(self) -> ModelRecord | None:
        """Return the single active registry record, if present."""
        return next((record for record in self._records if record.is_active), None)

    def get_model(self, identifier: str) -> ModelRecord:
        """Return one registered model or raise a concise domain error."""
        for record in self._records:
            if record.identifier == identifier:
                return record
        raise ModelPackageValidationError("所选模型已不存在，请刷新模型列表。")

    def inspect_package(self, package_path: Path) -> PackageInspection:
        """Verify checksums and validate the GUI-supported manifest contract."""
        package = Path(package_path).resolve()
        if not package.is_dir():
            raise ModelPackageValidationError("请选择包含 manifest.json 的模型包目录。")
        verification = self.engine.verify_model_package(package)
        try:
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelPackageValidationError("manifest.json 无法读取或不是有效 JSON。") from exc
        if not isinstance(manifest, dict):
            raise ModelPackageValidationError("manifest.json 顶层必须是对象。")
        self._validate_manifest(manifest, package)
        return PackageInspection(
            package_path=package,
            manifest=manifest,
            verification=verification,
            runtime_version=self.engine.runtime_version,
        )

    def import_package(self, package_path: Path) -> ModelRecord:
        """Copy, re-verify and atomically register one model package."""
        source = self.inspect_package(package_path)
        name = str(source.manifest["model_name"])
        version = str(source.manifest["model_version"])
        identifier = f"{name}@{version}"
        if any(record.identifier == identifier for record in self._records):
            raise DuplicateModelError(f"模型 {name} {version} 已经导入。")

        target_parent = self.model_directory / name
        target = target_parent / version
        if target.exists():
            raise DuplicateModelError(f"模型目录已存在：{target}")
        target_parent.mkdir(parents=True, exist_ok=True)
        temporary = target_parent / f".{version}.{uuid4().hex}.importing"
        original_records = list(self._records)
        try:
            shutil.copytree(source.package_path, temporary)
            copied = self.inspect_package(temporary)
            os.replace(temporary, target)
            record = ModelRecord(
                model_name=name,
                model_version=version,
                package_path=target,
                manifest=copied.manifest,
                installed_at=self._utc_now(),
                is_active=False,
                integrity_status="valid",
            )
            self._records.append(record)
            self._save_registry()
        except Exception:
            self._records = original_records
            shutil.rmtree(temporary, ignore_errors=True)
            if target.exists() and not any(
                item.package_path == target for item in original_records
            ):
                shutil.rmtree(target, ignore_errors=True)
            raise
        return record

    def activate_model(self, identifier: str, *, device: str = "auto") -> LoadedModel:
        """Verify and load one model, then atomically mark it active."""
        record = self.get_model(identifier)
        try:
            self.inspect_package(record.package_path)
            inspection = self.engine.load_model(record.package_path, device=device)
        except Exception:
            LOGGER.exception(
                "Model activation failed; previous active record and session are retained | "
                "identifier=%s | device=%s",
                identifier,
                device,
            )
            raise

        updated = [
            replace(
                item,
                is_active=item.identifier == identifier,
                integrity_status=(
                    "valid" if item.identifier == identifier else item.integrity_status
                ),
            )
            for item in self._records
        ]
        original_records = self._records
        self._records = updated
        try:
            self._save_registry()
        except Exception as exc:
            self._records = original_records
            raise ModelActivationError("模型已加载，但活动状态无法保存。") from exc

        active = self.get_model(identifier)
        return LoadedModel(
            record=active,
            runtime_version=str(inspection.get("runtime_version", self.engine.runtime_version)),
            device=str(inspection.get("device", "unknown")),
            prediction_mode=str(
                inspection.get("prediction_mode", active.manifest.get("prediction_mode", "unknown"))
            ),
            labels=tuple(str(label) for label in inspection.get("labels", [])),
            inspection=inspection,
        )

    def check_integrity(self, identifier: str) -> PackageInspection:
        """Re-check an installed package and persist its integrity state."""
        record = self.get_model(identifier)
        try:
            inspection = self.inspect_package(record.package_path)
        except Exception:
            self._replace_record(record, integrity_status="invalid")
            raise
        self._replace_record(record, integrity_status="valid")
        return inspection

    def delete_model(self, identifier: str) -> None:
        """Delete only an inactive registered model."""
        record = self.get_model(identifier)
        if record.is_active:
            raise ActiveModelDeletionError("活动模型不能删除，请先激活其他模型。")
        resolved_root = self.model_directory.resolve()
        resolved_package = record.package_path.resolve()
        if resolved_root not in resolved_package.parents:
            raise ModelPackageValidationError("拒绝删除模型目录之外的路径。")

        temporary = record.package_path.with_name(
            f".{record.package_path.name}.{uuid4().hex}.deleting"
        )
        os.replace(record.package_path, temporary)
        original_records = list(self._records)
        self._records = [item for item in self._records if item.identifier != identifier]
        try:
            self._save_registry()
        except Exception:
            self._records = original_records
            os.replace(temporary, record.package_path)
            raise
        shutil.rmtree(temporary)

    def _replace_record(self, record: ModelRecord, **changes: Any) -> None:
        self._records = [
            replace(item, **changes) if item.identifier == record.identifier else item
            for item in self._records
        ]
        self._save_registry()

    def _validate_manifest(self, manifest: dict[str, Any], package: Path) -> None:
        required = (
            "model_name",
            "model_version",
            "package_schema_version",
            "runtime_version",
            "labels",
            "prediction_mode",
        )
        missing = [key for key in required if key not in manifest]
        if missing:
            raise ModelPackageValidationError(f"模型 manifest 缺少字段：{', '.join(missing)}")
        schema = str(manifest["package_schema_version"])
        if schema not in SUPPORTED_PACKAGE_SCHEMAS:
            raise ModelPackageValidationError(f"不支持模型包 schema：{schema}")
        name = str(manifest["model_name"])
        version = str(manifest["model_version"])
        if not SAFE_COMPONENT.fullmatch(name) or not SAFE_COMPONENT.fullmatch(version):
            raise ModelPackageValidationError("模型名称或版本包含不安全的路径字符。")
        labels = manifest["labels"]
        if (
            not isinstance(labels, list)
            or not labels
            or not all(isinstance(label, str) and label.strip() for label in labels)
        ):
            raise ModelPackageValidationError("模型标签列表必须是非空字符串数组。")
        mode = str(manifest["prediction_mode"])
        if mode not in SUPPORTED_PREDICTION_MODES:
            raise ModelPackageValidationError(f"不支持 prediction mode：{mode}")
        if not (package / "model.pt").is_file():
            raise ModelPackageValidationError("模型包缺少 model.pt。")
        self._check_runtime_compatibility(str(manifest["runtime_version"]))

    def _check_runtime_compatibility(self, required: str) -> None:
        installed = self.engine.runtime_version
        try:
            required_version = Version(required)
            installed_version = Version(installed)
        except InvalidVersion as exc:
            raise RuntimeCompatibilityError(
                f"runtime 版本格式无效：需要 {required}，当前 {installed}"
            ) from exc
        if (
            required_version.major != installed_version.major
            or installed_version < required_version
        ):
            raise RuntimeCompatibilityError(
                f"runtime 版本不兼容：模型需要 {required}，当前安装 {installed}。"
            )

    def _load_registry(self) -> list[ModelRecord]:
        if not self.registry_path.exists():
            return []
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
            records = [ModelRecord.from_dict(item) for item in payload.get("models", [])]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            LOGGER.exception("Model registry is invalid | path=%s", self.registry_path)
            return []
        active_seen = False
        normalized: list[ModelRecord] = []
        for record in records:
            keep_active = record.is_active and not active_seen
            active_seen = active_seen or keep_active
            normalized.append(replace(record, is_active=keep_active))
        return normalized

    def _save_registry(self) -> None:
        self.model_directory.mkdir(parents=True, exist_ok=True)
        temporary = self.registry_path.with_suffix(".json.tmp")
        payload = {
            "schema_version": 1,
            "models": [record.to_dict() for record in self._records],
        }
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.registry_path)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["ModelService"]
