"""Single-file prediction orchestration and explicit result export."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from noise_source_studio.domain.interfaces import InferenceEngine
from noise_source_studio.domain.models import ModelRecord, PredictionOutcome, SignalPreview

LOGGER = logging.getLogger("noise_source_studio.prediction_service")
SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


class PredictionService:
    """Lock model metadata around one runtime prediction task."""

    def __init__(self, engine: InferenceEngine, output_directory: Path) -> None:
        self.engine = engine
        self.output_directory = Path(output_directory)

    def preview_file(self, source_path: Path) -> SignalPreview:
        """Parse a selected CSV through the runtime adapter."""
        return self.engine.preview_file(Path(source_path))

    def predict_file(
        self,
        source_path: Path,
        model: ModelRecord,
    ) -> PredictionOutcome:
        """Run one prediction and attach stable task diagnostics."""
        source = Path(source_path)
        task_id = uuid4().hex
        started_at = datetime.now(UTC)
        started_counter = perf_counter()
        try:
            model_inspection = self.engine.inspect_model()
        except Exception:
            model_inspection = {}
        device = str(model_inspection.get("device", "unknown"))
        LOGGER.info(
            "Prediction started | task_id=%s | file=%s | model=%s | device=%s",
            task_id,
            source,
            model.identifier,
            device,
        )
        try:
            result = self.engine.predict_file(source)
        except Exception:
            LOGGER.exception(
                "Prediction failed | task_id=%s | file=%s | model=%s | device=%s",
                task_id,
                source,
                model.identifier,
                device,
            )
            raise
        completed_at = datetime.now(UTC)
        duration = perf_counter() - started_counter
        LOGGER.info(
            "Prediction completed | task_id=%s | file=%s | model=%s | "
            "runtime=%s | device=%s | input_shape=%s | prediction_mode=%s | elapsed=%.6fs",
            task_id,
            source,
            model.identifier,
            getattr(result, "runtime_version", "unknown"),
            getattr(result, "device", "unknown"),
            getattr(result, "input_shape", "unknown"),
            getattr(result, "decision_mode", "unknown"),
            duration,
        )
        return PredictionOutcome(
            task_id=task_id,
            source_path=source,
            model=model,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            result=result,
        )

    def export_result(
        self,
        outcome: PredictionOutcome,
        *,
        include_contract: bool = False,
    ) -> tuple[Path, Path | None]:
        """Export JSON and, only when requested, the inference contract."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        source_name = SAFE_FILENAME.sub("-", outcome.source_path.stem).strip("-") or "input"
        model_version = SAFE_FILENAME.sub("-", outcome.model.model_version).strip("-") or "model"
        base_name = f"{source_name}_{timestamp}_{model_version}"
        json_path = self.output_directory / f"{base_name}_prediction.json"
        contract_path = (
            self.output_directory / f"{base_name}_inference-contract.md"
            if include_contract
            else None
        )
        try:
            exported = self.engine.export_result(
                outcome.result,
                json_path,
                contract_path=contract_path,
            )
        except Exception:
            LOGGER.exception(
                "Prediction export failed | task_id=%s | file=%s | model=%s | device=%s",
                outcome.task_id,
                outcome.source_path,
                outcome.model.identifier,
                getattr(outcome.result, "device", "unknown"),
            )
            raise
        LOGGER.info(
            "Prediction exported | task_id=%s | json=%s | contract=%s",
            outcome.task_id,
            exported[0],
            exported[1],
        )
        return exported


__all__ = ["PredictionService"]
