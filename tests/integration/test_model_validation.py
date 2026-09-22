"""Opt-in real-runtime validation workflow test."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from noise_source_studio.domain.models import LoadedModel, ModelRecord
from noise_source_studio.infrastructure.inference import RuntimeAdapter, ValidationWorker
from noise_source_studio.services import ValidationService


@pytest.mark.integration
def test_real_manifest_validation_reuses_one_session(tmp_path: Path) -> None:
    package_text = os.environ.get("NOISE_STUDIO_MODEL_PACKAGE")
    manifest_text = os.environ.get("NOISE_STUDIO_VALIDATION_MANIFEST")
    if not package_text or not manifest_text:
        pytest.skip("未配置真实验证模型包和 manifest 环境变量")
    package = Path(package_text)
    manifest_path = Path(manifest_text)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    engine = RuntimeAdapter()
    inspection = engine.load_model(package, device="cpu")
    record = ModelRecord(
        str(manifest["model_name"]),
        str(manifest["model_version"]),
        package,
        manifest,
        "integration",
        True,
        "valid",
    )
    model = LoadedModel(
        record,
        engine.runtime_version,
        str(inspection["device"]),
        str(inspection["prediction_mode"]),
        tuple(str(value) for value in inspection["labels"]),
        inspection,
    )
    service = ValidationService(tmp_path / "out")
    report = service.inspect_manifest(
        manifest_path,
        model,
        missing_policy="skip",
    )
    assert report.can_start
    task = service.create_task(report, model)
    ValidationWorker(engine, task).run()
    exported = service.export_results(task)

    assert task.success_count > 0
    assert task.inference_failed_count >= 1
    assert task.skipped_count >= 1
    assert sum(sum(row) for row in task.metrics["confusion_matrix"]["counts"]) == (
        task.success_count
    )
    assert len(task.metrics["labels"]) == len(model.labels)
    assert engine.model_load_count == 1
    assert exported.report_path.is_file()

    training_repository = os.environ.get("NOISE_STUDIO_TRAINING_REPO")
    if training_repository and task.decision_mode == "multilabel":
        import numpy as np

        sys.path.insert(0, training_repository)
        try:
            from src.evaluate import compute_combo_confusion, compute_metrics

            successful = [sample for sample in task.samples if sample.status.value == "success"]
            probabilities = np.asarray(
                [sample.multilabel_probabilities for sample in successful],
                dtype=np.float32,
            )
            targets = np.asarray(
                [sample.true_label_vector for sample in successful],
                dtype=np.int32,
            )
            thresholds = np.asarray(successful[0].thresholds, dtype=np.float32)
            predictions = (probabilities >= thresholds.reshape(1, -1)).astype(np.int32)
            training_metrics = compute_metrics(
                probabilities,
                targets,
                task.labels,
                thresholds,
            )
            assert [sample.predicted_label_vector for sample in successful] == [
                row.tolist() for row in predictions
            ]
            assert [sample.predicted_combination for sample in successful] == [
                "".join(str(value) for value in row) for row in predictions
            ]
            assert task.metrics["overall"]["exact_match_accuracy"] == pytest.approx(
                training_metrics["overall"]["exact_match"]
            )
            assert task.metrics["overall"]["micro_f1"] == pytest.approx(
                training_metrics["overall"]["micro_f1"]
            )
            assert task.metrics["overall"]["macro_f1"] == pytest.approx(
                training_metrics["overall"]["macro_f1"]
            )
            for row in task.metrics["labels"]:
                expected = training_metrics["per_source"][row["label"]]
                assert (row["tp"], row["fp"], row["tn"], row["fn"]) == (
                    expected["true_positive"],
                    expected["false_positive"],
                    expected["true_negative"],
                    expected["false_negative"],
                )
            training_confusion = compute_combo_confusion(targets, predictions)
            studio_confusion = task.metrics["confusion_matrix"]
            studio_counts = {
                true_combination: {
                    predicted_combination: studio_confusion["counts"][row_index][
                        column_index
                    ]
                    for column_index, predicted_combination in enumerate(
                        studio_confusion["labels"]
                    )
                }
                for row_index, true_combination in enumerate(studio_confusion["labels"])
            }
            for true_combination, predicted_counts in training_confusion.items():
                normalized_true = "".join(
                    character for character in true_combination if character in {"0", "1"}
                )
                for predicted_combination, count in predicted_counts.items():
                    normalized_prediction = "".join(
                        character
                        for character in predicted_combination
                        if character in {"0", "1"}
                    )
                    assert (
                        studio_counts[normalized_true][normalized_prediction] == count
                    )
        finally:
            sys.path.remove(training_repository)
    engine.close()
