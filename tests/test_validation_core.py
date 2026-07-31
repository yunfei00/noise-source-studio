"""Manifest contracts and validation metric formula tests."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from noise_source_studio.domain.validation import (
    ValidationSample,
    ValidationSampleStatus,
)
from noise_source_studio.services.validation_manifest import ValidationManifestService
from noise_source_studio.services.validation_metrics import (
    calculate_validation_metrics,
    legal_combinations,
)

LABELS = ["fan", "motor", "switch_power"]


def _signal(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("DATA\n0,1\n", encoding="utf-8")
    return path


def _manifest(path: Path, headers: list[str], rows: list[list[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def test_manifest_relative_path_resolution(tmp_path: Path) -> None:
    source = _signal(tmp_path / "data" / "样本.csv")
    path = _manifest(
        tmp_path / "validation_manifest.csv",
        ["file_path", "true_combination"],
        [["data/样本.csv", "101"]],
    )
    report = ValidationManifestService().inspect(path, LABELS)
    assert report.can_start
    assert report.samples[0].file_path == source.resolve()


def test_manifest_absolute_path_resolution(tmp_path: Path) -> None:
    source = _signal(tmp_path / "绝对路径.csv")
    path = _manifest(
        tmp_path / "manifest.csv",
        ["file_path", "true_combination"],
        [[str(source), "010"]],
    )
    report = ValidationManifestService().inspect(path, LABELS)
    assert report.samples[0].file_path == source.resolve()


@pytest.mark.parametrize("value", ["", "1010", "10x", "000"])
def test_true_combination_validation(tmp_path: Path, value: str) -> None:
    source = _signal(tmp_path / "a.csv")
    path = _manifest(
        tmp_path / "manifest.csv",
        ["file_path", "true_combination"],
        [[str(source), value]],
    )
    report = ValidationManifestService().inspect(path, LABELS)
    assert not report.can_start
    assert report.invalid_label_count == 1


def test_dynamic_label_length(tmp_path: Path) -> None:
    labels = ["a", "b", "c", "d"]
    source = _signal(tmp_path / "a.csv")
    path = _manifest(
        tmp_path / "manifest.csv",
        ["file_path", "true_combination"],
        [[str(source), "1001"]],
    )
    report = ValidationManifestService().inspect(path, labels)
    assert report.can_start
    assert report.samples[0].true_label_vector == [1, 0, 0, 1]


def test_per_label_columns_read_in_model_order(tmp_path: Path) -> None:
    source = _signal(tmp_path / "a.csv")
    path = _manifest(
        tmp_path / "manifest.csv",
        ["file_path", *LABELS, "frequency_mhz"],
        [[str(source), "1", "0", "1", "650"]],
    )
    report = ValidationManifestService().inspect(path, LABELS)
    assert report.can_start
    assert report.samples[0].true_combination == "101"
    assert report.metadata_fields == ["frequency_mhz"]


def test_per_label_columns_reject_wrong_order(tmp_path: Path) -> None:
    source = _signal(tmp_path / "a.csv")
    path = _manifest(
        tmp_path / "manifest.csv",
        ["file_path", "motor", "fan", "switch_power"],
        [[str(source), "0", "1", "1"]],
    )
    report = ValidationManifestService().inspect(path, LABELS)
    assert not report.can_start
    assert any(issue.code == "label_order_unknown" for issue in report.fatal_issues)


def test_combination_and_label_columns_conflict(tmp_path: Path) -> None:
    source = _signal(tmp_path / "a.csv")
    path = _manifest(
        tmp_path / "manifest.csv",
        ["file_path", "true_combination", *LABELS],
        [[str(source), "101", "1", "1", "0"]],
    )
    report = ValidationManifestService().inspect(path, LABELS)
    assert not report.can_start
    assert any(issue.code == "truth_conflict" for issue in report.issues)


def test_duplicate_and_conflicting_duplicate_detection(tmp_path: Path) -> None:
    source = _signal(tmp_path / "a.csv")
    path = _manifest(
        tmp_path / "manifest.csv",
        ["file_path", "true_combination"],
        [[str(source), "100"], [str(source), "010"]],
    )
    report = ValidationManifestService().inspect(path, LABELS)
    assert report.duplicate_count == 1
    assert any(issue.code == "conflicting_duplicate" for issue in report.fatal_issues)


def test_missing_file_stop_and_skip_policies(tmp_path: Path) -> None:
    existing = _signal(tmp_path / "existing.csv")
    missing = tmp_path / "missing.csv"
    path = _manifest(
        tmp_path / "manifest.csv",
        ["file_path", "true_combination"],
        [[str(existing), "100"], [str(missing), "010"]],
    )
    stopped = ValidationManifestService().inspect(path, LABELS, missing_policy="stop")
    skipped = ValidationManifestService().inspect(path, LABELS, missing_policy="skip")
    assert not stopped.can_start
    assert skipped.can_start
    assert skipped.samples[1].status == ValidationSampleStatus.SKIPPED


def _sample(
    sequence: int,
    true_vector: list[int],
    predicted_vector: list[int],
    *,
    confidence: float = 0.8,
    metadata: dict[str, str] | None = None,
) -> ValidationSample:
    true_combination = "".join(map(str, true_vector))
    predicted_combination = "".join(map(str, predicted_vector))
    sample = ValidationSample(
        sequence,
        Path(f"sample-{sequence}.csv"),
        true_vector,
        true_combination,
        metadata=metadata or {},
        predicted_label_vector=predicted_vector,
        predicted_combination=predicted_combination,
        labels=["a", "b"],
        confidence=confidence,
        combination_labels=["01", "10", "11"],
        combination_probabilities=[0.1, 0.8, 0.1],
        elapsed_ms=float(sequence * 10),
        status=ValidationSampleStatus.SUCCESS,
        result={
            "decision_mode": "structured",
            "combination_labels": ["01", "10", "11"],
            "combination_probabilities": [0.1, 0.8, 0.1],
        },
    )
    sample.true_source_count = sum(true_vector)
    sample.predicted_source_count = sum(predicted_vector)
    sample.exact_match = true_vector == predicted_vector
    return sample


def _metric_samples() -> list[ValidationSample]:
    samples = [
        _sample(1, [1, 0], [1, 0], metadata={"frequency_mhz": "600"}),
        _sample(2, [0, 1], [1, 1], metadata={"frequency_mhz": "600"}),
        _sample(3, [1, 1], [0, 1], metadata={"frequency_mhz": "700"}),
    ]
    failed = ValidationSample(
        4,
        Path("failed.csv"),
        [1, 0],
        "10",
        metadata={"frequency_mhz": "700"},
        status=ValidationSampleStatus.INFERENCE_FAILED,
        error_type="PredictionError",
    )
    return [*samples, failed]


def test_exact_micro_macro_and_hamming_metrics() -> None:
    metrics = calculate_validation_metrics(
        _metric_samples(), ["a", "b"], decision_mode="structured"
    )
    overall = metrics["overall"]
    assert overall["exact_match_accuracy"] == pytest.approx(1 / 3)
    assert overall["micro_precision"] == pytest.approx(0.75)
    assert overall["micro_recall"] == pytest.approx(0.75)
    assert overall["micro_f1"] == pytest.approx(0.75)
    assert overall["macro_f1"] == pytest.approx(0.75)
    assert overall["hamming_loss"] == pytest.approx(1 / 3)


def test_per_label_confusion_and_rates() -> None:
    rows = calculate_validation_metrics(_metric_samples(), ["a", "b"], decision_mode="structured")[
        "labels"
    ]
    first = rows[0]
    assert (first["tp"], first["fp"], first["tn"], first["fn"]) == (1, 1, 0, 1)
    assert first["false_positive_rate"] == 1.0
    assert first["false_negative_rate"] == 0.5


def test_combination_confusion_matrix_and_accuracy() -> None:
    metrics = calculate_validation_metrics(
        _metric_samples(), ["a", "b"], decision_mode="structured"
    )
    matrix = metrics["confusion_matrix"]
    assert matrix["labels"] == ["01", "10", "11"]
    assert sum(sum(row) for row in matrix["counts"]) == 3
    combo_10 = next(row for row in metrics["combinations"] if row["true_combination"] == "10")
    assert combo_10["exact_accuracy"] == 1.0


def test_over_under_and_source_count_accuracy() -> None:
    overall = calculate_validation_metrics(
        _metric_samples(), ["a", "b"], decision_mode="structured"
    )["overall"]
    assert overall["overprediction_rate"] == pytest.approx(1 / 3)
    assert overall["underprediction_rate"] == pytest.approx(1 / 3)
    assert overall["source_count_accuracy"] == pytest.approx(1 / 3)


def test_inference_failure_is_excluded_from_accuracy_denominator() -> None:
    metrics = calculate_validation_metrics(
        _metric_samples(), ["a", "b"], decision_mode="structured"
    )
    assert metrics["denominator"]["sample_count"] == 3
    assert metrics["overall"]["inference_failed_count"] == 1
    assert metrics["overall"]["exact_match_accuracy"] == pytest.approx(1 / 3)


def test_group_metrics_are_computed_from_metadata() -> None:
    groups = calculate_validation_metrics(
        _metric_samples(), ["a", "b"], decision_mode="structured"
    )["groups"]["frequency_mhz"]
    assert {row["value"] for row in groups} == {"600", "700"}
    group_600 = next(row for row in groups if row["value"] == "600")
    assert group_600["sample_count"] == 2
    assert group_600["exact_match"] == 0.5


def test_single_sample_and_empty_group_are_safe() -> None:
    metrics = calculate_validation_metrics(
        [_sample(1, [1, 0], [1, 0])],
        ["a", "b"],
        decision_mode="multilabel",
        group_fields=["missing"],
    )
    assert metrics["overall"]["exact_match_accuracy"] == 1.0
    assert metrics["groups"]["missing"] == []
    assert metrics["top_k"] is None


def test_legal_combinations_are_dynamic() -> None:
    assert legal_combinations(3) == ["001", "010", "011", "100", "101", "110", "111"]


def test_structured_top_one_and_top_two_metrics() -> None:
    top_k = calculate_validation_metrics(
        _metric_samples(),
        ["a", "b"],
        decision_mode="structured",
    )["top_k"]
    assert top_k is not None
    assert top_k["top_1_accuracy"] == pytest.approx(1 / 3)
    assert top_k["top_2_accuracy"] == pytest.approx(2 / 3)
