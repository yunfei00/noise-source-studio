"""Validation center models, filters and linked-view tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from noise_source_studio.domain.models import LoadedModel, ModelRecord
from noise_source_studio.domain.validation import (
    ManifestValidationReport,
    ValidationSample,
    ValidationSampleStatus,
    ValidationStatus,
    ValidationTask,
)
from noise_source_studio.presentation.models import (
    ValidationSampleFilterProxyModel,
    ValidationSampleTableModel,
)
from noise_source_studio.presentation.pages.validation_page import ValidationPage
from noise_source_studio.services.validation_metrics import calculate_validation_metrics

LABELS = ["fan", "motor", "switch_power"]


def _model(tmp_path: Path) -> LoadedModel:
    record = ModelRecord(
        "model",
        "1.0",
        tmp_path / "model",
        {"labels": LABELS, "prediction_mode": "structured"},
        "2026-01-01T00:00:00Z",
        True,
        "valid",
    )
    return LoadedModel(record, "1.0.0", "cpu", "structured", tuple(LABELS), {})


def _sample(
    sequence: int,
    true: str = "110",
    predicted: str = "111",
    *,
    status: ValidationSampleStatus = ValidationSampleStatus.SUCCESS,
) -> ValidationSample:
    sample = ValidationSample(
        sequence,
        Path(f"sample-{sequence}.csv"),
        [int(value) for value in true],
        true,
        metadata={"frequency_mhz": "600" if sequence % 2 else "700"},
        predicted_label_vector=[int(value) for value in predicted],
        predicted_combination=predicted,
        predicted_sources=[
            label for label, value in zip(LABELS, predicted, strict=True) if value == "1"
        ],
        labels=list(LABELS),
        display_probabilities=[0.8, 0.7, 0.6],
        multilabel_probabilities=[0.8, 0.7, 0.6],
        label_marginal_probabilities=[0.8, 0.7, 0.6],
        combination_labels=["110", "111", "101"],
        combination_probabilities=[0.35, 0.55, 0.10],
        thresholds=[0.5, 0.5, 0.5],
        exact_match=true == predicted,
        false_positive_labels=[
            label
            for label, expected, actual in zip(LABELS, true, predicted, strict=True)
            if expected == "0" and actual == "1"
        ],
        false_negative_labels=[
            label
            for label, expected, actual in zip(LABELS, true, predicted, strict=True)
            if expected == "1" and actual == "0"
        ],
        confidence=0.55,
        confidence_margin=0.20,
        elapsed_ms=12.0,
        status=status,
        result={
            "decision_mode": "structured",
            "combination_labels": ["110", "111", "101"],
            "combination_probabilities": [0.35, 0.55, 0.10],
        },
    )
    sample.true_source_count = true.count("1")
    sample.predicted_source_count = predicted.count("1")
    return sample


def _task(tmp_path: Path) -> ValidationTask:
    samples = [
        _sample(1, "110", "111"),
        _sample(2, "011", "011"),
        _sample(3, "101", "001"),
    ]
    task = ValidationTask(
        tmp_path / "manifest.csv",
        tmp_path,
        tmp_path / "output",
        created_at=datetime.now(UTC),
        model_name="model",
        model_version="1.0",
        runtime_version="1.0.0",
        device="cpu",
        labels=list(LABELS),
        decision_mode="structured",
        status=ValidationStatus.COMPLETED,
        metadata_fields=["frequency_mhz"],
        samples=samples,
    )
    task.refresh_counts()
    task.metrics = calculate_validation_metrics(
        samples,
        LABELS,
        decision_mode="structured",
        group_fields=["frequency_mhz"],
    )
    return task


def test_validation_page_has_six_tabs(qapp: QApplication, qtbot: QtBot) -> None:
    page = ValidationPage()
    qtbot.addWidget(page)
    assert page.tabs.count() == 6
    assert [page.tabs.tabText(index) for index in range(6)] == [
        "验证配置",
        "总体结果",
        "标签分析",
        "组合分析",
        "分组分析",
        "样本明细",
    ]


def test_start_requires_model_and_checked_manifest(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    page = ValidationPage()
    qtbot.addWidget(page)
    page.manifest_edit.setText(str(tmp_path / "manifest.csv"))
    assert not page.start_button.isEnabled()
    page.set_model(_model(tmp_path))
    manifest_sample = _sample(1)
    manifest_sample.status = ValidationSampleStatus.PENDING
    report = ManifestValidationReport(
        tmp_path / "manifest.csv",
        tmp_path,
        list(LABELS),
        samples=[manifest_sample],
        source_row_count=1,
    )
    page.show_manifest_report(report)
    assert page.start_button.isEnabled()


def test_result_tables_use_dynamic_labels(qapp: QApplication, qtbot: QtBot, tmp_path: Path) -> None:
    page = ValidationPage()
    qtbot.addWidget(page)
    page.set_model(_model(tmp_path))
    page.set_task(_task(tmp_path))
    assert page.label_model.rowCount() == 3
    assert page.sample_model.rowCount() == 3
    assert page.matrix_model.rowCount() == 7


def test_sample_filters_cover_combinations_labels_and_metadata(tmp_path: Path) -> None:
    task = _task(tmp_path)
    model = ValidationSampleTableModel()
    proxy = ValidationSampleFilterProxyModel()
    proxy.setSourceModel(model)
    model.set_task(task)
    proxy.set_filters(true_combination="110")
    assert proxy.rowCount() == 1
    proxy.set_filters(true_combination="", label="switch_power", label_relation="false_positive")
    assert proxy.rowCount() == 1
    proxy.set_filters(
        label="",
        label_relation="",
        metadata_field="frequency_mhz",
        metadata_value="700",
    )
    assert proxy.rowCount() == 1


def test_typical_misclassification_filters(tmp_path: Path) -> None:
    task = _task(tmp_path)
    model = ValidationSampleTableModel()
    proxy = ValidationSampleFilterProxyModel()
    proxy.setSourceModel(model)
    model.set_task(task)
    proxy.special = "double_to_triple"
    proxy.set_filters(outcome="all")
    assert proxy.rowCount() == 1


def test_matrix_click_links_to_sample_filters(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    page = ValidationPage()
    qtbot.addWidget(page)
    page.set_task(_task(tmp_path))
    true_index = page.matrix_model.labels.index("110")
    predicted_index = page.matrix_model.labels.index("111")
    page._matrix_clicked(page.matrix_model.index(true_index, predicted_index))
    assert page.tabs.currentIndex() == 5
    assert page.sample_proxy.true_combination == "110"
    assert page.sample_proxy.predicted_combination == "111"
    assert page.sample_proxy.rowCount() == 1


def test_chart_public_link_applies_filter(qapp: QApplication, qtbot: QtBot, tmp_path: Path) -> None:
    page = ValidationPage()
    qtbot.addWidget(page)
    page.set_task(_task(tmp_path))
    page.apply_chart_filter("combination", "011")
    assert page.tabs.currentIndex() == 5
    assert page.sample_proxy.true_combination == "011"


def test_confidence_and_error_chart_links_apply_specific_filters(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    page = ValidationPage()
    qtbot.addWidget(page)
    task = _task(tmp_path)
    task.samples[0].status = ValidationSampleStatus.INFERENCE_FAILED
    task.samples[0].error_type = "PredictionError"
    task.samples[0].exact_match = None
    page.set_task(task)
    page.apply_chart_filter("confidence", "40–60%")
    assert page.sample_proxy.confidence_min == 0.4
    assert page.sample_proxy.confidence_max == 0.6
    page.apply_chart_filter("error", "PredictionError")
    assert page.sample_proxy.error_type == "PredictionError"
    assert page.sample_proxy.rowCount() == 1


def test_selected_sample_populates_dynamic_detail(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    page = ValidationPage()
    qtbot.addWidget(page)
    task = _task(tmp_path)
    page.set_task(task)
    page._sample_selected(page.sample_proxy.index(0, 0))
    assert "true_label_vector" in page.sample_detail.toPlainText()
    assert len(page._probability_widgets) == len(LABELS)


def test_ten_thousand_validation_rows_load_quickly(tmp_path: Path) -> None:
    prototype = _sample(1)
    task = ValidationTask(tmp_path / "manifest.csv", tmp_path, tmp_path / "out")
    task.samples = [
        ValidationSample(
            index,
            prototype.file_path,
            list(prototype.true_label_vector),
            prototype.true_combination,
            predicted_label_vector=list(prototype.predicted_label_vector),
            predicted_combination=prototype.predicted_combination,
            labels=list(prototype.labels),
            status=ValidationSampleStatus.SUCCESS,
        )
        for index in range(1, 10_001)
    ]
    started = perf_counter()
    model = ValidationSampleTableModel()
    proxy = ValidationSampleFilterProxyModel()
    proxy.setSourceModel(model)
    model.set_task(task)
    elapsed = perf_counter() - started
    assert model.rowCount() == 10_000
    assert proxy.rowCount() == 10_000
    assert elapsed < 1.0
