"""Batch result center filtering, history, detail and performance tests."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication, QProgressBar, QTableWidget
from pytestqt.qtbot import QtBot

from noise_source_studio.domain.batch import (
    BatchFileItem,
    BatchItemStatus,
    BatchPredictionTask,
    BatchStatus,
)
from noise_source_studio.domain.confidence import (
    candidate_probability_margin,
    is_low_confidence,
)
from noise_source_studio.presentation.models import (
    BatchResultFilterProxyModel,
    BatchResultTableModel,
)
from noise_source_studio.presentation.pages import BatchPredictionPage
from noise_source_studio.services import BatchPredictionService


def _payload(
    *,
    mode: str = "structured",
    top: float = 0.75,
    second: float = 0.15,
    labels: list[str] | None = None,
) -> dict:
    labels = labels or ["fan", "motor", "switch_power"]
    return {
        "labels": labels,
        "decision_mode": mode,
        "multilabel_probabilities": [0.82, 0.48, 0.71][: len(labels)],
        "label_marginal_probabilities": [0.79, 0.36, 0.68][: len(labels)],
        "combination_labels": ["001", "010", "011", "100", "101", "110", "111"],
        "combination_probabilities": [
            top,
            second,
            0.03,
            0.02,
            0.02,
            0.02,
            max(0.0, 1.0 - top - second - 0.09),
        ],
        "decoded_label_vector": [1, 0, 1][: len(labels)],
        "predicted_combination": "101",
        "predicted_sources": [labels[0], labels[-1]],
        "thresholds": [0.5] * len(labels),
        "thresholds_applicable": mode == "multilabel",
        "input_shape": [1, 1, 128, 64],
    }


def _item(
    tmp_path: Path,
    sequence: int,
    *,
    combination: str = "101",
    sources: list[str] | None = None,
    status: BatchItemStatus = BatchItemStatus.SUCCESS,
    top: float = 0.75,
    second: float = 0.15,
    labels: list[str] | None = None,
) -> BatchFileItem:
    path = tmp_path / f"sample_{sequence}.csv"
    path.write_text("DATA\n0,1\n", encoding="utf-8")
    payload = _payload(top=top, second=second, labels=labels)
    payload["predicted_combination"] = combination
    payload["predicted_sources"] = sources or ["fan", "switch_power"]
    item = BatchFileItem(sequence, path)
    item.status = status
    item.status_message = status.value
    item.elapsed_ms = 100.0 + sequence
    item.labels = list(payload["labels"])
    item.decision_mode = str(payload["decision_mode"])
    item.display_probabilities = list(payload["label_marginal_probabilities"])
    item.predicted_combination = combination
    item.predicted_sources = list(payload["predicted_sources"])
    item.decoded_label_vector = list(payload["decoded_label_vector"])
    item.result = payload
    if status == BatchItemStatus.FAILED:
        item.error_type = "CsvContractError"
        item.error_message = "CSV 缺少 DATA"
        item.predicted_sources = []
        item.result = None
    return item


def _task(tmp_path: Path) -> BatchPredictionTask:
    task = BatchPredictionTask("result-center", tmp_path / "outputs")
    task.model_name = "model"
    task.model_version = "1.0.0"
    task.runtime_version = "1.0.0"
    task.device = "cpu"
    task.status = BatchStatus.COMPLETED_WITH_ERRORS
    task.started_at = datetime.now(UTC)
    task.finished_at = task.started_at
    task.items = [
        _item(tmp_path, 1, combination="101", sources=["fan"], top=0.82, second=0.10),
        _item(
            tmp_path,
            2,
            combination="110",
            sources=["fan", "motor"],
            top=0.55,
            second=0.45,
        ),
        _item(tmp_path, 3, status=BatchItemStatus.FAILED),
    ]
    task.refresh_counts()
    return task


def _models(task: BatchPredictionTask) -> tuple[BatchResultTableModel, BatchResultFilterProxyModel]:
    model = BatchResultTableModel()
    model.set_task(task)
    proxy = BatchResultFilterProxyModel()
    proxy.setSourceModel(model)
    return model, proxy


def test_result_table_loads_all_items(tmp_path: Path) -> None:
    model, proxy = _models(_task(tmp_path))
    assert model.rowCount() == 3
    assert proxy.rowCount() == 3
    assert model.columnCount() == 10


def test_result_table_supports_dynamic_labels(tmp_path: Path) -> None:
    task = _task(tmp_path)
    task.items[0] = _item(tmp_path, 10, labels=["a", "b", "c", "d", "e"])
    model, _ = _models(task)
    item = model.item_at(0)
    assert item is not None
    assert len(item.labels) == 5


def test_combination_filter(tmp_path: Path) -> None:
    _, proxy = _models(_task(tmp_path))
    proxy.set_filters(combination="110")
    assert [item.sequence for item in proxy.filtered_items()] == [2]


def test_noise_source_multi_filter_uses_any_selected_source(tmp_path: Path) -> None:
    _, proxy = _models(_task(tmp_path))
    proxy.set_filters(sources={"motor", "switch_power"})
    assert [item.sequence for item in proxy.filtered_items()] == [2]


def test_low_confidence_filter_uses_central_policy(tmp_path: Path) -> None:
    task = _task(tmp_path)
    assert not is_low_confidence(task.items[0].result or {})
    assert is_low_confidence(task.items[1].result or {})
    _, proxy = _models(task)
    proxy.set_filters(low_confidence_only=True)
    assert [item.sequence for item in proxy.filtered_items()] == [2]


def test_multilabel_low_confidence_uses_threshold_distance() -> None:
    payload = _payload(mode="multilabel")
    payload["multilabel_probabilities"] = [0.58, 0.8, 0.2]
    payload["thresholds"] = [0.5, 0.5, 0.5]
    assert is_low_confidence(payload)
    payload["multilabel_probabilities"] = [0.7, 0.8, 0.2]
    assert not is_low_confidence(payload)


def test_confidence_interval_filter(tmp_path: Path) -> None:
    _, proxy = _models(_task(tmp_path))
    proxy.set_filters(confidence_min=0.5, confidence_max=0.6)
    assert [item.sequence for item in proxy.filtered_items()] == [2, 3]


def test_error_filter(tmp_path: Path) -> None:
    _, proxy = _models(_task(tmp_path))
    proxy.set_filters(errors_only=True)
    assert [item.sequence for item in proxy.filtered_items()] == [3]


def test_result_sorting_uses_numeric_probability(tmp_path: Path) -> None:
    _, proxy = _models(_task(tmp_path))
    proxy.sort(6, Qt.SortOrder.DescendingOrder)
    assert proxy.filtered_items()[0].sequence == 1


def test_selected_result_detail_uses_structured_marginals(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    page = BatchPredictionPage()
    qtbot.addWidget(page)
    page.set_task(_task(tmp_path))
    page.tabs.setCurrentIndex(1)
    page.result_table.selectRow(0)
    page._show_selected_result_detail()
    bars = page.probability_widget.findChildren(QProgressBar)
    assert [bar.value() for bar in bars] == [7900, 3600, 6800]
    assert '"combination_probabilities"' in page.result_metadata.toPlainText()
    assert '"thresholds_applicable"' in page.result_metadata.toPlainText()


def test_candidate_margin_is_exposed() -> None:
    assert candidate_probability_margin(_payload(top=0.55, second=0.45)) == pytest.approx(0.1)


def test_history_result_round_trip(tmp_path: Path) -> None:
    service = BatchPredictionService(tmp_path / "outputs")
    original = _task(tmp_path)
    exported = service.export_results(original)
    restored = service.load_history(exported.output_directory)
    assert exported.task_path.is_file()
    assert restored.task_id == original.task_id
    assert len(restored.items) == 3
    assert restored.items[0].result == original.items[0].result
    assert restored.items[2].error_type == "CsvContractError"


def test_statistics_chart_link_applies_result_filter(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path
) -> None:
    page = BatchPredictionPage()
    qtbot.addWidget(page)
    page.set_task(_task(tmp_path))
    page.apply_statistics_filter("combination", "110")
    assert page.tabs.currentIndex() == 1
    assert [item.sequence for item in page.result_proxy.filtered_items()] == [2]


def test_ten_thousand_result_model_load_performance(tmp_path: Path) -> None:
    source = tmp_path / "shared.csv"
    source.write_text("DATA\n0,1\n", encoding="utf-8")
    task = BatchPredictionTask("large", tmp_path)
    payload = _payload()
    task.items = []
    for sequence in range(1, 10_001):
        item = BatchFileItem(sequence, source)
        item.status = BatchItemStatus.SUCCESS
        item.labels = list(payload["labels"])
        item.display_probabilities = list(payload["label_marginal_probabilities"])
        item.predicted_combination = "101"
        item.predicted_sources = ["fan"]
        item.result = payload
        task.items.append(item)
    started = perf_counter()
    model, proxy = _models(task)
    elapsed = perf_counter() - started
    assert model.rowCount() == 10_000
    assert proxy.rowCount() == 10_000
    assert elapsed < 1.0


def test_export_current_filtered_results(tmp_path: Path) -> None:
    service = BatchPredictionService(tmp_path / "outputs")
    task = _task(tmp_path)
    _, proxy = _models(task)
    proxy.set_filters(combination="110")
    destination = tmp_path / "筛选结果.csv"
    service.export_filtered_results(task, proxy.filtered_items(), destination)
    with destination.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["sequence"] == "2"
    assert json.loads(rows[0]["predicted_sources"]) == ["fan", "motor"]


def test_compare_two_successful_results(qapp: QApplication, qtbot: QtBot, tmp_path: Path) -> None:
    page = BatchPredictionPage()
    qtbot.addWidget(page)
    page.set_task(_task(tmp_path))
    selection = page.result_table.selectionModel()
    for row in (0, 1):
        selection.select(
            page.result_proxy.index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )
    page._compare_selected_results()
    assert page.last_comparison_dialog is not None
    table = page.last_comparison_dialog.findChild(QTableWidget)
    assert table is not None
    assert table.columnCount() == 2
