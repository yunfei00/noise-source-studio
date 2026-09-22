"""Incremental batch-page update and throttling regression tests."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTableView
from pytest import MonkeyPatch
from pytestqt.qtbot import QtBot

from noise_source_studio.domain.batch import (
    BatchFileItem,
    BatchItemStatus,
    BatchPredictionTask,
    BatchStatus,
)
from noise_source_studio.presentation.pages import BatchPredictionPage


def _task(tmp_path: Path, count: int) -> BatchPredictionTask:
    source = tmp_path / "shared.csv"
    source.write_text("DATA\n0,1\n", encoding="utf-8")
    task = BatchPredictionTask("ui-performance", tmp_path / "outputs")
    task.items = []
    for sequence in range(1, count + 1):
        item = BatchFileItem(sequence, source)
        item.file_name = f"sample_{sequence}.csv"
        task.items.append(item)
    task.refresh_counts()
    return task


def _complete(task: BatchPredictionTask, item: BatchFileItem) -> None:
    payload = {
        "labels": ["fan", "motor"],
        "decision_mode": "structured",
        "label_marginal_probabilities": [0.8, 0.3],
        "combination_labels": ["01", "10", "11"],
        "combination_probabilities": [0.1, 0.8, 0.1],
        "predicted_combination": "10",
        "predicted_sources": ["fan"],
    }
    task.transition_item_status(item, BatchItemStatus.SUCCESS)
    item.status_message = "成功"
    item.elapsed_ms = 12.0
    item.labels = ["fan", "motor"]
    item.display_probabilities = [0.8, 0.3]
    item.predicted_combination = "10"
    item.predicted_sources = ["fan"]
    item.result = payload


def _page(
    qapp: QApplication,
    qtbot: QtBot,
    task: BatchPredictionTask,
) -> BatchPredictionPage:
    page = BatchPredictionPage()
    qtbot.addWidget(page)
    page.set_task(task)
    return page


def test_refresh_item_never_calls_full_refresh(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    task = _task(tmp_path, 1)
    page = _page(qapp, qtbot, task)
    monkeypatch.setattr(page, "refresh_table", lambda: (_ for _ in ()).throw(AssertionError()))
    page.refresh_item(task.items[0])
    page.flush_pending_ui_updates()


def test_single_item_update_preserves_other_rows(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, 3)
    page = _page(qapp, qtbot, task)
    untouched = page.task_model.item_at(0)
    _complete(task, task.items[1])
    page.update_item_row(task.items[1])
    assert page.task_model.item_at(0) is untouched
    assert page.task_model.data(
        page.task_model.index(1, 3),
        Qt.ItemDataRole.DisplayRole,
    ) == "成功"
    assert page.item_row_update_count == 1


def test_high_frequency_states_are_coalesced(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, 1)
    page = _page(qapp, qtbot, task)
    page.reset_performance_counters()
    for status in (
        BatchItemStatus.VALIDATING,
        BatchItemStatus.RUNNING,
        BatchItemStatus.SUCCESS,
    ):
        task.items[0].status = status
        page.refresh_item(task.items[0])
    assert page.dirty_item_ids == {task.items[0].item_id}
    assert page.item_row_update_count == 0
    page.flush_pending_ui_updates()
    assert page.item_row_update_count == 1


def test_running_items_do_not_redraw_charts(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, 100)
    page = _page(qapp, qtbot, task)
    task.status = BatchStatus.RUNNING
    page.begin_batch_updates()
    for item in task.items:
        _complete(task, item)
        page.refresh_item(item)
    page.flush_pending_ui_updates()
    page._refresh_dirty_ui()
    assert page.chart_refresh_count == 0


def test_item_update_never_auto_resizes_columns(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    task = _task(tmp_path, 1)
    page = _page(qapp, qtbot, task)

    def fail_resize(_table: QTableView) -> None:
        raise AssertionError("item update must keep fixed column widths")

    monkeypatch.setattr(QTableView, "resizeColumnsToContents", fail_resize)
    _complete(task, task.items[0])
    page.update_item_row(task.items[0])


def test_result_filter_options_are_maintained_incrementally(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, 1)
    page = _page(qapp, qtbot, task)
    _complete(task, task.items[0])
    page.refresh_item(task.items[0])
    page.flush_pending_ui_updates()
    assert page.combination_filter.findData("10") >= 0
    assert [page.source_filter.item(row).text() for row in range(page.source_filter.count())] == [
        "fan"
    ]


def test_final_calibration_preserves_results_and_statistics(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, 100)
    page = _page(qapp, qtbot, task)
    task.status = BatchStatus.RUNNING
    page.begin_batch_updates()
    for item in task.items:
        _complete(task, item)
        page.refresh_item(item)
    task.status = BatchStatus.COMPLETED
    page.finalize_batch_updates()
    assert task.success_count == 100
    assert page.result_model.rowCount() == 100
    assert all(page.result_model.item_at(row) is task.items[row] for row in range(100))
    assert page.analysis_metrics["success"].text() == "成功数\n100"


def test_pause_resume_stop_states_refresh_without_rebuild(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, 3)
    page = _page(qapp, qtbot, task)
    page.reset_performance_counters()
    task.status = BatchStatus.PAUSED
    page.refresh_summary()
    page.refresh_actions()
    assert page.state_label.text() == "已暂停"
    assert page.resume_button.isEnabled()
    task.status = BatchStatus.RUNNING
    page.refresh_summary()
    page.refresh_actions()
    assert page.state_label.text() == "批量推理中"
    task.status = BatchStatus.STOPPED
    for item in task.items:
        item.status = BatchItemStatus.STOPPED
    task.refresh_counts()
    page.flush_pending_ui_updates()
    page.refresh_summary()
    page.refresh_actions()
    assert page.state_label.text() == "已停止"
    assert page.full_table_rebuild_count == 0


def test_ten_thousand_task_rows_load_and_filter(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, 10_000)
    started = perf_counter()
    page = _page(qapp, qtbot, task)
    elapsed = perf_counter() - started
    page.search_edit.setText("sample_9999.csv")
    assert page.task_model.rowCount() == 10_000
    assert page.task_proxy.rowCount() == 1
    assert elapsed < 15.0


def test_thousand_item_run_keeps_rebuilds_and_charts_bounded(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, 1_000)
    page = _page(qapp, qtbot, task)
    task.status = BatchStatus.RUNNING
    page.begin_batch_updates()
    for item in task.items:
        _complete(task, item)
        page.refresh_item(item)
    page.flush_pending_ui_updates()
    task.status = BatchStatus.COMPLETED
    page.finalize_batch_updates()
    counters = page.performance_counters()
    assert counters["full_table_rebuild_count"] <= 3
    assert counters["item_row_update_count"] == 1_000
    assert counters["chart_refresh_count"] <= 2
    assert task.success_count == 1_000
