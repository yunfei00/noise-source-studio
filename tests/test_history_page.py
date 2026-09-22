"""History page model, filters, pagination, and action wiring."""

from __future__ import annotations

from datetime import UTC, datetime

from PySide6.QtCore import Qt

from noise_source_studio.domain.history import (
    HistoryOverview,
    HistoryPageResult,
    HistoryStatus,
    IntegrityStatus,
    TaskHistoryRecord,
    TaskType,
)
from noise_source_studio.presentation.pages.history_page import HistoryPage


def _record(index: int = 1) -> TaskHistoryRecord:
    return TaskHistoryRecord(
        task_id=f"history-{index}",
        task_type=TaskType.BATCH if index % 2 else TaskType.SINGLE,
        task_name=f"验收任务 {index}",
        status=(
            HistoryStatus.COMPLETED if index % 2 else HistoryStatus.COMPLETED_WITH_ERRORS
        ),
        created_at=datetime(2026, 8, index, tzinfo=UTC).isoformat(),
        finished_at=datetime(2026, 8, index, 0, 1, tzinfo=UTC).isoformat(),
        model_name="noise-net",
        model_version="2.1",
        model_identifier="noise-net@2.1",
        device="cpu",
        total_count=10,
        success_count=9,
        failed_count=1,
        primary_summary="风机 + 泵 78.20%",
        result_directory=f"C:/results/{index}",
        integrity_status=IntegrityStatus.OK,
        tags=("review",),
        notes="需要人工复核",
    )


def test_table_loads_records_and_overview(qtbot) -> None:  # type: ignore[no-untyped-def]
    page = HistoryPage()
    qtbot.addWidget(page)
    records = (_record(1), _record(2))
    page.show_results(
        HistoryPageResult(records, total=2, page=1, page_size=50),
        HistoryOverview(total=2, completed=2, failed=0, missing_artifacts=1),
    )
    assert page.table_model.rowCount() == 2
    assert page.overview_values[0].text() == "2"
    assert page.overview_values[5].text() == "1"
    assert "共 2 条" in page.page_label.text()


def test_search_is_debounced_for_250_ms(qtbot) -> None:  # type: ignore[no-untyped-def]
    page = HistoryPage()
    qtbot.addWidget(page)
    with qtbot.waitSignal(page.query_requested, timeout=600) as blocker:
        page.keyword_edit.setText("sample.csv")
    assert blocker.args[0].keyword == "sample.csv"


def test_task_type_status_and_failure_filters(qtbot) -> None:  # type: ignore[no-untyped-def]
    page = HistoryPage()
    qtbot.addWidget(page)
    page.type_filter.setCurrentIndex(2)
    page.status_filter.setCurrentIndex(5)
    page.failed_only.setChecked(True)
    query = page.current_query()
    assert query.task_type == TaskType.BATCH
    assert query.status == HistoryStatus.FAILED
    assert query.failed_only


def test_date_filter_is_opt_in(qtbot) -> None:  # type: ignore[no-untyped-def]
    page = HistoryPage()
    qtbot.addWidget(page)
    assert page.current_query().date_from == ""
    page.date_filter_enabled.setChecked(True)
    assert page.current_query().date_from.endswith("T00:00:00")
    assert page.current_query().date_to.endswith("T23:59:59.999999")


def test_model_version_and_device_filters(qtbot) -> None:  # type: ignore[no-untyped-def]
    page = HistoryPage()
    qtbot.addWidget(page)
    page.model_filter.setText("noise-net")
    page.version_filter.setText("2.1")
    page.device_filter.setText("cuda:0")
    query = page.current_query()
    assert query.model_name == "noise-net"
    assert query.model_version == "2.1"
    assert query.device == "cuda:0"


def test_selection_populates_complete_detail_and_actions(qtbot) -> None:  # type: ignore[no-untyped-def]
    page = HistoryPage()
    qtbot.addWidget(page)
    page.show_results(
        HistoryPageResult((_record(),), total=1, page=1, page_size=50),
        HistoryOverview(total=1),
    )
    page.table.selectRow(0)
    qtbot.waitUntil(lambda: page.selected_record() is not None)
    assert '"task_id": "history-1"' in page.detail_view.toPlainText()
    assert page.open_button.isEnabled()
    assert page.delete_files_button.isEnabled()


def test_open_verify_and_delete_actions_emit_selected_id(qtbot) -> None:  # type: ignore[no-untyped-def]
    page = HistoryPage()
    qtbot.addWidget(page)
    page.show_results(
        HistoryPageResult((_record(),), total=1, page=1, page_size=50),
        HistoryOverview(total=1),
    )
    page.table.selectRow(0)
    with qtbot.waitSignal(page.open_task_requested) as opened:
        qtbot.mouseClick(page.open_button, Qt.MouseButton.LeftButton)
    with qtbot.waitSignal(page.verify_requested) as verified:
        qtbot.mouseClick(page.verify_button, Qt.MouseButton.LeftButton)
    with qtbot.waitSignal(page.delete_requested) as deleted:
        qtbot.mouseClick(page.delete_index_button, Qt.MouseButton.LeftButton)
    assert opened.args == ["history-1"]
    assert verified.args == ["history-1"]
    assert deleted.args == ["history-1", False]


def test_pagination_requests_next_database_page(qtbot) -> None:  # type: ignore[no-untyped-def]
    page = HistoryPage()
    qtbot.addWidget(page)
    page.show_results(
        HistoryPageResult((_record(),), total=60, page=1, page_size=20),
        HistoryOverview(total=60),
    )
    with qtbot.waitSignal(page.query_requested) as blocker:
        qtbot.mouseClick(page.next_button, Qt.MouseButton.LeftButton)
    assert blocker.args[0].page == 2
    assert blocker.args[0].page_size == 50


def test_first_and_last_page_controls(qtbot) -> None:  # type: ignore[no-untyped-def]
    page = HistoryPage()
    qtbot.addWidget(page)
    page.show_results(
        HistoryPageResult((_record(),), total=120, page=2, page_size=50),
        HistoryOverview(total=120),
    )
    with qtbot.waitSignal(page.query_requested) as last:
        qtbot.mouseClick(page.last_button, Qt.MouseButton.LeftButton)
    assert last.args[0].page == 3
    page.show_results(
        HistoryPageResult((_record(),), total=120, page=3, page_size=50),
        HistoryOverview(total=120),
    )
    with qtbot.waitSignal(page.query_requested) as first:
        qtbot.mouseClick(page.first_button, Qt.MouseButton.LeftButton)
    assert first.args[0].page == 1


def test_sort_requests_database_order(qtbot) -> None:  # type: ignore[no-untyped-def]
    page = HistoryPage()
    qtbot.addWidget(page)
    with qtbot.waitSignal(page.query_requested) as blocker:
        page.table_model.sort(6, Qt.SortOrder.DescendingOrder)
    assert blocker.args[0].sort_by == "total_count"
    assert blocker.args[0].descending


def test_unavailable_state_disables_history_only(qtbot) -> None:  # type: ignore[no-untyped-def]
    page = HistoryPage()
    qtbot.addWidget(page)
    page.set_unavailable("database locked")
    assert not page.table.isEnabled()
    assert not page.scan_button.isEnabled()
    assert "database locked" in page.loading_label.text()
