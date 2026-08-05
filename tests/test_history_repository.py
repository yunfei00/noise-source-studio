"""SQLite history schema, query, lifecycle, and scale tests."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter

import pytest

from noise_source_studio.domain.history import (
    HistoryArtifact,
    HistoryQuery,
    HistoryStatus,
    IntegrityStatus,
    TaskHistoryRecord,
    TaskType,
)
from noise_source_studio.infrastructure.history import (
    HistoryDatabase,
    HistoryRepository,
    HistorySchemaTooNewError,
)


@pytest.fixture
def repository(tmp_path: Path) -> HistoryRepository:
    value = HistoryRepository(HistoryDatabase(tmp_path / "history" / "history.db"))
    value.initialize()
    return value


def _record(
    index: int = 1,
    *,
    task_type: TaskType = TaskType.SINGLE,
    status: HistoryStatus = HistoryStatus.COMPLETED,
) -> TaskHistoryRecord:
    created = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
    return TaskHistoryRecord(
        task_id=f"task-{index:05d}",
        task_type=task_type,
        task_name=f"任务 {index}",
        status=status,
        created_at=created.isoformat(),
        finished_at=(created + timedelta(seconds=2)).isoformat(),
        duration_ms=2000.0,
        model_name="baseline" if index % 2 else "candidate",
        model_version="1.0",
        model_identifier="baseline@1.0",
        device="cpu" if index % 2 else "cuda:0",
        total_count=index,
        success_count=max(0, index - 1),
        failed_count=1 if status == HistoryStatus.FAILED else 0,
        primary_summary=f"signal-{index}.csv",
        source_description=f"signal-{index}.csv",
        integrity_status=(
            IntegrityStatus.MISSING if index % 7 == 0 else IntegrityStatus.OK
        ),
        tags=("review",) if index % 3 == 0 else (),
    )


def test_schema_initializes_at_version_one(repository: HistoryRepository) -> None:
    with repository.database.connect() as connection:
        version = connection.execute(
            "SELECT value FROM schema_metadata WHERE key='schema_version'"
        ).fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert version == "1"
    assert {"history_tasks", "history_artifacts", "schema_metadata"} <= tables


def test_newer_schema_is_rejected(tmp_path: Path) -> None:
    database = HistoryDatabase(tmp_path / "history.db")
    database.initialize()
    with database.connect() as connection:
        connection.execute(
            "UPDATE schema_metadata SET value='999' WHERE key='schema_version'"
        )
    with pytest.raises(HistorySchemaTooNewError):
        database.initialize()


def test_connection_pragmas_are_enabled(repository: HistoryRepository) -> None:
    with repository.database.connect() as connection:
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
    assert foreign_keys == 1
    assert journal_mode.casefold() == "wal"
    assert busy_timeout == 5000


def test_task_and_artifact_round_trip(repository: HistoryRepository) -> None:
    record = _record()
    artifact = HistoryArtifact("prediction", "single/result.json", file_size=12)
    repository.upsert_task(record, (artifact,))
    restored = repository.get_task(record.task_id)
    assert restored == record
    assert repository.artifacts(record.task_id)[0].path == "single/result.json"


def test_upsert_preserves_one_task(repository: HistoryRepository) -> None:
    record = _record()
    repository.upsert_task(record)
    repository.upsert_task(replace(record, primary_summary="updated"))
    assert repository.list_tasks(HistoryQuery()).total == 1
    assert repository.get_task(record.task_id).primary_summary == "updated"  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (HistoryQuery(task_type=TaskType.BATCH), {"task-00002"}),
        (HistoryQuery(status=HistoryStatus.FAILED), {"task-00003"}),
        (HistoryQuery(model_name="candidate"), {"task-00002"}),
        (HistoryQuery(model_version="1.0"), {"task-00001", "task-00002", "task-00003"}),
        (HistoryQuery(device="cpu"), {"task-00001", "task-00003"}),
        (HistoryQuery(failed_only=True), {"task-00003"}),
        (HistoryQuery(missing_only=True), set()),
        (HistoryQuery(keyword="signal-2"), {"task-00002"}),
        (HistoryQuery(keyword="review"), {"task-00003"}),
    ],
)
def test_database_filters(
    repository: HistoryRepository,
    query: HistoryQuery,
    expected: set[str],
) -> None:
    repository.upsert_many(
        (
            _record(1),
            _record(2, task_type=TaskType.BATCH),
            _record(3, task_type=TaskType.VALIDATION, status=HistoryStatus.FAILED),
        )
    )
    records = repository.list_tasks(query).records
    assert {record.task_id for record in records} == expected


def test_date_filter(repository: HistoryRepository) -> None:
    repository.upsert_many((_record(1), _record(2), _record(3)))
    query = HistoryQuery(
        date_from="2026-01-01T00:02:00+00:00",
        date_to="2026-01-01T00:02:59+00:00",
    )
    assert [record.task_id for record in repository.list_tasks(query).records] == [
        "task-00002"
    ]


def test_pagination_and_sort(repository: HistoryRepository) -> None:
    repository.upsert_many(_record(index) for index in range(1, 26))
    result = repository.list_tasks(
        HistoryQuery(page=2, page_size=10, sort_by="total_count", descending=False)
    )
    assert result.total == 25
    assert result.page_count == 3
    assert [record.total_count for record in result.records] == list(range(11, 21))


def test_update_notes_and_status(repository: HistoryRepository) -> None:
    repository.upsert_task(_record())
    assert repository.update_task(
        "task-00001",
        notes="需要复核",
        tags=("important", "客户 A"),
        status=HistoryStatus.FAILED,
    )
    record = repository.get_task("task-00001")
    assert record is not None
    assert record.notes == "需要复核"
    assert record.tags == ("important", "客户 A")
    assert record.status == HistoryStatus.FAILED


def test_invalid_update_field_is_rejected(repository: HistoryRepository) -> None:
    repository.upsert_task(_record())
    with pytest.raises(ValueError):
        repository.update_task("task-00001", sql_injection="DROP TABLE history_tasks")


def test_delete_cascades_artifacts(repository: HistoryRepository) -> None:
    repository.upsert_task(_record(), (HistoryArtifact("result", "result.json"),))
    assert repository.delete_task("task-00001")
    assert repository.get_task("task-00001") is None
    assert repository.artifacts("task-00001") == ()


def test_running_tasks_become_interrupted(repository: HistoryRepository) -> None:
    repository.upsert_many(
        (
            _record(1, status=HistoryStatus.RUNNING),
            _record(2, status=HistoryStatus.COMPLETED),
        )
    )
    assert repository.interrupt_running("2026-02-01T00:00:00+00:00") == 1
    assert repository.get_task("task-00001").status == HistoryStatus.INTERRUPTED  # type: ignore[union-attr]
    assert repository.get_task("task-00002").status == HistoryStatus.COMPLETED  # type: ignore[union-attr]


def test_overview_counts(repository: HistoryRepository) -> None:
    now = datetime.now(UTC).isoformat()
    repository.upsert_many(
        (
            replace(_record(1), created_at=now),
            replace(_record(2, status=HistoryStatus.FAILED), created_at=now),
            replace(
                _record(7, status=HistoryStatus.INTERRUPTED),
                created_at=now,
                integrity_status=IntegrityStatus.MISSING,
            ),
        )
    )
    overview = repository.overview()
    assert (overview.total, overview.completed, overview.failed) == (3, 1, 1)
    assert overview.interrupted == 1
    assert overview.last_seven_days == 3
    assert overview.missing_artifacts == 1


def test_database_backup_is_readable(repository: HistoryRepository) -> None:
    repository.upsert_task(_record())
    backup = repository.database.backup("test")
    with sqlite3.connect(backup) as connection:
        assert connection.execute("SELECT COUNT(*) FROM history_tasks").fetchone()[0] == 1


def test_concurrent_short_writes(repository: HistoryRepository) -> None:
    records = [_record(index) for index in range(1, 41)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(repository.upsert_task, records))
    assert repository.list_tasks(HistoryQuery(page_size=50)).total == 40


def test_ten_thousand_rows_remain_queryable(repository: HistoryRepository) -> None:
    repository.upsert_many(_record(index) for index in range(1, 10_001))
    started = perf_counter()
    result = repository.list_tasks(
        HistoryQuery(keyword="signal-9999", page_size=20, sort_by="task_id")
    )
    elapsed = perf_counter() - started
    assert result.total == 1
    assert result.records[0].task_id == "task-09999"
    assert elapsed < 2.0
