"""Generate synthetic history indexes and report paginated query timings."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter

from noise_source_studio.domain.history import (
    HistoryQuery,
    HistoryStatus,
    IntegrityStatus,
    TaskHistoryRecord,
    TaskType,
)
from noise_source_studio.services.history_service import HistoryService


def _records(count: int) -> list[TaskHistoryRecord]:
    started = datetime(2026, 1, 1, tzinfo=UTC)
    task_types = tuple(TaskType)
    statuses = (
        HistoryStatus.COMPLETED,
        HistoryStatus.COMPLETED_WITH_ERRORS,
        HistoryStatus.FAILED,
        HistoryStatus.STOPPED,
    )
    return [
        TaskHistoryRecord(
            task_id=f"benchmark-{index:06d}",
            task_type=task_types[index % len(task_types)],
            task_name=f"历史性能任务 {index}",
            status=statuses[index % len(statuses)],
            created_at=(started + timedelta(seconds=index)).isoformat(),
            finished_at=(started + timedelta(seconds=index + 1)).isoformat(),
            duration_ms=15.0 + index % 300,
            model_name=f"model-{index % 5}",
            model_version=f"{1 + index % 3}.0",
            device="cuda:0" if index % 2 else "cpu",
            total_count=1 + index % 1000,
            success_count=index % 100,
            failed_count=1 if index % 11 == 0 else 0,
            primary_summary=f"模拟结果 {index}",
            integrity_status=(
                IntegrityStatus.MISSING if index % 97 == 0 else IntegrityStatus.OK
            ),
        )
        for index in range(count)
    ]


def benchmark(count: int, workspace: Path) -> dict[str, float | int]:
    root = workspace / str(count)
    service = HistoryService(root / "history.db", root / "outputs")
    records = _records(count)
    started = perf_counter()
    service.repository.upsert_many(records)
    insert_seconds = perf_counter() - started

    started = perf_counter()
    page = service.list_tasks(HistoryQuery(page=1, page_size=50))
    first_page_ms = (perf_counter() - started) * 1000.0

    started = perf_counter()
    filtered = service.list_tasks(
        HistoryQuery(
            task_type=TaskType.BATCH,
            model_version="2.0",
            device="cuda:0",
            page=1,
            page_size=50,
        )
    )
    filtered_page_ms = (perf_counter() - started) * 1000.0

    started = perf_counter()
    searched = service.search_tasks(f"模拟结果 {count - 1}")
    search_ms = (perf_counter() - started) * 1000.0
    return {
        "task_count": count,
        "insert_seconds": round(insert_seconds, 4),
        "first_page_ms": round(first_page_ms, 3),
        "filtered_page_ms": round(filtered_page_ms, 3),
        "search_ms": round(search_ms, 3),
        "first_page_rows": len(page.records),
        "filtered_total": filtered.total,
        "search_total": searched.total,
        "database_bytes": service.database_path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("counts", nargs="*", type=int, default=[100, 1000, 10_000])
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="nss-history-benchmark-") as directory:
        results = [benchmark(count, Path(directory)) for count in arguments.counts]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
