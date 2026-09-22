"""Parameterized SQLite repository for the unified history index."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import fields
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from noise_source_studio.domain.history import (
    HistoryArtifact,
    HistoryOverview,
    HistoryPageResult,
    HistoryQuery,
    HistoryStatus,
    IntegrityStatus,
    TaskHistoryRecord,
    TaskType,
)
from noise_source_studio.infrastructure.history.database import HistoryDatabase

SORT_COLUMNS = {
    "task_id",
    "task_type",
    "task_name",
    "status",
    "created_at",
    "finished_at",
    "duration_ms",
    "model_name",
    "device",
    "total_count",
    "success_count",
    "failed_count",
    "integrity_status",
}


class HistoryRepository:
    """Persist and query compact task metadata without retaining SQLite handles."""

    def __init__(self, database: HistoryDatabase) -> None:
        self.database = database

    def initialize(self) -> int:
        return self.database.initialize()

    def upsert_task(
        self,
        record: TaskHistoryRecord,
        artifacts: Iterable[HistoryArtifact] = (),
    ) -> None:
        """Atomically insert or replace task metadata and supplied artifacts."""
        with self.database.connect() as connection:
            self._upsert_record(connection, record)
            for artifact in artifacts:
                self._upsert_artifact(connection, record.task_id, artifact)

    def upsert_many(self, records: Iterable[TaskHistoryRecord]) -> None:
        """Bulk upsert records in one transaction for rebuild and performance tests."""
        with self.database.connect() as connection:
            for record in records:
                self._upsert_record(connection, record)

    def get_task(self, task_id: str) -> TaskHistoryRecord | None:
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM history_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return self._record_from_row(row) if row is not None else None

    def list_tasks(self, query: HistoryQuery) -> HistoryPageResult:
        where, parameters = self._query_where(query)
        page_size = min(200, max(1, query.page_size))
        page = max(1, query.page)
        sort_by = query.sort_by if query.sort_by in SORT_COLUMNS else "created_at"
        direction = "DESC" if query.descending else "ASC"
        with self.database.connect() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM history_tasks {where}",  # noqa: S608
                    parameters,
                ).fetchone()[0]
            )
            max_page = max(1, (total + page_size - 1) // page_size)
            page = min(page, max_page)
            rows = connection.execute(
                f"""SELECT * FROM history_tasks {where}
                ORDER BY {sort_by} {direction}, task_id ASC LIMIT ? OFFSET ?""",  # noqa: S608
                (*parameters, page_size, (page - 1) * page_size),
            ).fetchall()
        return HistoryPageResult(
            records=tuple(self._record_from_row(row) for row in rows),
            total=total,
            page=page,
            page_size=page_size,
        )

    def list_all(self, query: HistoryQuery) -> tuple[TaskHistoryRecord, ...]:
        """Return every matching row for explicit CSV summary export."""
        where, parameters = self._query_where(query)
        sort_by = query.sort_by if query.sort_by in SORT_COLUMNS else "created_at"
        direction = "DESC" if query.descending else "ASC"
        with self.database.connect() as connection:
            rows = connection.execute(
                f"""SELECT * FROM history_tasks {where}
                ORDER BY {sort_by} {direction}, task_id ASC""",  # noqa: S608
                parameters,
            ).fetchall()
        return tuple(self._record_from_row(row) for row in rows)

    def overview(self) -> HistoryOverview:
        since = (datetime.now(UTC) - timedelta(days=7)).isoformat()
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS total,
                    SUM(CASE WHEN status IN ('completed', 'completed_with_errors')
                        THEN 1 ELSE 0 END) AS completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
                    SUM(CASE WHEN status = 'interrupted' THEN 1 ELSE 0 END) AS interrupted,
                    SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS last_seven_days,
                    SUM(CASE WHEN integrity_status IN ('missing', 'partial', 'corrupt')
                        THEN 1 ELSE 0 END) AS missing_artifacts
                FROM history_tasks
                """,
                (since,),
            ).fetchone()
        return HistoryOverview(*(int(value or 0) for value in row))

    def update_task(self, task_id: str, **changes: Any) -> bool:
        """Update a strict whitelist of fields with bound SQL parameters."""
        allowed = {
            field.name for field in fields(TaskHistoryRecord)
        } - {"task_id", "task_type"}
        normalized: dict[str, Any] = {}
        for key, value in changes.items():
            if key not in allowed:
                raise ValueError(f"Unsupported history field: {key}")
            column = "tags_json" if key == "tags" else key
            normalized[column] = self._database_value(key, value)
        if not normalized:
            return False
        assignments = ", ".join(f"{column} = ?" for column in normalized)
        with self.database.connect() as connection:
            cursor = connection.execute(
                f"UPDATE history_tasks SET {assignments} WHERE task_id = ?",  # noqa: S608
                (*normalized.values(), task_id),
            )
        return cursor.rowcount > 0

    def delete_task(self, task_id: str) -> bool:
        with self.database.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM history_tasks WHERE task_id = ?", (task_id,)
            )
        return cursor.rowcount > 0

    def interrupt_running(self, finished_at: str) -> int:
        with self.database.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE history_tasks
                SET status = 'interrupted', finished_at = ?,
                    primary_summary = CASE WHEN primary_summary = ''
                        THEN '上次应用退出前任务尚未正常结束。'
                        ELSE primary_summary END
                WHERE status = 'running'
                """,
                (finished_at,),
            )
        return cursor.rowcount

    def artifacts(self, task_id: str) -> tuple[HistoryArtifact, ...]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM history_artifacts WHERE task_id = ? ORDER BY artifact_type",
                (task_id,),
            ).fetchall()
        return tuple(
            HistoryArtifact(
                artifact_id=str(row["artifact_id"]),
                artifact_type=str(row["artifact_type"]),
                path=str(row["relative_path"]),
                file_size=row["file_size"],
                sha256=str(row["sha256"]),
                created_at=str(row["created_at"]),
                exists_status=str(row["exists_status"]),
            )
            for row in rows
        )

    @staticmethod
    def _upsert_record(
        connection: sqlite3.Connection,
        record: TaskHistoryRecord,
    ) -> None:
        values = HistoryRepository._record_values(record)
        columns = tuple(values)
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(
            f"{column}=excluded.{column}" for column in columns if column != "task_id"
        )
        connection.execute(
            f"""INSERT INTO history_tasks ({', '.join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(task_id) DO UPDATE SET {updates}""",  # noqa: S608
            tuple(values.values()),
        )

    @staticmethod
    def _upsert_artifact(
        connection: sqlite3.Connection,
        task_id: str,
        artifact: HistoryArtifact,
    ) -> None:
        artifact_id = artifact.artifact_id or uuid4().hex
        connection.execute(
            """
            INSERT INTO history_artifacts(
                artifact_id, task_id, artifact_type, relative_path, file_size,
                sha256, created_at, exists_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id, artifact_type, relative_path) DO UPDATE SET
                file_size=excluded.file_size, sha256=excluded.sha256,
                created_at=excluded.created_at, exists_status=excluded.exists_status
            """,
            (
                artifact_id,
                task_id,
                artifact.artifact_type,
                artifact.path,
                artifact.file_size,
                artifact.sha256,
                artifact.created_at,
                artifact.exists_status,
            ),
        )

    @staticmethod
    def _record_values(record: TaskHistoryRecord) -> dict[str, Any]:
        return {
            field.name if field.name != "tags" else "tags_json":
                HistoryRepository._database_value(field.name, getattr(record, field.name))
            for field in fields(TaskHistoryRecord)
        }

    @staticmethod
    def _database_value(name: str, value: Any) -> Any:
        if name == "tags":
            return json.dumps(list(value), ensure_ascii=False)
        if isinstance(value, (TaskType, HistoryStatus, IntegrityStatus)):
            return value.value
        if isinstance(value, bool):
            return int(value)
        return value

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> TaskHistoryRecord:
        return TaskHistoryRecord(
            **{
                field.name: HistoryRepository._row_value(field.name, row)
                for field in fields(TaskHistoryRecord)
            }
        )

    @staticmethod
    def _row_value(name: str, row: sqlite3.Row) -> Any:
        if name == "task_type":
            return TaskType(row[name])
        if name == "status":
            return HistoryStatus(row[name])
        if name == "integrity_status":
            return IntegrityStatus(row[name])
        if name == "tags":
            try:
                return tuple(str(value) for value in json.loads(row["tags_json"]))
            except (TypeError, json.JSONDecodeError):
                return ()
        if name in {"created_by_application", "imported_from_legacy"}:
            return bool(row[name])
        return row[name]

    @staticmethod
    def _query_where(query: HistoryQuery) -> tuple[str, tuple[Any, ...]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if query.task_type is not None:
            clauses.append("task_type = ?")
            parameters.append(query.task_type.value)
        if query.status is not None:
            clauses.append("status = ?")
            parameters.append(query.status.value)
        if query.date_from:
            clauses.append("created_at >= ?")
            parameters.append(query.date_from)
        if query.date_to:
            clauses.append("created_at <= ?")
            parameters.append(query.date_to)
        for column, value in (
            ("model_name", query.model_name),
            ("model_version", query.model_version),
            ("device", query.device),
        ):
            if value:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if query.failed_only:
            clauses.append("(status IN ('failed', 'completed_with_errors') OR failed_count > 0)")
        if query.missing_only:
            clauses.append("integrity_status IN ('missing', 'partial', 'corrupt')")
        if query.keyword.strip():
            token = f"%{query.keyword.strip()}%"
            clauses.append(
                """(task_id LIKE ? OR task_name LIKE ? OR source_description LIKE ?
                OR source_path LIKE ? OR model_name LIKE ? OR model_version LIKE ?
                OR primary_summary LIKE ? OR notes LIKE ? OR tags_json LIKE ?)"""
            )
            parameters.extend([token] * 9)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, tuple(parameters)


__all__ = ["HistoryRepository"]
