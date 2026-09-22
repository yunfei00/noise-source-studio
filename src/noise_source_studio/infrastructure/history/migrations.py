"""Versioned SQLite schema migrations for task history."""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 1


def migrate_to_v1(connection: sqlite3.Connection) -> None:
    """Create the first production history schema."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS history_tasks (
            task_id TEXT PRIMARY KEY,
            task_type TEXT NOT NULL,
            task_name TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            started_at TEXT NOT NULL DEFAULT '',
            finished_at TEXT NOT NULL DEFAULT '',
            duration_ms REAL NOT NULL DEFAULT 0,
            model_name TEXT NOT NULL DEFAULT '',
            model_version TEXT NOT NULL DEFAULT '',
            model_identifier TEXT NOT NULL DEFAULT '',
            model_package_sha256 TEXT NOT NULL DEFAULT '',
            runtime_version TEXT NOT NULL DEFAULT '',
            application_version TEXT NOT NULL DEFAULT '',
            device TEXT NOT NULL DEFAULT '',
            decision_mode TEXT NOT NULL DEFAULT '',
            total_count INTEGER NOT NULL DEFAULT 0,
            success_count INTEGER NOT NULL DEFAULT 0,
            failed_count INTEGER NOT NULL DEFAULT 0,
            skipped_count INTEGER NOT NULL DEFAULT 0,
            stopped_count INTEGER NOT NULL DEFAULT 0,
            primary_summary TEXT NOT NULL DEFAULT '',
            source_description TEXT NOT NULL DEFAULT '',
            source_path TEXT NOT NULL DEFAULT '',
            result_directory TEXT NOT NULL DEFAULT '',
            task_file TEXT NOT NULL DEFAULT '',
            summary_file TEXT NOT NULL DEFAULT '',
            detail_file TEXT NOT NULL DEFAULT '',
            error_file TEXT NOT NULL DEFAULT '',
            report_file TEXT NOT NULL DEFAULT '',
            manifest_file TEXT NOT NULL DEFAULT '',
            integrity_status TEXT NOT NULL DEFAULT 'unknown',
            error_type TEXT NOT NULL DEFAULT '',
            error_message TEXT NOT NULL DEFAULT '',
            tags_json TEXT NOT NULL DEFAULT '[]',
            notes TEXT NOT NULL DEFAULT '',
            created_by_application INTEGER NOT NULL DEFAULT 1,
            imported_from_legacy INTEGER NOT NULL DEFAULT 0,
            last_opened_at TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS history_artifacts (
            artifact_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            artifact_type TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            file_size INTEGER,
            sha256 TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT '',
            exists_status TEXT NOT NULL DEFAULT 'unknown',
            FOREIGN KEY(task_id) REFERENCES history_tasks(task_id) ON DELETE CASCADE,
            UNIQUE(task_id, artifact_type, relative_path)
        );

        CREATE INDEX IF NOT EXISTS idx_history_tasks_created
            ON history_tasks(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_history_tasks_type_created
            ON history_tasks(task_type, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_history_tasks_status_created
            ON history_tasks(status, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_history_tasks_model
            ON history_tasks(model_name, model_version);
        CREATE INDEX IF NOT EXISTS idx_history_tasks_device
            ON history_tasks(device);
        CREATE INDEX IF NOT EXISTS idx_history_tasks_integrity
            ON history_tasks(integrity_status);
        CREATE INDEX IF NOT EXISTS idx_history_artifacts_task
            ON history_artifacts(task_id);
        """
    )


MIGRATIONS = {1: migrate_to_v1}


__all__ = ["MIGRATIONS", "SCHEMA_VERSION"]
