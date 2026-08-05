"""SQLite connection, migration, and backup management."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from noise_source_studio.infrastructure.history.migrations import MIGRATIONS, SCHEMA_VERSION

LOGGER = logging.getLogger("noise_source_studio.history_database")


class HistoryDatabaseError(RuntimeError):
    """Raised when the task-history index cannot be safely used."""


class HistorySchemaTooNewError(HistoryDatabaseError):
    """Raised when a newer application has already upgraded the index."""


class _ClosingConnection(sqlite3.Connection):
    """Commit or roll back like sqlite3's context manager, then always close."""

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc, traceback))
        finally:
            self.close()


class HistoryDatabase:
    """Own short-lived, consistently configured SQLite connections."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        """Open one configured connection; callers close it promptly."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path,
            timeout=5.0,
            factory=_ClosingConnection,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.execute("PRAGMA busy_timeout = 5000")
        except Exception:
            connection.close()
            raise
        return connection

    def initialize(self) -> int:
        """Apply pending migrations, backing up an older index first."""
        try:
            with self.connect() as connection:
                current = self._schema_version(connection)
            if current > SCHEMA_VERSION:
                raise HistorySchemaTooNewError(
                    f"历史数据库版本 {current} 高于当前支持版本 {SCHEMA_VERSION}。"
                )
            if current and current < SCHEMA_VERSION:
                backup = self.backup("pre-migration")
                LOGGER.info("History pre-migration backup created | path=%s", backup)
            for version in range(current + 1, SCHEMA_VERSION + 1):
                with self.connect() as connection:
                    MIGRATIONS[version](connection)
                    connection.execute(
                        """
                        INSERT INTO schema_metadata(key, value, updated_at)
                        VALUES('schema_version', ?, ?)
                        ON CONFLICT(key) DO UPDATE SET
                            value=excluded.value, updated_at=excluded.updated_at
                        """,
                        (str(version), datetime.now(UTC).isoformat()),
                    )
                LOGGER.info("History schema migration applied | version=%s", version)
            return SCHEMA_VERSION
        except HistoryDatabaseError:
            raise
        except sqlite3.Error as exc:
            raise HistoryDatabaseError(f"无法初始化历史数据库：{exc}") from exc

    def backup(self, reason: str = "manual") -> Path:
        """Create a transactionally consistent SQLite backup."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        safe_reason = "".join(char for char in reason if char.isalnum() or char in "_-")
        backup_path = self.path.parent / f"history-{timestamp}-{safe_reason or 'backup'}.db"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.connect() as source, sqlite3.connect(
                backup_path,
                factory=_ClosingConnection,
            ) as destination:
                source.backup(destination)
        except sqlite3.Error as exc:
            backup_path.unlink(missing_ok=True)
            raise HistoryDatabaseError(f"无法备份历史数据库：{exc}") from exc
        return backup_path

    @staticmethod
    def _schema_version(connection: sqlite3.Connection) -> int:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_metadata'"
        ).fetchone()
        if exists is None:
            return 0
        row = connection.execute(
            "SELECT value FROM schema_metadata WHERE key='schema_version'"
        ).fetchone()
        return int(row[0]) if row is not None else 0


__all__ = ["HistoryDatabase", "HistoryDatabaseError", "HistorySchemaTooNewError"]
