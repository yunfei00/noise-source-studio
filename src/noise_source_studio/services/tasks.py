"""Small QRunnable wrapper for all model and inference work."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


@dataclass(frozen=True, slots=True)
class TaskFailure:
    """Exception and formatted traceback emitted back to the GUI thread."""

    exception: Exception
    traceback_text: str


class TaskSignals(QObject):
    """Signals shared by every background task."""

    progress = Signal(int, str)
    succeeded = Signal(object)
    failed = Signal(object)
    finished = Signal()


class BackgroundTask(QRunnable):
    """Execute a callable off the GUI thread and report a uniform lifecycle."""

    def __init__(self, operation: Callable[[], Any], description: str) -> None:
        super().__init__()
        self.task_id = uuid4().hex
        self.operation = operation
        self.description = description
        self.signals = TaskSignals()

    @Slot()
    def run(self) -> None:
        """Run the operation and never leak an exception across Qt."""
        self.signals.progress.emit(0, self.description)
        try:
            result = self.operation()
        except Exception as exc:
            self.signals.failed.emit(
                TaskFailure(
                    exception=exc,
                    traceback_text=traceback.format_exc(),
                )
            )
        else:
            self.signals.progress.emit(100, self.description)
            self.signals.succeeded.emit(result)
        finally:
            self.signals.finished.emit()


__all__ = ["BackgroundTask", "TaskFailure", "TaskSignals"]
