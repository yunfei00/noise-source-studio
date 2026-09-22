"""Shared cooperative pause/resume/stop control for sequential workers."""

from __future__ import annotations

from collections.abc import Callable
from threading import Condition

from noise_source_studio.common.exceptions import (
    InferenceEngineNotConfiguredError,
    PredictionBusyError,
)


class CooperativeWorkerControl:
    """Pause only at item boundaries and stop after the active item returns."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._pause_requested = False
        self._stop_requested = False

    @property
    def stop_requested(self) -> bool:
        with self._condition:
            return self._stop_requested

    def request_pause(self) -> None:
        with self._condition:
            self._pause_requested = True

    def request_resume(self) -> None:
        with self._condition:
            self._pause_requested = False
            self._condition.notify_all()

    def request_stop(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._pause_requested = False
            self._condition.notify_all()

    def wait_at_boundary(
        self,
        on_paused: Callable[[], None],
        on_resumed: Callable[[], None],
    ) -> bool:
        """Wait cooperatively and return whether execution should stop."""
        with self._condition:
            if self._stop_requested:
                return True
            if self._pause_requested:
                on_paused()
                while self._pause_requested and not self._stop_requested:
                    self._condition.wait()
                if self._stop_requested:
                    return True
                on_resumed()
            return self._stop_requested


def is_fatal_inference_error(exc: Exception) -> bool:
    """Classify session/device failures that invalidate the complete task."""
    if isinstance(exc, (InferenceEngineNotConfiguredError, PredictionBusyError)):
        return True
    message = str(exc).casefold()
    return any(
        marker in message
        for marker in (
            "session closed",
            "会话已关闭",
            "没有已加载",
            "没有已激活",
            "cuda device",
            "cuda 设备",
            "model state corrupted",
            "模型内部状态",
        )
    )


__all__ = ["CooperativeWorkerControl", "is_fatal_inference_error"]
