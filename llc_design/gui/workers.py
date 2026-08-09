"""Reusable QRunnable wrapper for non-blocking engineering calculations."""

from __future__ import annotations

import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    finished = Signal()
    result = Signal(object)
    error = Signal(str)
    progress = Signal(float, str)


class FunctionWorker(QRunnable):
    """QRunnable whose signals stay alive until emission completes.

    The thread pool auto-deletes the QRunnable after run(), so callers must
    hold a strong reference to this object until `finished` fires; otherwise
    the signals QObject can be garbage-collected mid-emission and the result
    callback silently never runs ("Signal source has been deleted").
    """

    def __init__(self, function: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self.setAutoDelete(False)
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.function(*self.args, **self.kwargs)
        except Exception:
            self.signals.error.emit(traceback.format_exc())
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
