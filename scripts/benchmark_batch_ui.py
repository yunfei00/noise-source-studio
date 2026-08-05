"""Measure batch-page load, incremental updates, filtering and final calibration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from noise_source_studio.domain.batch import (  # noqa: E402
    BatchFileItem,
    BatchItemStatus,
    BatchPredictionTask,
    BatchStatus,
)
from noise_source_studio.presentation.pages import BatchPredictionPage  # noqa: E402


def _task(root: Path, count: int) -> BatchPredictionTask:
    root.mkdir(parents=True, exist_ok=True)
    source = root / "shared.csv"
    source.write_text("DATA\n0,1\n", encoding="utf-8")
    task = BatchPredictionTask(f"benchmark-{count}", root / "outputs")
    for sequence in range(1, count + 1):
        item = BatchFileItem(sequence, source)
        item.file_name = f"sample_{sequence}.csv"
        task.items.append(item)
    task.refresh_counts()
    return task


def _complete(task: BatchPredictionTask, item: BatchFileItem) -> None:
    task.transition_item_status(item, BatchItemStatus.SUCCESS)
    item.elapsed_ms = 10.0
    item.labels = ["fan", "motor"]
    item.display_probabilities = [0.8, 0.3]
    item.predicted_combination = "10"
    item.predicted_sources = ["fan"]
    item.result = {
        "decision_mode": "structured",
        "labels": item.labels,
        "label_marginal_probabilities": item.display_probabilities,
        "combination_labels": ["01", "10", "11"],
        "combination_probabilities": [0.1, 0.8, 0.1],
        "predicted_combination": item.predicted_combination,
        "predicted_sources": item.predicted_sources,
    }


def benchmark(root: Path, count: int) -> dict[str, int | float]:
    task = _task(root, count)
    page = BatchPredictionPage()

    started = perf_counter()
    page.set_task(task)
    load_ms = (perf_counter() - started) * 1000.0

    task.status = BatchStatus.RUNNING
    page.begin_batch_updates()
    started = perf_counter()
    for item in task.items:
        _complete(task, item)
        page.refresh_item(item)
    page.flush_pending_ui_updates()
    update_ms = (perf_counter() - started) * 1000.0

    task.status = BatchStatus.COMPLETED
    started = perf_counter()
    page.finalize_batch_updates()
    final_ms = (perf_counter() - started) * 1000.0

    started = perf_counter()
    page.result_proxy.set_filters(keyword=f"sample_{count}.csv")
    filtered_rows = page.result_proxy.rowCount()
    filter_ms = (perf_counter() - started) * 1000.0
    result = {
        "rows": count,
        "load_ms": round(load_ms, 3),
        "incremental_update_ms": round(update_ms, 3),
        "final_calibration_ms": round(final_ms, 3),
        "filter_ms": round(filter_ms, 3),
        "filtered_rows": filtered_rows,
        **page.performance_counters(),
    }
    page.deleteLater()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sizes", nargs="*", type=int, default=[100, 1_000, 10_000])
    arguments = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    with TemporaryDirectory(prefix="noise-source-studio-benchmark-") as directory:
        root = Path(directory)
        results = [benchmark(root / str(size), size) for size in arguments.sizes]
    app.processEvents()
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
