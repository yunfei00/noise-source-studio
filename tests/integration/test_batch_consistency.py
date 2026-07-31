"""Opt-in three-file real-runtime consistency test."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from noise_source_studio.domain.batch import BatchFileItem, BatchPredictionTask
from noise_source_studio.infrastructure.inference import BatchWorker, RuntimeAdapter
from noise_source_studio.services.result_adapter import normalize_prediction_result

FIELDS = (
    "labels",
    "decision_mode",
    "multilabel_probabilities",
    "label_marginal_probabilities",
    "combination_probabilities",
    "decoded_label_vector",
    "predicted_combination",
    "predicted_sources",
    "input_shape",
)


@pytest.mark.integration
def test_three_real_files_match_direct_runtime(tmp_path: Path) -> None:
    package_text = os.environ.get("NOISE_STUDIO_MODEL_PACKAGE")
    csv_directory_text = os.environ.get("NOISE_STUDIO_BATCH_CSV_DIR")
    if not package_text or not csv_directory_text:
        pytest.skip("未配置真实批量模型和 CSV 环境变量")
    package = Path(package_text)
    files = sorted(Path(csv_directory_text).rglob("*.csv"))[:3]
    if len(files) < 3:
        pytest.skip("真实批量目录中不足三个 CSV")

    engine = RuntimeAdapter()
    engine.load_model(package, device="cpu")
    expected = [normalize_prediction_result(engine.predict_file(path)) for path in files]
    task = BatchPredictionTask("integration", tmp_path)
    task.items = [BatchFileItem(index, path) for index, path in enumerate(files, 1)]
    task.refresh_counts()
    BatchWorker(engine, task).run()

    assert task.success_count == 3
    for direct, item in zip(expected, task.items, strict=True):
        assert item.result is not None
        for field in FIELDS:
            if "probabilities" in field:
                assert item.result.get(field) == pytest.approx(direct.get(field))
            else:
                assert item.result.get(field) == direct.get(field)
    assert engine.model_load_count == 1
    engine.close()
