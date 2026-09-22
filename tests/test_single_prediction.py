"""Single-file page rendering, worker and export tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PySide6.QtCore import QThread, QThreadPool
from PySide6.QtWidgets import QApplication, QProgressBar
from pytestqt.qtbot import QtBot

from noise_source_studio.domain.models import (
    LoadedModel,
    ModelRecord,
    PredictionOutcome,
    SignalPreview,
)
from noise_source_studio.presentation.pages.single_prediction_page import (
    SinglePredictionPage,
)
from noise_source_studio.services import PredictionService
from noise_source_studio.services.tasks import BackgroundTask


def _record() -> ModelRecord:
    return ModelRecord(
        model_name="dynamic-model",
        model_version="0.2.0",
        package_path=Path("models/dynamic-model/0.2.0"),
        manifest={"labels": ["a", "b", "c", "d"], "prediction_mode": "structured"},
        installed_at="2026-07-30T00:00:00Z",
        is_active=True,
        integrity_status="valid",
    )


def _loaded(mode: str = "structured") -> LoadedModel:
    record = _record()
    return LoadedModel(
        record=record,
        runtime_version="1.0.0",
        device="cpu",
        prediction_mode=mode,
        labels=("a", "b", "c", "d"),
        inspection={},
    )


def _outcome(source: Path, result: Any) -> PredictionOutcome:
    now = datetime.now(UTC)
    return PredictionOutcome(
        task_id="task-1",
        source_path=source,
        model=_record(),
        started_at=now,
        completed_at=now,
        duration_seconds=0.125,
        result=result,
    )


def _preview(source: Path) -> SignalPreview:
    return SignalPreview(
        source_path=source,
        display_values=(1.0, 2.0, 3.0),
        original_point_count=3,
        raw_minimum=1.0,
        raw_maximum=3.0,
        raw_mean=2.0,
        raw_standard_deviation=0.816,
        parser_mode="strict_data",
        data_start_line=4,
        selected_columns=(1,),
        encoding="utf-8-sig",
        delimiter="comma",
    )


def test_unconfigured_model_disables_prediction(
    qapp: QApplication,
    qtbot: QtBot,
) -> None:
    page = SinglePredictionPage()
    qtbot.addWidget(page)

    assert not page.predict_button.isEnabled()
    assert "激活" in page.engine_message.text()


def test_file_selection_displays_metadata_and_clears_result(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    source = tmp_path / "signal.csv"
    source.write_text("DATA\n0,1\n", encoding="utf-8")
    page = SinglePredictionPage()
    qtbot.addWidget(page)

    with qtbot.waitSignal(page.preview_requested):
        page.set_selected_file(source)

    assert page.file_name_label.text() == "signal.csv"
    assert page.file_path_label.text() == str(source)
    assert page.file_size_label.text() != "—"
    assert page.current_outcome is None


def test_structured_mode_uses_marginals_and_decoded_vector(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    source = tmp_path / "signal.csv"
    source.write_text("DATA\n0,1\n", encoding="utf-8")
    page = SinglePredictionPage()
    qtbot.addWidget(page)
    page.set_model(_loaded())
    page.set_selected_file(source)
    page.show_preview(_preview(source))
    result = SimpleNamespace(
        runtime_version="1.0.0",
        device="cpu",
        labels=["a", "b", "c", "d"],
        decision_mode="structured",
        multilabel_probabilities=[0.9, 0.8, 0.7, 0.6],
        combination_labels=["0000", "1010"],
        combination_probabilities=[0.1, 0.9],
        label_marginal_probabilities=[0.2, 0.3, 0.4, 0.5],
        decoded_label_vector=[1, 0, 1, 0],
        predicted_combination="1010",
        predicted_sources=["a", "c"],
        thresholds=[0.5] * 4,
        thresholds_applicable=False,
        auxiliary_logits={"combo": [1.0, 2.0]},
        input_shape=[1, 1, 128, 64],
    )

    page.show_prediction(_outcome(source, result))

    bars = page.result_content.findChildren(QProgressBar)
    assert [bar.value() for bar in bars] == [2000, 3000, 4000, 5000]
    assert page.final_vector_label.text() == "[1, 0, 1, 0]"
    assert page.highest_combination_label.text().startswith("1010")
    assert "argmax" in page.detail_view.toPlainText()


def test_multilabel_mode_uses_probabilities_thresholds_and_dynamic_labels(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    source = tmp_path / "signal.csv"
    source.write_text("DATA\n0,1\n", encoding="utf-8")
    page = SinglePredictionPage()
    qtbot.addWidget(page)
    page.set_model(_loaded("multilabel"))
    page.set_selected_file(source)
    page.show_preview(_preview(source))
    result = SimpleNamespace(
        runtime_version="1.0.0",
        device="cpu",
        labels=["x", "y"],
        decision_mode="multilabel",
        multilabel_probabilities=[0.25, 0.75],
        combination_labels=[],
        combination_probabilities=None,
        label_marginal_probabilities=[0.99, 0.01],
        decoded_label_vector=[0, 1],
        predicted_combination="01",
        predicted_sources=["y"],
        thresholds=[0.4, 0.6],
        thresholds_applicable=True,
        auxiliary_logits=None,
        input_shape=[1, 1, 128, 64],
    )

    page.show_prediction(_outcome(source, result))

    bars = page.result_content.findChildren(QProgressBar)
    assert [bar.value() for bar in bars] == [2500, 7500]
    details = page.detail_view.toPlainText()
    assert '"thresholds": [' in details
    assert "大于等于对应阈值" in details
    assert len(bars) == 2


def test_prediction_error_restores_controls(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    source = tmp_path / "signal.csv"
    source.write_text("DATA\n0,1\n", encoding="utf-8")
    page = SinglePredictionPage()
    qtbot.addWidget(page)
    page.set_model(_loaded())
    page.set_selected_file(source)
    page.show_preview(_preview(source))

    page.begin_prediction()
    page.show_prediction_error("推理失败")

    assert page.predict_button.text() == "开始预测"
    assert page.predict_button.isEnabled()
    assert page.drop_zone.isEnabled()


def test_background_task_runs_outside_gui_thread(
    qapp: QApplication,
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    class ThreadCaptureEngine:
        def __init__(self) -> None:
            self.worker_thread: QThread | None = None

        def predict_file(self, source: Path) -> SimpleNamespace:
            self.worker_thread = QThread.currentThread()
            return SimpleNamespace(
                runtime_version="1.0.0",
                device="cpu",
                decision_mode="multilabel",
                input_shape=[1, 1, 128, 64],
            )

    engine = ThreadCaptureEngine()
    service = PredictionService(engine, tmp_path)  # type: ignore[arg-type]
    pool = QThreadPool()
    task = BackgroundTask(
        lambda: service.predict_file(tmp_path / "signal.csv", _record()),
        "prediction-thread-check",
    )

    with qtbot.waitSignal(task.signals.succeeded) as blocker:
        pool.start(task)

    assert isinstance(blocker.args[0], PredictionOutcome)
    assert engine.worker_thread is not None
    assert engine.worker_thread is not qapp.thread()
    pool.waitForDone()


class FakeExportEngine:
    runtime_version = "1.0.0"

    def export_result(
        self,
        result: Any,
        json_path: Path,
        *,
        contract_path: Path | None = None,
    ) -> tuple[Path, Path | None]:
        json_path.write_text("{}", encoding="utf-8")
        if contract_path is not None:
            contract_path.write_text("# Contract", encoding="utf-8")
        return json_path, contract_path


def test_explicit_result_export_uses_unique_runtime_paths(tmp_path: Path) -> None:
    service = PredictionService(FakeExportEngine(), tmp_path / "outputs")  # type: ignore[arg-type]
    outcome = _outcome(tmp_path / "signal.csv", SimpleNamespace())

    json_path, contract_path = service.export_result(outcome, include_contract=True)

    assert json_path.is_file()
    assert contract_path is not None and contract_path.is_file()
    assert "signal_" in json_path.name
    assert "_0.2.0_prediction.json" in json_path.name
