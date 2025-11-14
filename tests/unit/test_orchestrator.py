import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.main import Orchestrator


@pytest.fixture
def orchestrator(monkeypatch, orchestrator_paths, light_model_configs):
    monkeypatch.setattr("src.main.MODEL_CONFIGS", light_model_configs)
    return Orchestrator(
        cleaned_train_csv=str(orchestrator_paths["cleaned_train_csv"]),
        cleaned_test_csv=str(orchestrator_paths["cleaned_test_csv"]),
        models_dir=str(orchestrator_paths["models_dir"]),
        metrics_dir=str(orchestrator_paths["metrics_dir"]),
        reports_dir=str(orchestrator_paths["reports_dir"]),
        test_size=0.2,
        random_state=0,
    )


def test_run_stage_data(orchestrator, orchestrator_paths):
    print("\nTEST_ORCHESTRATOR_RUN_STAGE_DATA STARTED")
    raw_csv = orchestrator_paths["raw_csv"]
    orchestrator.run(stage="data", csv=str(raw_csv), target="cnt")
    assert Path(orchestrator.cleaned_train_csv).exists()
    assert Path(orchestrator.cleaned_test_csv).exists()
    print("TEST_ORCHESTRATOR_RUN_STAGE_DATA PASSED")


def test_run_stage_train(orchestrator, mock_mlflow, tmp_path):
    print("\nTEST_ORCHESTRATOR_RUN_STAGE_TRAIN STARTED")
    orchestrator.run(stage="train", cleaned_train_csv=orchestrator.cleaned_train_csv, target="cnt")
    saved_models = list(Path(orchestrator.models_dir).glob("*.pkl"))
    assert saved_models
    print("TEST_ORCHESTRATOR_RUN_STAGE_TRAIN PASSED")


def test_run_stage_evaluate(orchestrator, mock_mlflow, tmp_path):
    print("\nTEST_ORCHESTRATOR_RUN_STAGE_EVALUATE STARTED")
    orchestrator.run(stage="train", cleaned_train_csv=orchestrator.cleaned_train_csv, target="cnt")
    orchestrator.run(stage="evaluate", cleaned_test_csv=orchestrator.cleaned_test_csv, target="cnt")
    json_metrics = list(Path(orchestrator.metrics_dir).glob("*_test_results.json"))
    assert json_metrics
    print("TEST_ORCHESTRATOR_RUN_STAGE_EVALUATE PASSED")
