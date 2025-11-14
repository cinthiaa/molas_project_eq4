import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pickle

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from sklearn.linear_model import LinearRegression

from src.data import DataProcessor
from src.train_predict import Model

NUM_COLS = ["temp", "hum", "windspeed"]
CAT_COLS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
]


@pytest.fixture
def sample_dataframe():
    """DataFrame pequeño con columnas numéricas y categóricas válidas."""
    return pd.DataFrame(
        {
            "season": [1, 2, 3],
            "yr": [0, 1, 0],
            "mnth": [1, 2, 3],
            "hr": [0, 1, 2],
            "holiday": [0, 0, 0],
            "weekday": [1, 2, 3],
            "workingday": [1, 1, 0],
            "weathersit": [1, 2, 2],
            "temp": [12.0, 18.0, 25.0],
            "hum": [0.3, 0.5, 0.7],
            "windspeed": [0.1, 0.2, 0.3],
            "cnt": [120, 150, 130],
        }
    )


@pytest.fixture
def sample_csv(tmp_path, sample_dataframe):
    """Escribe el DataFrame de ejemplo en un CSV temporal y retorna la ruta."""
    csv_path = tmp_path / "sample.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def regression_dataframe():
    X = pd.DataFrame(
        {
            "season": [1, 2, 3, 4],
            "yr": [0, 1, 0, 1],
            "mnth": [1, 2, 3, 4],
            "hr": [0, 1, 2, 3],
            "holiday": [0, 0, 1, 0],
            "weekday": [1, 2, 3, 4],
            "workingday": [1, 1, 0, 0],
            "weathersit": [1, 2, 2, 3],
            "temp": [12.0, 18.0, 25.0, 30.0],
            "hum": [0.3, 0.5, 0.7, 0.65],
            "windspeed": [0.1, 0.2, 0.3, 0.15],
        }
    )
    y = pd.Series([120, 150, 130, 200], name="cnt")
    return X, y


@pytest.fixture
def data_processor():
    processor = DataProcessor(
        numerical_var_cols=NUM_COLS,
        categorical_var_cols=CAT_COLS,
    )
    processor.build()
    return processor


@pytest.fixture
def linear_model():
    return LinearRegression()


@pytest.fixture
def trained_model(tmp_path, regression_dataframe, data_processor, linear_model):
    X, y = regression_dataframe
    model = Model(
        name="linear_regressor",
        estimator=linear_model,
        param_grid={"model__fit_intercept": [True]},
        preprocessor=data_processor.column_transformer,
        description="Fixture linear model",
    )
    model.build_pipeline()
    model.fit(X, y, cv=2, n_jobs=1, verbose=0)

    model_path = tmp_path / "linear_regressor.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model.best_estimator_, f)

    return model, model_path


@pytest.fixture
def orchestrator_paths(tmp_path):
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    raw_dir = data_dir / "raw"
    models_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"
    reports_dir = tmp_path / "reports"

    for d in (processed_dir, raw_dir, models_dir, metrics_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    raw_df = pd.DataFrame(
        {
            "instant": list(range(1, 9)),
            "dteday": [
                "2011-01-01",
                "2011-01-02",
                "2011-01-03",
                "2011-01-04",
                "2011-01-05",
                "2011-01-06",
                "2011-01-07",
                "2011-01-08",
            ],
            "season": [1, 2, 3, 4, 1, 2, 3, 4],
            "yr": [0, 0, 0, 0, 1, 1, 1, 1],
            "mnth": [1, 1, 1, 1, 2, 2, 2, 2],
            "hr": [0, 1, 2, 3, 4, 5, 6, 7],
            "holiday": [0, 0, 0, 0, 0, 0, 0, 0],
            "weekday": [6, 0, 1, 2, 3, 4, 5, 6],
            "workingday": [0, 0, 1, 1, 1, 1, 0, 0],
            "weathersit": [1, 2, 2, 3, 1, 2, 3, 1],
            "temp": [9.84, 14.76, 16.66, 18.84, 20.5, 22.14, 23.0, 24.5],
            "atemp": [14.395, 18.182, 20.454, 22.364, 24.0, 25.5, 26.0, 27.5],
            "hum": [0.81, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4],
            "windspeed": [0.0, 0.0, 0.1, 0.1, 0.12, 0.15, 0.2, 0.25],
            "casual": [3, 8, 12, 15, 18, 20, 22, 25],
            "registered": [13, 32, 45, 50, 60, 65, 70, 75],
            "cnt": [16, 40, 57, 65, 78, 85, 92, 100],
            "mixed_type_col": ["X", "Y", "Z", "X", "Y", "Z", "X", "Y"],
        }
    )
    raw_csv = raw_dir / "raw.csv"
    raw_df.to_csv(raw_csv, index=False)

    train_df = raw_df.drop(columns=["instant", "dteday", "casual", "registered", "atemp", "mixed_type_col"])
    test_df = train_df.copy()

    train_csv = processed_dir / "cleaned_train.csv"
    test_csv = processed_dir / "cleaned_test.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return {
        "raw_csv": raw_csv,
        "cleaned_train_csv": train_csv,
        "cleaned_test_csv": test_csv,
        "models_dir": models_dir,
        "metrics_dir": metrics_dir,
        "reports_dir": reports_dir,
    }


@pytest.fixture
def light_model_configs():
    return {
        "linear_only": {
            "estimator": LinearRegression(),
            "params": {"model__fit_intercept": [True]},
            "description": "Ligero para tests",
        }
    }


@pytest.fixture
def mock_mlflow():
    with patch("src.main.mlflow.start_run") as start_run, patch(
        "src.main.mlflow.log_artifact"
    ) as log_artifact, patch(
        "src.main.mlflow.log_artifacts"
    ) as log_artifacts, patch(
        "src.main.mlflow.sklearn.log_model"
    ) as log_model:
        start_run.return_value.__enter__.return_value = MagicMock()
        yield {
            "start_run": start_run,
            "log_artifact": log_artifact,
            "log_artifacts": log_artifacts,
            "log_model": log_model,
        }
